"""GPT-style decoder, written from scratch.

Constraint (workplan 1.2): no `nn.Transformer*`, no `nn.MultiheadAttention`, and no
fused scaled-dot-product-attention kernel. Every piece below - the QKV projection,
the head split/merge, the causal mask, the softmax attention, the FFN and the
residual wiring - is explicit so it can be inspected and defended.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask.

    Shapes: x (B, T, C) -> (B, T, C), with C = n_head * head_dim.
    The mask is a lower-triangular buffer; scores above the diagonal are set to
    -inf BEFORE softmax, so position i can only attend to j <= i.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 dropout: float, bias: bool = True) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # One projection producing Q, K and V, then split - cheaper than three matmuls.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale          # (B, nh, T, T)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                            # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)       # merge heads
        y = self.resid_dropout(self.proj(y))
        return (y, att) if return_attn else y


class FeedForward(nn.Module):
    """Position-wise FFN, 4x inner width as in Attention Is All You Need."""

    def __init__(self, n_embd: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + ffn(ln(x)).

    Pre-LN (rather than the original post-LN) keeps the residual path clean, which
    is what makes a small model trainable without a long warm-up.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int,
                 dropout: float, bias: bool = True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """Character-level GPT: learned token + position embeddings, N blocks, LM head."""

    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_embd: int,
                 block_size: int, dropout: float = 0.1, bias: bool = True,
                 tie_weights: bool = False) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)   # learnable, not sinusoidal
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale residual-path projections by 1/sqrt(2*n_layer) (GPT-2 init) so the
        # residual stream does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, greedy: bool = False,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive sampling. greedy=True ignores temperature (argmax)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                      # last position only
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k is not None:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
