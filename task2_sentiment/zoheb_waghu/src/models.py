"""Three sentiment classifiers. Embeddings are nn.Embedding initialised randomly
and learned during training - no pretrained vectors, no pretrained LM anywhere.

M1 bilstm_mean       (baseline)    sequential encoder, mean pooling
M2 cnn_multikernel   (experiment)  parallel n-gram convolutions, max pooling
M3 bilstm_attention  (experiment)  deeper/wider baseline + learned token weighting
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Base(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        nn.init.uniform_(self.emb.weight, -0.1, 0.1)
        with torch.no_grad():
            self.emb.weight[padding_idx].fill_(0)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BiLSTMMean(_Base):
    """M1 baseline. Mean pool over real (non-pad) timesteps only - padding must
    not dilute the representation of short reviews."""

    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers,
                 dropout, classifier_hidden=0, **_) -> None:
        super().__init__(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        out_dim = hidden_size * 2
        self.head = (nn.Linear(out_dim, 1) if not classifier_hidden else
                     nn.Sequential(nn.Linear(out_dim, classifier_hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(classifier_hidden, 1)))

    def forward(self, x, lengths=None):
        mask = (x != 0).unsqueeze(-1).float()
        h, _ = self.lstm(self.emb(x))
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.drop(pooled)).squeeze(-1)


class TextCNN(_Base):
    """M2. Each kernel width is an n-gram detector; global max pooling keeps the
    strongest activation of each filter regardless of where in the review it fired."""

    def __init__(self, vocab_size, emb_dim, kernel_sizes, num_filters,
                 dropout, classifier_hidden=0, **_) -> None:
        super().__init__(vocab_size, emb_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        out_dim = num_filters * len(kernel_sizes)
        self.head = (nn.Linear(out_dim, 1) if not classifier_hidden else
                     nn.Sequential(nn.Linear(out_dim, classifier_hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(classifier_hidden, 1)))

    def forward(self, x, lengths=None):
        e = self.emb(x).transpose(1, 2)                   # (B, C, T)
        feats = [F.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.head(self.drop(torch.cat(feats, dim=1))).squeeze(-1)


class BiLSTMAttention(_Base):
    """M3. Additive (Bahdanau-style) attention pooling: the model learns which
    tokens to weight instead of averaging them. Written out rather than using
    nn.MultiheadAttention so the scoring is explicit."""

    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, dropout,
                 attention_dim=128, classifier_hidden=0, **_) -> None:
        super().__init__(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * 2
        self.att_proj = nn.Linear(out_dim, attention_dim)
        self.att_vec = nn.Linear(attention_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = (nn.Linear(out_dim, 1) if not classifier_hidden else
                     nn.Sequential(nn.Linear(out_dim, classifier_hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(classifier_hidden, 1)))

    def forward(self, x, lengths=None, return_attn=False):
        pad_mask = (x == 0)
        h, _ = self.lstm(self.emb(x))
        scores = self.att_vec(torch.tanh(self.att_proj(h))).squeeze(-1)   # (B, T)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        w = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(1)
        logit = self.head(self.drop(pooled)).squeeze(-1)
        return (logit, w.squeeze(-1)) if return_attn else logit


def build_model(cfg: dict, vocab_size: int) -> nn.Module:
    m, e = cfg["model"], cfg["embedding"]
    if e["source"] != "from_scratch":
        raise ValueError("workplan 2.0 forbids pretrained embeddings")
    kw = dict(vocab_size=vocab_size, emb_dim=e["dim"], dropout=m["dropout"],
              classifier_hidden=m.get("classifier_hidden", 0))
    name = m["name"]
    if name == "bilstm_mean":
        return BiLSTMMean(hidden_size=m["hidden_size"], num_layers=m["num_layers"], **kw)
    if name == "cnn_multikernel":
        return TextCNN(kernel_sizes=m["kernel_sizes"], num_filters=m["num_filters"], **kw)
    if name == "bilstm_attention":
        return BiLSTMAttention(hidden_size=m["hidden_size"], num_layers=m["num_layers"],
                               attention_dim=m["attention_dim"], **kw)
    raise ValueError(f"unknown model name: {name}")
