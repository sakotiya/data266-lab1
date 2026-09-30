"""Task 1 metrics: every item on the workplan's list for this task."""
from __future__ import annotations

import math
from collections import Counter

import torch


@torch.no_grad()
def eval_loss_and_accuracy(model, batcher) -> dict:
    """Cross-entropy, perplexity, bits-per-character and top-1 next-char accuracy
    over a sequential (non-overlapping) sweep, so each token is scored once."""
    model.eval()
    tot_loss, tot_correct, tot_tokens = 0.0, 0, 0
    for x, y in batcher.iter_sequential():
        logits, loss = model(x, y)
        n = y.numel()
        tot_loss += loss.item() * n
        tot_correct += (logits.argmax(-1) == y).sum().item()
        tot_tokens += n
    mean_loss = tot_loss / max(tot_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "bits_per_char": mean_loss / math.log(2),   # natural-log CE -> bits
        "top1_next_char_acc": tot_correct / max(tot_tokens, 1),
        "tokens_scored": tot_tokens,
    }


def distinct_n(text: str, n: int) -> float:
    """Distinct-n: unique n-grams / total n-grams. Generation diversity."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Fraction of n-gram occurrences that are repeats of an n-gram already seen.
    High values indicate the degenerate looping failure mode."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / len(grams)


def generation_metrics(samples: list[str], ngram_max: int = 3,
                       repeat_n: int = 4) -> dict:
    joined = "\n".join(samples)
    out = {f"distinct_{n}": distinct_n(joined, n) for n in range(1, ngram_max + 1)}
    out[f"repeated_{repeat_n}gram_rate"] = repeated_ngram_rate(joined, repeat_n)
    return out


@torch.no_grad()
def causal_mask_probe(model, vocab_size: int, seq_len: int, device) -> dict:
    """BEHAVIOURAL proof of causality (workplan 1.2, explicitly not a visual check).

    Perturb the token at position j and measure how much the logits move at every
    position. A correct causal model: positions i < j are bit-identical (delta 0),
    positions i >= j change. A non-zero max delta below j means the mask leaks.
    """
    model.eval()
    j = seq_len // 2
    max_before, min_after = 0.0, float("inf")
    for t in range(5):
        g = torch.Generator(device="cpu").manual_seed(1000 + t)
        idx = torch.randint(0, vocab_size, (1, seq_len), generator=g).to(device)
        base, _ = model(idx)

        alt = idx.clone()
        new_tok = (idx[0, j].item() + 1 + t) % vocab_size
        alt[0, j] = new_tok
        pert, _ = model(alt)

        delta = (pert - base).abs()                      # (1, T, V)
        max_before = max(max_before, delta[:, :j, :].max().item())
        min_after = min(min_after, delta[:, j:, :].max().item())
    return {
        "probe_position": j,
        "max_logit_delta_before_j": max_before,    # must be ~0
        "min_logit_delta_from_j": min_after,       # must be > 0
        "passed": max_before == 0.0 and min_after > 0.0,
    }


def gradient_stability(grad_norms: list[float], losses: list[float],
                       spike_factor: float = 2.0) -> dict:
    """Loss spikes = a logged loss more than `spike_factor` x the running median."""
    if not grad_norms:
        return {"grad_norm_mean": 0.0, "grad_norm_max": 0.0, "loss_spikes": 0}
    spikes, window = 0, 20
    for i in range(window, len(losses)):
        med = sorted(losses[i - window:i])[window // 2]
        if losses[i] > spike_factor * med:
            spikes += 1
    return {
        "grad_norm_mean": sum(grad_norms) / len(grad_norms),
        "grad_norm_max": max(grad_norms),
        "loss_spikes": spikes,
    }
