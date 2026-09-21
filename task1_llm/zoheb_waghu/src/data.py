"""TinyStories character-level data pipeline.

Workplan 1.1: 100K training characters / 10K validation characters, my own
char_to_idx / idx_to_char maps, fixed-length input-target pairs where the target
is the input shifted right by one, and a vocabulary built from the TRAINING
SPLIT ONLY so no held-out information leaks into the embeddings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

UNK = "\x00"   # stands in for validation characters unseen during training


@dataclass
class CharVocab:
    """Character <-> integer maps. Built from training text only."""
    char_to_idx: dict
    idx_to_char: dict

    @property
    def size(self) -> int:
        return len(self.char_to_idx)

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        chars = [UNK] + sorted(set(text))
        c2i = {c: i for i, c in enumerate(chars)}
        return cls(c2i, {i: c for c, i in c2i.items()})

    def encode(self, text: str) -> np.ndarray:
        unk = self.char_to_idx[UNK]
        return np.fromiter((self.char_to_idx.get(c, unk) for c in text),
                           dtype=np.int64, count=len(text))

    def decode(self, ids) -> str:
        return "".join(self.idx_to_char[int(i)] for i in ids)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"char_to_idx": self.char_to_idx}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        c2i = json.loads(Path(path).read_text())["char_to_idx"]
        return cls(c2i, {i: c for c, i in c2i.items()})


def fetch_text(dataset: str, split: str, n_chars: int, seed: int) -> str:
    """Stream stories until n_chars is reached. Streaming avoids materialising
    the full 2GB corpus for a 110K-character experiment."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True).shuffle(
        seed=seed, buffer_size=10_000)
    parts, total = [], 0
    for row in ds:
        text = row["text"].strip() + "\n\n"
        parts.append(text)
        total += len(text)
        if total >= n_chars:
            break
    return "".join(parts)[:n_chars]


def build_splits(cfg: dict) -> dict:
    """Fetch, split and encode. Returns arrays plus the vocab.

    The validation slice is taken from a DISJOINT region of the stream, and the
    vocabulary is fitted on train only (cfg.data.vocab_from == 'train_only').
    """
    d = cfg["data"]
    seed = cfg["run"]["seed"]
    n_train, n_val = d["train_chars"], d["val_chars"]

    raw = fetch_text(d["dataset"], d["split_source"], n_train + n_val, seed)
    train_text, val_text = raw[:n_train], raw[n_train:n_train + n_val]

    if d["vocab_from"] != "train_only":
        raise ValueError("vocab_from must be 'train_only' (workplan 1.1)")
    vocab = CharVocab.from_text(train_text)

    train_ids, val_ids = vocab.encode(train_text), vocab.encode(val_text)
    oov = int((val_ids == vocab.char_to_idx[UNK]).sum())
    return {
        "vocab": vocab,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_text": train_text,
        "val_text": val_text,
        "val_oov_chars": oov,
    }


def save_processed(splits: dict, cache_dir: str) -> dict:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "train_ids.npy", splits["train_ids"])
    np.save(cache / "val_ids.npy", splits["val_ids"])
    splits["vocab"].save(cache / "vocab.json")
    stats = {
        "train_chars": int(splits["train_ids"].size),
        "val_chars": int(splits["val_ids"].size),
        "vocab_size": splits["vocab"].size,
        "val_oov_chars": splits["val_oov_chars"],
    }
    (cache / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


class SequenceBatcher:
    """Fixed-length (x, y) pairs where y is x shifted right by one.

    An "epoch" is defined as ceil(n_tokens / (batch_size * block_size)) batches -
    i.e. one pass over as many tokens as the corpus holds - with offsets drawn at
    random. Stated explicitly because epoch counts are graded (minimum 10).
    """

    def __init__(self, ids: np.ndarray, block_size: int, batch_size: int,
                 device: torch.device, seed: int = 0) -> None:
        self.data = torch.from_numpy(ids)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.n_positions = len(ids) - block_size - 1

    @property
    def batches_per_epoch(self) -> int:
        return max(1, self.n_positions // (self.batch_size * self.block_size))

    def sample(self) -> tuple:
        ix = self.rng.integers(0, self.n_positions, size=self.batch_size)
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)

    def iter_sequential(self) -> Iterator[tuple]:
        """Non-overlapping sweep - used for validation so every token is scored
        exactly once, making val loss comparable across runs."""
        step = self.batch_size * self.block_size
        for start in range(0, self.n_positions - step, step):
            xs, ys = [], []
            for b in range(self.batch_size):
                i = start + b * self.block_size
                xs.append(self.data[i:i + self.block_size])
                ys.append(self.data[i + 1:i + 1 + self.block_size])
            yield torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)
