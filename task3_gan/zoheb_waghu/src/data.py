"""Unpaired image domains, held in memory.

A = Monet (300 images), B = photo (7,038 images), all 256x256. Each domain is resized to
`load_size` once (bicubic) and kept as a uint8 tensor on the device; each training step takes a
random `image_size` crop (and an optional horizontal flip) and scales to [-1, 1]. No dataloader
workers are needed, which matters on Windows.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_domain(folder, load_size: int, device) -> tuple:
    """-> (uint8 tensor (N, 3, load_size, load_size) on `device`, sorted file names)."""
    files = sorted(Path(folder).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no .jpg images in {folder}")
    arr = np.stack([np.asarray(Image.open(p).convert("RGB").resize((load_size, load_size), Image.BICUBIC))
                    for p in files])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device), [p.name for p in files]


def random_batch(imgs: torch.Tensor, idx, size: int, flip: bool) -> torch.Tensor:
    """Random crop (+ flip) of the chosen images, float in [-1, 1]."""
    span = imgs.shape[-1] - size + 1
    out = []
    for i in idx:
        y, x = random.randrange(span), random.randrange(span)
        t = imgs[i, :, y:y + size, x:x + size]
        out.append(t.flip(-1) if flip and random.random() < 0.5 else t)
    return torch.stack(out).float().div(127.5).sub(1.0)


class ImagePool:
    """Buffer of previously generated images (Shrivastava et al.): the discriminator is shown a
    mix of new and past fakes, which stabilises training. size 0 disables it."""

    def __init__(self, size: int) -> None:
        self.size, self.items = size, []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return batch
        out = []
        for img in batch.detach():
            img = img.unsqueeze(0)
            if len(self.items) < self.size:
                self.items.append(img)
                out.append(img)
            elif random.random() < 0.5:                 # swap with a stored fake
                j = random.randrange(self.size)
                out.append(self.items[j].clone())
                self.items[j] = img
            else:
                out.append(img)
        return torch.cat(out)
