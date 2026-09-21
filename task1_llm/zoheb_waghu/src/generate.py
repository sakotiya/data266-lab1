"""Text generation: greedy and temperature sampling (workplan 1.3)."""
from __future__ import annotations

import time

import torch


def run_generation(model, vocab, cfg: dict, device) -> tuple:
    """Generate one sample per (prompt, strategy). Returns (samples, tokens/sec)."""
    g = cfg["generate"]
    samples, total_tokens, elapsed = [], 0, 0.0
    for prompt in g["prompts"]:
        ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long, device=device)
        for strat in g["strategies"]:
            greedy = strat["name"] == "greedy"
            temp = float(strat.get("temperature", 1.0))
            label = "greedy" if greedy else f"temperature={temp}"
            t0 = time.time()
            out = model.generate(ids, g["max_new_tokens"], temperature=temp, greedy=greedy)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed += time.time() - t0
            total_tokens += g["max_new_tokens"]
            samples.append({"prompt": prompt, "label": label,
                            "text": vocab.decode(out[0].tolist())})
    return samples, total_tokens / max(elapsed, 1e-9)
