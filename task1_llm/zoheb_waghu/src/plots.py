"""Loss curves and training-stability plots for Task 1 (workplan 1.3).

    python task1_llm/zoheb_waghu/src/plots.py --history outputs/history_<run_id>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    hp = Path(a.history)
    h = json.loads(hp.read_text())
    run_id = hp.stem.replace("history_", "")
    # figures belong in outputs/plots/ in the team layout
    out = Path(a.out_dir) if a.out_dir else hp.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)

    ep = h["history"]
    epochs = [r["epoch"] + 1 for r in ep]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(epochs, [r["train_loss"] for r in ep], "o-", label="train")
    ax[0].plot(epochs, [r["loss"] for r in ep], "s-", label="validation")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy (nats)",
              title=f"Loss curves - {run_id}")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(epochs, [r["perplexity"] for r in ep], "o-", color="tab:purple")
    ax[1].set(xlabel="epoch", ylabel="perplexity", title="Validation perplexity")
    ax[1].grid(alpha=.3)
    ax2 = ax[1].twinx()
    ax2.plot(epochs, [r["bits_per_char"] for r in ep], "s--", color="tab:orange", alpha=.7)
    ax2.set_ylabel("bits per character", color="tab:orange")

    ax[2].plot(h["step_losses"], lw=.6, color="tab:blue", label="step loss")
    ax[2].plot(h["grad_norms"], lw=.6, color="tab:red", alpha=.6, label="grad norm")
    ax[2].set(xlabel="optimiser step", title="Training stability")
    ax[2].legend(); ax[2].grid(alpha=.3)

    fig.tight_layout()
    p = out / f"loss_curves_{run_id}.png"
    fig.savefig(p, dpi=140)
    print(f"wrote {p}")

    gap = [r["loss"] - r["train_loss"] for r in ep]
    fig2, ax3 = plt.subplots(figsize=(6, 4))
    ax3.axhline(0, color="k", lw=.8)
    ax3.plot(epochs, gap, "o-", color="tab:green")
    ax3.set(xlabel="epoch", ylabel="val loss - train loss",
            title=f"Generalization gap - {run_id}")
    ax3.grid(alpha=.3)
    fig2.tight_layout()
    p2 = out / f"generalization_gap_{run_id}.png"
    fig2.savefig(p2, dpi=140)
    print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
