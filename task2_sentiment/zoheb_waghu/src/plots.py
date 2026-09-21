"""Task 2 figures: EDA, confusion matrices, calibration, model comparison.

    python task2_sentiment/zoheb_waghu/src/plots.py --config <any model config>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[2]))

from common.config import config_arg_parser, load_config  # noqa: E402
from metrics import core_metrics  # noqa: E402

MODELS = [("t2_m1_baseline", "M1 BiLSTM-mean (baseline)"),
          ("t2_m2_cnn", "M2 TextCNN"),
          ("t2_m3_bilstm_attn", "M3 BiLSTM-attention")]


def eda_figure(cache: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    eda = json.loads((cache / "eda.json").read_text())
    feats = json.loads((cache / "feats.json").read_text())
    lens = [f["len_tokens"] for f in feats["test"]]
    z = np.load(cache / "test.npz")
    y = z["y"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(lens, bins=60, color="tab:blue", alpha=.8)
    ax[0].axvline(np.median(lens), color="k", ls="--", label=f"median {np.median(lens):.0f}")
    ax[0].axvline(np.percentile(lens, 95), color="r", ls="--",
                  label=f"p95 {np.percentile(lens, 95):.0f}")
    ax[0].set(xlabel="tokens after preprocessing", ylabel="reviews",
              title="Review length distribution")
    ax[0].legend()

    counts = [int((y == 0).sum()), int((y == 1).sum())]
    ax[1].bar(["negative", "positive"], counts, color=["tab:red", "tab:green"], alpha=.8)
    for i, c in enumerate(counts):
        ax[1].text(i, c, f"{c}\n({c / len(y):.1%})", ha="center", va="bottom")
    ax[1].set(ylabel="reviews", title="Class distribution (test)")
    ax[1].set_ylim(0, max(counts) * 1.2)

    ax[2].axis("off")
    rows = [("train rows", f"{eda['train_rows']:,}"), ("val rows", f"{eda['val_rows']:,}"),
            ("test rows", f"{eda['test_rows']:,}"), ("vocab size", f"{eda['vocab_size']:,}"),
            ("mean length", f"{eda['token_len_mean']:.1f}"),
            ("median length", f"{eda['token_len_median']:.0f}"),
            ("p95 length", f"{eda['token_len_p95']:.0f}"),
            ("truncated @256", f"{eda['truncated_frac']:.2%}"),
            ("train pos frac", f"{eda['train_pos_frac']:.3f}"),
            ("stopwords removed", f"{eda['stopwords_removed']}")]
    ax[2].table(cellText=rows, loc="center", cellLoc="left", colWidths=[.55, .45])
    ax[2].set_title("Dataset summary")

    fig.tight_layout()
    fig.savefig(out / "eda_overview.png", dpi=140)
    print("wrote", out / "eda_overview.png")


def confusion_and_calibration(out: Path, y_true: np.ndarray,
                              probs_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    avail = [(tag, name) for tag, name in MODELS
             if (probs_dir / f"test_probs_{tag}.npy").exists()]
    if not avail:
        return
    fig, ax = plt.subplots(2, len(avail), figsize=(4.6 * len(avail), 8))
    ax = np.atleast_2d(ax)
    for i, (tag, name) in enumerate(avail):
        p = np.load(probs_dir / f"test_probs_{tag}.npy")
        m = core_metrics(y_true, p)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a = ax[0, i]
        a.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                a.text(c, r, f"{cm[r, c]:,}\n{cm[r, c] / cm.sum():.1%}", ha="center",
                       va="center", color="white" if cm[r, c] > cm.max() / 2 else "black")
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred neg", "pred pos"],
              yticklabels=["true neg", "true pos"],
              title=f"{name}\nacc {m['accuracy']:.4f} · F1 {m['f1_macro']:.4f}")

        b = ax[1, i]
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for k in range(10):
            msk = idx == k
            if msk.sum() > 5:
                xs.append(p[msk].mean())
                ys.append(y_true[msk].mean())
        b.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect")
        b.plot(xs, ys, "o-", color="tab:orange", label="model")
        b.set(xlabel="predicted probability", ylabel="observed frequency",
              title=f"Calibration · Brier {m['brier_score']:.4f}")
        b.legend(); b.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "confusion_and_calibration.png", dpi=140)
    print("wrote", out / "confusion_and_calibration.png")


def comparison_figure(csv_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df[df.model_name.notna() & ~df.run_id.str.contains("smoke")]
    if df.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(df))
    ax[0].bar(x, df.accuracy, color="tab:blue", alpha=.85,
              yerr=[df.accuracy - df.acc_ci_low, df.acc_ci_high - df.accuracy],
              capsize=6)
    ax[0].set(xticks=x, ylabel="accuracy", title="Test accuracy with 95% bootstrap CI")
    ax[0].set_xticklabels(df.model_name, rotation=12)
    ax[0].set_ylim(min(df.acc_ci_low) - .005, max(df.acc_ci_high) + .005)
    for i, (a, lo, hi) in enumerate(zip(df.accuracy, df.acc_ci_low, df.acc_ci_high)):
        ax[0].text(i, hi + .0008, f"{a:.4f}\n[{lo:.4f}, {hi:.4f}]", ha="center", fontsize=8)

    slices = ["short", "long", "negation", "highpunct"]
    w = .8 / len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        ax[1].bar(np.arange(len(slices)) + i * w,
                  [r[f"slice_{s}_f1_macro"] for s in slices], w, label=r.model_name, alpha=.85)
    ax[1].set(xticks=np.arange(len(slices)) + .4 - w / 2, ylabel="macro-F1",
              title="Macro-F1 per robustness slice")
    ax[1].set_xticklabels(slices)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=140)
    print("wrote", out / "model_comparison.png")


def main() -> int:
    args = config_arg_parser("Task 2 figures").parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_dir"])           # .npy predictions live here
    plots = Path(cfg["paths"]["plot_dir"])            # team layout: outputs/plots/
    confusion = Path(cfg["paths"]["confusion_dir"])   # team layout: outputs/confusion_matrices/
    cache = Path(cfg["data"]["cache_dir"])
    y = np.load(cache / "test.npz")["y"]
    eda_figure(cache, plots)
    confusion_and_calibration(confusion, y, out)
    # the comparison figure reads the EXTENDED table - the team header has no slice columns
    comparison_figure(Path(cfg["paths"]["extended_metrics_csv_path"]), plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
