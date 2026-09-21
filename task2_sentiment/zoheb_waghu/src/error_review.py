"""Pull the 20 errors the workplan's manual review requires, with raw review text.

    python task2_sentiment/zoheb_waghu/src/error_review.py \
        --config task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml --model t2_m1_baseline

Four groups of five: confident false positives, confident false negatives,
near-threshold errors, and errors inside the model's worst robustness slice.
Writes outputs/error_review_<model>.md for manual annotation - the error TYPE and
the testable FIX are judgement calls and are left blank on purpose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[2]))

from common.config import config_arg_parser, load_config  # noqa: E402
from data import is_malformed, slice_features  # noqa: E402
from metrics import core_metrics, error_buckets, slice_metrics  # noqa: E402
from data import clean_text, get_stopwords  # noqa: E402


def raw_test_texts(cfg: dict) -> list:
    """Re-derive the test texts in the SAME order build_dataset kept them, so
    indices from the probability array line up with review text."""
    from datasets import load_dataset
    from nltk.stem import WordNetLemmatizer

    d, pp = cfg["data"], cfg["preprocess"]
    ds = load_dataset(d["dataset"], split="test")
    if d["test_size"]:
        ds = ds.select(range(min(d["test_size"], len(ds))))
    stop = get_stopwords(pp["stopwords"])
    lem = WordNetLemmatizer() if pp["lemmatize"] else None
    texts = []
    for row in ds:
        if is_malformed(row["text"], row["label"]):
            continue
        if not clean_text(row["text"], pp, stop, lem):
            continue
        texts.append(row["text"])
    return texts


def excerpt(t: str, n: int = 320) -> str:
    t = " ".join(t.replace("\\n", " ").split())
    return t[:n] + ("…" if len(t) > n else "")


def main() -> int:
    ap = config_arg_parser("Task 2 error review")
    ap.add_argument("--model", required=True, help="model tag, e.g. t2_m1_baseline")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(cfg["paths"]["output_dir"])
    cache = Path(cfg["data"]["cache_dir"])

    probs = np.load(out / f"test_probs_{a.model}.npy")
    y = np.load(cache / "test.npz")["y"]
    feats = json.loads((cache / "feats.json").read_text())["test"]
    texts = raw_test_texts(cfg)
    if not (len(texts) == len(y) == len(probs)):
        raise RuntimeError(f"alignment mismatch: texts={len(texts)} y={len(y)} probs={len(probs)}")

    from data import compute_slices
    masks = compute_slices(feats, cfg["eval"]["slices"])
    sl = slice_metrics(y, probs, masks)
    worst = min((k for k in sl if not np.isnan(sl[k]["f1_macro"])),
                key=lambda k: sl[k]["f1_macro"])
    b = error_buckets(y, probs, masks=masks, worst_slice=worst)
    m = core_metrics(y, probs)

    groups = [
        ("A. Confident false positives (predicted positive, actually negative)",
         b["confident_false_positives"]),
        ("B. Confident false negatives (predicted negative, actually positive)",
         b["confident_false_negatives"]),
        ("C. Near-threshold errors (|p - 0.5| <= 0.05)", b["near_threshold_errors"]),
        (f"D. Worst-slice errors - slice `{worst}` "
         f"(macro-F1 {sl[worst]['f1_macro']:.4f} vs overall {m['f1_macro']:.4f})",
         b.get("slice_errors", [])),
    ]

    lines = [f"# Task 2 - Error review: `{a.model}`", "",
             f"Test accuracy {m['accuracy']:.4f} · macro-F1 {m['f1_macro']:.4f} · "
             f"MCC {m['mcc']:.4f} · Brier {m['brier_score']:.4f}", "",
             "Error type vocabulary: `negation` · `sarcasm/irony` · `mixed sentiment` · "
             "`aspect confusion` · `rating-text mismatch` · `domain term` · "
             "`length truncation` · `rare vocabulary / OOV` · `label noise` · `other`", ""]
    n = 1
    for title, idxs in groups:
        lines += [f"## {title}", ""]
        for i in idxs:
            f = feats[i]
            lines += [
                f"### {n}. test index {i} — p(pos) = {probs[i]:.4f}, true = "
                f"{'positive' if y[i] == 1 else 'negative'}",
                f"- tokens after preprocessing: {f['len_tokens']} · "
                f"exclamations: {f['exclamation_count']} · negation present: "
                f"{'yes' if f['has_negation'] else 'no'}",
                "",
                f"> {excerpt(texts[i])}",
                "",
                "- **Error type:** _<assign one>_",
                "- **Testable fix:** _<one change you could make>_",
                "- **Metric that should move:** _<which number, in which direction>_",
                "",
            ]
            n += 1
    (out / f"error_review_{a.model}.md").write_text("\n".join(lines))
    print(f"wrote {out / f'error_review_{a.model}.md'} ({n - 1} errors, worst slice: {worst})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
