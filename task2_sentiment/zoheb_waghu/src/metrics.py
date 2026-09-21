"""Every metric on the Task 2 list, plus the two statistical tests the workplan
insists on: 95% bootstrap CIs and the paired McNemar test.

The CI/McNemar pairing matters: overlapping CIs mean two models are NOT
distinguishable on this test set regardless of point estimates, and McNemar is
the more sensitive comparison because it is paired on the same examples.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """ECE: |accuracy - confidence| averaged over equal-width confidence bins."""
    conf = np.where(probs >= 0.5, probs, 1 - probs)
    pred = (probs >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def core_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(probs) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, pred, average="micro", zero_division=0),
        "precision_weighted": precision_score(y_true, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier_score": brier_score_loss(y_true, probs),
    }


def bootstrap_ci(y_true, probs, stat: str, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI over test examples, resampled with replacement."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    fn = {"accuracy": lambda t, p: accuracy_score(t, (p >= 0.5).astype(int)),
          "f1_macro": lambda t, p: f1_score(t, (p >= 0.5).astype(int),
                                            average="macro", zero_division=0),
          "mcc": lambda t, p: matthews_corrcoef(t, (p >= 0.5).astype(int))}[stat]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:      # degenerate resample
            vals[b] = np.nan
            continue
        vals[b] = fn(y_true[idx], probs[idx])
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def mcnemar_test(y_true, probs_a, probs_b) -> dict:
    """Paired McNemar, exact binomial. a = baseline, b = experimental.

    b01 = baseline right / experimental wrong; b10 = the reverse. Only the
    disagreements carry information, which is what makes it more sensitive than
    comparing two independent accuracy CIs.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    y = np.asarray(y_true)
    ca = ((np.asarray(probs_a) >= 0.5).astype(int) == y)
    cb = ((np.asarray(probs_b) >= 0.5).astype(int) == y)
    tbl = np.array([[int((ca & cb).sum()), int((ca & ~cb).sum())],
                    [int((~ca & cb).sum()), int((~ca & ~cb).sum())]])
    exact = tbl[0, 1] + tbl[1, 0] < 25
    res = mcnemar(tbl, exact=exact, correction=not exact)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "baseline_only_correct": tbl[0, 1], "experimental_only_correct": tbl[1, 0],
            "exact": exact}


def slice_metrics(y_true, probs, masks: dict) -> dict:
    """Macro-F1 and error rate per robustness slice."""
    out = {}
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    for name, m in masks.items():
        m = np.asarray(m)
        if m.sum() < 2 or len(np.unique(y_true[m])) < 2:
            out[name] = {"n": int(m.sum()), "f1_macro": float("nan"),
                         "error_rate": float("nan")}
            continue
        pred = (probs[m] >= 0.5).astype(int)
        out[name] = {
            "n": int(m.sum()),
            "f1_macro": float(f1_score(y_true[m], pred, average="macro", zero_division=0)),
            "error_rate": float((pred != y_true[m]).mean()),
        }
    return out


def error_buckets(y_true, probs, texts=None, masks=None,
                  worst_slice: str | None = None) -> dict:
    """The four groups of five the workplan's error review requires:
    confident FPs, confident FNs, near-threshold errors, worst-slice errors."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    pred = (probs >= 0.5).astype(int)
    wrong = pred != y_true
    fp = np.where(wrong & (pred == 1))[0]
    fn = np.where(wrong & (pred == 0))[0]
    near = np.where(wrong & (np.abs(probs - 0.5) <= 0.05))[0]
    out = {
        "confident_false_positives": fp[np.argsort(-probs[fp])][:5].tolist(),
        "confident_false_negatives": fn[np.argsort(probs[fn])][:5].tolist(),
        "near_threshold_errors": near[np.argsort(np.abs(probs[near] - 0.5))][:5].tolist(),
    }
    if masks and worst_slice and worst_slice in masks:
        sl = np.asarray(masks[worst_slice])
        out["slice_errors"] = np.where(wrong & sl)[0][:5].tolist()
        out["slice_name"] = worst_slice
    return out
