#!/usr/bin/env python3
"""Paired McNemar test between the two members' best Task 2 models.

Both members evaluate on the *same* official 38K Yelp polarity test split, in the
same order, so a paired test is valid even though the models were trained on
different amounts of data. That is the point: comparing accuracies across members
confounds architecture with training-set size, but McNemar conditions on the
examples where the two models actually disagree.

Needs one file per member: the model's predicted P(positive) for every test row,
in test-split order.

  zoheb_waghu     already saved:
      task2_sentiment/zoheb_waghu/outputs/test_probs_t2_m3_bilstm_attn.npy

  shreya_akotiya  to export, add this at the end of task2_sentiment.ipynb after
                  scoring the best model (exp_bilstm):

      import numpy as np
      np.save(OUT / "test_probs_exp_bilstm.npy", test_probs)   # shape (38000,), float

Then run from the repo root:

    python task2_sentiment/cross_member_mcnemar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "task2_sentiment/shreya_akotiya/outputs/test_probs_exp_bilstm.npy"
B = ROOT / "task2_sentiment/zoheb_waghu/outputs/test_probs_t2_m3_bilstm_attn.npy"
LABELS = ROOT / "task2_sentiment/zoheb_waghu/data_processed/test.npz"


def main() -> int:
    missing = [p for p in (A, B, LABELS) if not p.exists()]
    if missing:
        print("missing input(s):")
        for p in missing:
            print("   ", p.relative_to(ROOT))
        print("\nSee the module docstring for how to export the predictions.")
        return 1

    y = np.load(LABELS)["y"]
    pa, pb = np.load(A), np.load(B)
    if not (len(y) == len(pa) == len(pb)):
        print(f"length mismatch: labels {len(y)}, shreya {len(pa)}, zoheb {len(pb)} - "
              "both members must score the full official test split, in its original order")
        return 1

    ca = (pa >= 0.5).astype(int) == y
    cb = (pb >= 0.5).astype(int) == y
    b01 = int((ca & ~cb).sum())      # shreya right, zoheb wrong
    b10 = int((~ca & cb).sum())      # zoheb right, shreya wrong

    from statsmodels.stats.contingency_tables import mcnemar
    table = np.array([[int((ca & cb).sum()), b01], [b10, int((~ca & ~cb).sum())]])
    exact = (b01 + b10) < 25
    res = mcnemar(table, exact=exact, correction=not exact)

    print(f"n = {len(y):,} test reviews, scored by both members\n")
    print(f"  shreya accuracy {ca.mean():.5f}")
    print(f"  zoheb  accuracy {cb.mean():.5f}")
    print(f"  both correct   {table[0,0]:,}   both wrong {table[1,1]:,}")
    print(f"  shreya only    {b01:,}   zoheb only {b10:,}   <- only these carry information")
    print(f"\n  McNemar ({'exact' if exact else 'chi2, continuity-corrected'}): "
          f"statistic {res.statistic:.4f}, p = {res.pvalue:.4e}")
    verdict = ("the two models differ significantly on this test set"
               if res.pvalue < 0.05 else
               "the two models are NOT distinguishable on this test set (p >= 0.05)")
    print(f"  -> {verdict}")
    print("\nNote: this compares two complete setups, not two architectures - the training-set "
          "sizes differ (540K vs 90K). It answers 'is one better here', not 'why'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
