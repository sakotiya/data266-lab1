#!/usr/bin/env python3
"""
Download the shared Yelp Polarity dataset for Task 2.

Saves two CSVs in task2_sentiment/data/ (shared by the whole team):
    train.csv  560,000 rows
    test.csv    38,000 rows
Columns: label (0=negative, 1=positive), text

Each member subsamples from these in their own config, so we only download once.

Usage:
    python task2_sentiment/shreya_akotiya/src/fetch_data.py
    python task2_sentiment/shreya_akotiya/src/fetch_data.py --inspect-only
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# LAB1_DATA lets Colab keep the CSVs on fast local disk instead of Google Drive.
DATA_DIR = (Path(os.environ["LAB1_DATA"]) if os.environ.get("LAB1_DATA")
            else Path(__file__).resolve().parents[2] / "data")
TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"
REPO = "fancyzhx/yelp_polarity"


def download():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, path in (("train", TRAIN), ("test", TEST)):
        if path.exists():
            print(f"{path.name} already there, skipping")
            continue
        print(f"downloading {split} ...", flush=True)
        ds = load_dataset(REPO, split=split)
        df = pd.DataFrame({"label": ds["label"], "text": ds["text"]})
        df.to_csv(path, index=False)
        print(f"  wrote {len(df):,} rows -> {path}")


def inspect():
    """Print the things the preprocessing code needs to know."""
    if not TRAIN.exists():
        sys.exit("no data yet, run without --inspect-only first")

    for path in (TRAIN, TEST):
        df = pd.read_csv(path)
        print(f"\n===== {path.name} =====")
        print(f"rows            : {len(df):,}")
        print(f"size on disk    : {path.stat().st_size / 1e6:.0f} MB")

        # class balance (2.1.1)
        counts = df.label.value_counts().sort_index()
        print("class counts    :", dict(counts))
        print(f"class balance   : {counts.min() / counts.max():.4f}  (1.0 = perfectly balanced)")

        # missing / malformed (2.1.2)
        n_null = df.text.isna().sum()
        n_empty = (df.text.fillna("").str.strip() == "").sum()
        bad_label = (~df.label.isin([0, 1])).sum()
        print(f"null text       : {n_null}")
        print(f"empty text      : {n_empty}")
        print(f"labels not 0/1  : {bad_label}")
        print(f"exact duplicates: {df.duplicated(subset=['text']).sum():,}")

        # length distribution (2.1.1)
        chars = df.text.fillna("").str.len()
        words = df.text.fillna("").str.split().str.len()
        print("\nchars per review:")
        print(chars.describe(percentiles=[.25, .5, .75, .9, .95, .99]).round(0).to_string())
        print("\nwords per review:")
        print(words.describe(percentiles=[.25, .5, .75, .9, .95, .99]).round(0).to_string())

        # how many words does a max_len of N cover?
        for n in (100, 150, 200, 300, 400):
            print(f"  max_len {n:>3}: keeps all of {(words <= n).mean() * 100:5.1f}% of reviews")

    df = pd.read_csv(TRAIN, nrows=3)
    print("\n===== sample rows =====")
    for _, r in df.iterrows():
        print(f"\nlabel={r.label}\n{r.text[:400]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-only", action="store_true")
    a = ap.parse_args()
    if not a.inspect_only:
        download()
    inspect()
