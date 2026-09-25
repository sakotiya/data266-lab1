#!/usr/bin/env python3
"""
Fetch the shared raw TinyStories corpus for Task 1 (team-level, downloaded once).

Writes ONE canonical file that every team member starts from:

    task1_llm/data/tinystories_raw.txt      (stories separated by a blank line)

This file is the *pool*. How much of it any individual run consumes is decided
separately, by `data.use_chars` / `data.member_offset_chars` in each member's
config.yaml -- so different members can train on disjoint slices of the same
shared download (spec 1.1.4: "each member creates their own split").

Usage
-----
    python task1_llm/shreya_akotiya/src/fetch_data.py                    # full train split (~1.9 GB)
    python task1_llm/shreya_akotiya/src/fetch_data.py --chars 34000000   # cap the download
    python task1_llm/shreya_akotiya/src/fetch_data.py --inspect-only     # describe what is on disk
    python task1_llm/shreya_akotiya/src/fetch_data.py --inspect-only --sample-mb 200

`--inspect-only` prints the facts the preprocessing code depends on: the
character inventory, the vocabulary size that results from each `min_char_freq`
floor, the coverage lost to <unk>, and the story-length distribution.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]   # task1_llm/
# LAB1_DATA lets Colab keep the corpus on fast local disk instead of Google Drive.
DATA_DIR = Path(os.environ["LAB1_DATA"]) if os.environ.get("LAB1_DATA") else TASK_DIR / "data"
OUT = DATA_DIR / "tinystories_raw.txt"
SEP = "\n\n"
HF_REPO = "roneneldan/TinyStories"
HF_RAW_FILE = "TinyStories-train.txt"      # fast path: the pre-joined plain-text file
HF_EOS = "<|endoftext|>"                   # separator used inside that file


# --------------------------------------------------------------------------- download
def _download_via_hub(target_chars: int) -> bool:
    """Fast path: grab the raw .txt from the Hub and re-separate it with blank lines.

    The upstream file glues stories together with the literal string
    '<|endoftext|>'. At character level that would be spelled out one character
    at a time, so we replace it with a blank line -- the boundary signal we
    actually want the model to learn.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    print(f"Trying fast path: {HF_REPO}/{HF_RAW_FILE} ...", flush=True)
    try:
        src = hf_hub_download(repo_id=HF_REPO, filename=HF_RAW_FILE, repo_type="dataset")
    except Exception as e:                                   # noqa: BLE001
        print(f"  fast path unavailable ({type(e).__name__}: {e}); falling back to streaming")
        return False

    print(f"  downloaded to HF cache: {src}")
    print("  re-separating stories -> blank lines ...", flush=True)

    written = n = 0
    t0 = time.time()
    buf: list[str] = []
    with open(src, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            if HF_EOS in line:
                head, _, tail = line.partition(HF_EOS)
                buf.append(head)
                story = "".join(buf).strip()
                buf = [tail]
                if not story:
                    continue
                fout.write(story + SEP)
                written += len(story) + len(SEP)
                n += 1
                if n % 200_000 == 0:
                    print(f"  {n:>9,} stories | {written/1e6:7.1f} MB | {time.time()-t0:5.0f}s",
                          flush=True)
                if target_chars and written >= target_chars:
                    break
            else:
                buf.append(line)
        else:
            story = "".join(buf).strip()
            if story:
                fout.write(story + SEP)
                written += len(story) + len(SEP)
                n += 1

    print(f"\nDone: {n:,} stories / {written:,} chars -> {OUT}  ({time.time()-t0:.0f}s)")
    return True


def _download_via_streaming(target_chars: int, dataset: str, split: str) -> None:
    """Fallback: stream the parquet dataset row by row."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "Missing dependency. Run:\n\n"
            "    pip install datasets huggingface_hub\n\n"
            "then re-run this script."
        )

    cap = f"{target_chars:,} characters" if target_chars else "the whole split"
    print(f"Streaming {dataset} [{split}] until {cap} ...", flush=True)
    ds = load_dataset(dataset, split=split, streaming=True)

    written = n = 0
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as f:
        for row in ds:
            story = row["text"].strip()
            if not story:
                continue
            f.write(story + SEP)
            written += len(story) + len(SEP)
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:>9,} stories | {written/1e6:7.1f} MB | {time.time()-t0:5.0f}s",
                      flush=True)
            if target_chars and written >= target_chars:
                break
    print(f"\nDone: {n:,} stories / {written:,} chars -> {OUT}  ({time.time()-t0:.0f}s)")


def download(target_chars: int, dataset: str, split: str, force_stream: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # The hub fast path grabs the whole 1.9 GB file even if we only want a slice of it.
    # With a small --chars cap (Colab), streaming stops early and is much cheaper.
    if not force_stream and dataset == HF_REPO and split == "train" and _download_via_hub(target_chars):
        return
    _download_via_streaming(target_chars, dataset, split)


# --------------------------------------------------------------------------- inspect
def inspect(sample_mb: int) -> None:
    """Report the facts the preprocessing code makes assumptions about.

    Reads in chunks so a 2 GB corpus never lands in memory all at once.
    Character statistics use the first `sample_mb` megabytes (the corpus is
    homogeneous, so a 200 MB sample is a faithful estimate of the inventory);
    story counting streams the whole file.
    """
    if not OUT.exists():
        sys.exit(f"{OUT} does not exist yet - run without --inspect-only first.")

    size = OUT.stat().st_size
    limit = sample_mb * 1_000_000
    counts: Counter[str] = Counter()
    lens: list[int] = []
    total_stories = 0
    total_chars = 0
    scanned = 0
    tail = ""

    t0 = time.time()
    with open(OUT, encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 24)                      # 16 MB
            if not chunk:
                break
            total_chars += len(chunk)
            if scanned < limit:
                counts.update(chunk)
                scanned += len(chunk)
            buf = tail + chunk
            parts = buf.split(SEP)
            tail = parts.pop()
            for p in parts:
                if p.strip():
                    total_stories += 1
                    if len(lens) < 400_000:              # enough for a stable distribution
                        lens.append(len(p))
    if tail.strip():
        total_stories += 1
        lens.append(len(tail))

    lens.sort()
    pct = lambda p: lens[min(int(len(lens) * p), len(lens) - 1)]   # noqa: E731

    print(f"\nfile             : {OUT}")
    print(f"size             : {size/1e9:.2f} GB / {total_chars:,} characters")
    print(f"stories          : {total_stories:,}")
    print(f"story length     : min {lens[0]}  p25 {pct(.25)}  median {pct(.50)}  "
          f"p75 {pct(.75)}  p99 {pct(.99)}  max {lens[-1]}   (n={len(lens):,})")
    print(f"mean story length: {sum(lens)/len(lens):.0f} chars")
    print(f"scan time        : {time.time()-t0:.0f}s")

    print(f"\n--- character inventory (from first {scanned/1e6:.0f} MB) ---")
    print(f"distinct characters: {len(counts)}")
    print(f"{'min_char_freq':>14} {'vocab (incl <unk>)':>20} {'coverage':>12} {'chars->unk':>12}")
    for floor in (0, 10, 50, 100, 200, 500, 1000, 5000):
        scaled = floor * scanned / 1e6 / 34          # floor is quoted per ~34M-char run
        kept = [c for c, k in counts.items() if k >= floor]
        lost = sum(k for c, k in counts.items() if k < floor)
        print(f"{floor:>14} {len(kept)+1:>20} {(1-lost/scanned)*100:>11.6f}% "
              f"{len(counts)-len(kept):>12}")

    print("\ntop 45 characters:")
    for c, k in counts.most_common(45):
        print(f"  {repr(c):<10} {k:>13,}  {k/scanned*100:7.4f}%")

    rare = sorted((k, c) for c, k in counts.items() if k < 200)
    print(f"\n{len(rare)} characters below frequency 200 in the sample (candidates for <unk>):")
    for k, c in rare[:80]:
        print(f"  {repr(c):<12} {k}")
    if len(rare) > 80:
        print(f"  ... and {len(rare)-80} more")

    non_ascii = sorted(((k, c) for c, k in counts.items() if ord(c) > 127), reverse=True)
    print(f"\n{len(non_ascii)} non-ASCII characters (what the normalisation map must handle):")
    for k, c in non_ascii[:40]:
        print(f"  {repr(c):<12} U+{ord(c):04X}  {k:>10,}")

    print("\n--- first 800 characters --------------------------------------------")
    with open(OUT, encoding="utf-8") as f:
        print(f.read(800))
    print("--- end --------------------------------------------------------------")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chars", type=int, default=0,
                    help="stop after roughly this many characters (0 = whole split)")
    ap.add_argument("--dataset", default=HF_REPO)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample-mb", type=int, default=200,
                    help="how much of the file to use for character statistics")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="force the streaming path; cheaper when --chars is small")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    a = ap.parse_args()

    if not a.inspect_only:
        enough = OUT.exists() and (not a.chars or OUT.stat().st_size >= 0.95 * a.chars)
        if enough and not a.force:
            print(f"{OUT} already exists ({OUT.stat().st_size/1e9:.2f} GB) - skipping download "
                  f"(use --force to re-download)")
        else:
            download(a.chars, a.dataset, a.split, force_stream=a.stream)
    inspect(a.sample_mb)
