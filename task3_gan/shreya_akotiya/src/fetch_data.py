#!/usr/bin/env python3
"""
Download the Monet/Photo dataset for Task 3 (Kaggle gan-getting-started).

Two unpaired domains:
    monet_jpg/   300 Monet paintings
    photo_jpg/  7038 photographs

Requires:
    1. Kaggle account with ~/.kaggle/kaggle.json
    2. Must JOIN the competition first: https://www.kaggle.com/c/gan-getting-started
       (otherwise download returns 403)

Usage:
    python task3_gan/shreya_akotiya/src/fetch_data.py
    python task3_gan/shreya_akotiya/src/fetch_data.py --inspect-only
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (Path(os.environ["LAB1_DATA"]) / "task3" if os.environ.get("LAB1_DATA")
            else TASK_DIR.parent / "data")
MONET_DIR = DATA_DIR / "monet_jpg"
PHOTO_DIR = DATA_DIR / "photo_jpg"
COMPETITION = "gan-getting-started"


def download():
    """Download and extract the dataset from Kaggle."""
    try:
        import kaggle
    except ImportError:
        sys.exit("pip install kaggle  # and set up ~/.kaggle/kaggle.json")

    if MONET_DIR.exists() and PHOTO_DIR.exists():
        n_monet = len(list(MONET_DIR.glob("*.jpg")))
        n_photo = len(list(PHOTO_DIR.glob("*.jpg")))
        if n_monet >= 300 and n_photo >= 7000:
            print(f"Data already exists: {n_monet} Monet, {n_photo} photos. Skipping download.")
            return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_zip = Path("/tmp/gan-getting-started.zip")

    print(f"Downloading {COMPETITION} from Kaggle...", flush=True)
    print("(Make sure you've joined the competition at https://www.kaggle.com/c/gan-getting-started)")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", COMPETITION,
        "-p", "/tmp"
    ], check=True)

    print(f"Extracting to {DATA_DIR}...", flush=True)
    subprocess.run([
        "unzip", "-q", "-o", str(tmp_zip), "-d", str(DATA_DIR)
    ], check=True)

    tmp_zip.unlink(missing_ok=True)
    print("Done!")


def inspect():
    """Print dataset statistics."""
    if not MONET_DIR.exists() or not PHOTO_DIR.exists():
        sys.exit(f"Data not found at {DATA_DIR}. Run without --inspect-only first.")

    monet_files = sorted(MONET_DIR.glob("*.jpg"))
    photo_files = sorted(PHOTO_DIR.glob("*.jpg"))

    print(f"\n===== Dataset: {COMPETITION} =====")
    print(f"Monet paintings : {len(monet_files):,}  ({MONET_DIR})")
    print(f"Photographs     : {len(photo_files):,}  ({PHOTO_DIR})")
    print(f"Domain ratio    : {len(photo_files)/len(monet_files):.1f}:1 (photo:monet)")

    # Check image sizes
    try:
        from PIL import Image
        monet_sample = Image.open(monet_files[0])
        photo_sample = Image.open(photo_files[0])
        print(f"\nSample Monet size: {monet_sample.size}")
        print(f"Sample Photo size: {photo_sample.size}")
    except ImportError:
        print("\n(pip install pillow to see image sizes)")

    # Disk usage
    monet_size = sum(f.stat().st_size for f in monet_files)
    photo_size = sum(f.stat().st_size for f in photo_files)
    print(f"\nDisk usage:")
    print(f"  Monet: {monet_size/1e6:.1f} MB")
    print(f"  Photo: {photo_size/1e6:.1f} MB")
    print(f"  Total: {(monet_size+photo_size)/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    if not args.inspect_only:
        download()
    inspect()
