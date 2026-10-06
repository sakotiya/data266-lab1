"""Task 3 local evaluation (shreya_akotiya).

The evaluation itself lives in two notebooks in src/; this script runs them in order and keeps
the metrics files in sync, so one command reproduces every reported Task 3 number.

    src/part3_evaluation_shreya.ipynb   instructor's method (first 300 sorted images per set,
                                        Inception-v3) -> FID / MiFID per direction -> submission.csv
    src/run1_team_metrics.ipynb         team metrics on the same 300 images (KID, precision/recall,
                                        density/coverage, cycle L1, LPIPS, content cosine, losses)
                                        -> metrics_report.csv, outputs/failure_candidates.csv

Direction names in the metric files follow the instructor/team: A = Monet, B = photo
(A2B = Monet -> photo, B2A = photo -> Monet). My notebook's outputs/pred_A2B/ is photo -> Monet.

Usage (from the repo root):
    python task3_gan/shreya_akotiya/evaluate_local.py metrics   # run both notebooks (GPU; Colab)
    python task3_gan/shreya_akotiya/evaluate_local.py sync      # metrics_report.csv -> full_metrics_report.csv
    python task3_gan/shreya_akotiya/evaluate_local.py check     # files present, complete and consistent

`metrics` needs task3_gan/data/{monet_jpg,photo_jpg}/, the predictions in outputs/pred_A2B and
outputs/pred_B2A, and the epoch-80 checkpoint t3_shreya_unet_128_20261002_023102_epoch080.pt
(1.0 GB, kept on Drive - see reproducibility/manifests/shreya_akotiya/task3_gan_manifest.md).
The notebooks are executed in place, so their saved outputs are the evidence for the numbers.
On Colab they find the repo via LAB1_ROOT, else the Drive copy.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

MEMBER_DIR = Path(__file__).resolve().parent
SRC = MEMBER_DIR / "src"
NOTEBOOKS = ["part3_evaluation_shreya.ipynb", "run1_team_metrics.ipynb"]
METRICS = MEMBER_DIR / "metrics_report.csv"
FULL = MEMBER_DIR / "full_metrics_report.csv"
SUBMISSION = MEMBER_DIR / "submission.csv"


def run_notebooks():
    for nb in NOTEBOOKS:
        print(f"executing {nb} ...")
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1",
                        str(SRC / nb)], check=True, cwd=SRC)


def sync():
    """full_metrics_report.csv carries the same rows as the team file (team schema, both
    directions), as in zoheb_waghu's folder."""
    shutil.copyfile(METRICS, FULL)
    print(f"wrote {FULL.relative_to(MEMBER_DIR.parents[1])}")


def check():
    ok = True
    rows = list(csv.DictReader(METRICS.open()))
    dirs = sorted(r["direction"] for r in rows)
    if dirs != ["A2B", "B2A"]:
        print(f"metrics_report.csv: expected rows A2B and B2A, found {dirs}"); ok = False
    for r in rows:
        empty = [k for k, v in r.items() if v is None or v.strip() == ""]
        if empty:
            print(f"metrics_report.csv {r['direction']}: empty cells {empty}"); ok = False
    if not FULL.exists() or FULL.read_bytes() != METRICS.read_bytes():
        print("full_metrics_report.csv missing or out of date - run `sync`"); ok = False
    sub = list(csv.DictReader(SUBMISSION.open()))
    if len(sub) != 1 or not {"FID", "MiFID"} <= set(sub[0]):
        print("submission.csv: expected one row with FID and MiFID"); ok = False
    else:
        fid, mifid = float(sub[0]["FID"]), float(sub[0]["MiFID"])
        fids = [float(r["fid"]) for r in rows]
        if abs(sum(fids) / len(fids) - fid) > 1e-3:
            print(f"submission FID {fid:.5f} != mean of metrics_report FIDs {sum(fids) / len(fids):.5f}")
            ok = False
        print(f"submission: FID {fid:.3f}, MiFID {mifid:.4f}, score (FID + MiFID) / 2 = {(fid + mifid) / 2:.2f}")
    print("all checks passed" if ok else "checks FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task 3 local evaluation (shreya_akotiya)")
    ap.add_argument("mode", choices=["metrics", "sync", "check"])
    mode = ap.parse_args().mode
    if mode == "metrics":
        run_notebooks()
        sync()
        sys.exit(0 if check() else 1)
    elif mode == "sync":
        sync()
    else:
        sys.exit(0 if check() else 1)
