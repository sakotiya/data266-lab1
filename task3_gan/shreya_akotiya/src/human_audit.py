"""Blinded human audit for Task 3 (30 fixed samples, 2 raters, Cohen's kappa).

Same protocol as the team (zoheb_waghu/evaluate_local.py audit-sheet): 15 samples per direction,
picked with a fixed seed, shuffled, file names stripped. Each sample image shows the source on the
left and the translation on the right, so raters can judge content as well as style.

Direction names follow the instructor/team: A = Monet, B = photo, so
A2B = Monet -> photo (my outputs/pred_B2A) and B2A = photo -> Monet (my outputs/pred_A2B).

Usage (run from the repo root, e.g. in Colab where the predictions are):
    python task3_gan/shreya_akotiya/src/human_audit.py make
    # raters fill outputs/human_audit_ratings.csv (scores 1-5)
    python task3_gan/shreya_akotiya/src/human_audit.py score
"""

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image

MEMBER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MEMBER_DIR.parent / "data"
OUT_DIR = MEMBER_DIR / "outputs"
AXES = ["style", "content", "artifacts"]
N_SAMPLES, N_RATERS, SEED = 30, 2, 42

# team name -> (source images, my prediction folder)
DIRECTIONS = {"A2B": (DATA_DIR / "monet_jpg", OUT_DIR / "pred_B2A"),   # Monet -> photo
              "B2A": (DATA_DIR / "photo_jpg", OUT_DIR / "pred_A2B")}   # photo -> Monet


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def make(out_dir):
    rng = random.Random(SEED)
    picks = []
    for name, (src_dir, pred_dir) in DIRECTIONS.items():
        sources = {p.stem: p for p in list_images(src_dir)}
        preds = [p for p in list_images(pred_dir) if p.stem in sources]
        if len(preds) < N_SAMPLES // 2:
            raise SystemExit(f"{pred_dir}: only {len(preds)} predictions with a matching source")
        picks += [(name, sources[p.stem], p) for p in rng.sample(preds, N_SAMPLES // 2)]
    rng.shuffle(picks)

    sheet = out_dir / "human_audit"
    sheet.mkdir(parents=True, exist_ok=True)
    with (out_dir / "human_audit_manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "direction", "source_file", "prediction_file"])
        for i, (name, src, pred) in enumerate(picks, 1):
            a = Image.open(src).convert("RGB").resize((256, 256))
            b = Image.open(pred).convert("RGB").resize((256, 256))
            pair = Image.new("RGB", (522, 256), "white")   # 10 px white gap
            pair.paste(a, (0, 0))
            pair.paste(b, (266, 0))
            pair.save(sheet / f"sample_{i:02d}.jpg", quality=95)
            w.writerow([f"sample_{i:02d}", name, src.name, pred.name])
    with (out_dir / "human_audit_ratings.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "rater_id", *AXES])
        for r in range(1, N_RATERS + 1):
            for i in range(1, len(picks) + 1):
                w.writerow([f"sample_{i:02d}", f"rater{r}"])
    print(f"Wrote {len(picks)} blinded samples to {sheet}\n"
          f"Give raters ONLY that folder, human_audit_ratings.csv and RATING_GUIDE.md.\n"
          f"Keep human_audit_manifest.csv away from them until rating is done.")


def score(out_dir):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    ratings = pd.read_csv(out_dir / "human_audit_ratings.csv")
    manifest = pd.read_csv(out_dir / "human_audit_manifest.csv")
    if ratings[AXES].isna().any().any():
        raise SystemExit("human_audit_ratings.csv still has empty scores")
    raters = sorted(ratings.rater_id.unique())
    if len(raters) != 2:
        raise SystemExit(f"kappa needs exactly 2 raters, found {raters}")

    res = {"raters": raters, "per_axis": {}, "per_direction": {}}
    for ax in AXES:
        wide = ratings.pivot(index="sample_id", columns="rater_id", values=ax)
        res["per_axis"][ax] = {
            "rater1_mean": float(wide[raters[0]].mean()), "rater2_mean": float(wide[raters[1]].mean()),
            "kappa": float(cohen_kappa_score(wide[raters[0]], wide[raters[1]])),
            "kappa_weighted": float(cohen_kappa_score(wide[raters[0]], wide[raters[1]], weights="linear")),
            "pct_agreement": float((wide[raters[0]] == wide[raters[1]]).mean()),
            "pct_within_1": float(((wide[raters[0]] - wide[raters[1]]).abs() <= 1).mean())}
    merged = ratings.merge(manifest, on="sample_id")
    for d in DIRECTIONS:
        res["per_direction"][d] = {ax: float(merged[merged.direction == d][ax].mean()) for ax in AXES}
    res["kappa_mean"] = float(np.mean([v["kappa"] for v in res["per_axis"].values()]))
    res["pct_agreement_mean"] = float(np.mean([v["pct_agreement"] for v in res["per_axis"].values()]))

    (out_dir / "human_audit_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "score"])
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    make(args.out_dir) if args.mode == "make" else score(args.out_dir)
