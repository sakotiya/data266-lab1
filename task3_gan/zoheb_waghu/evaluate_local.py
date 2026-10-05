"""Task 3 local evaluation: every metric in full_metrics_report.csv, from images on disk.

    python task3_gan/zoheb_waghu/evaluate_local.py metrics --config <yaml> \
        --train-run-id <training run id> --checkpoint <checkpoint file name> [--zip]
    python task3_gan/zoheb_waghu/evaluate_local.py audit-sheet --config <yaml>

Direction convention is the instructor's (temp/Part3_Evaluation_Script.ipynb):
A = Monet, B = photo. pred_A2B = Monet->Photo, pred_B2A = Photo->Monet.

Inputs, all under paths.output_dir:
    pred_A2B/, pred_B2A/     translations, named like their source image (needed for LPIPS,
                             content cosine)
    cycle_A/, cycle_B/       reconstructions A->B->A and B->A->B, named like their source image
                             (needed for cycle L1; NOT_MEASURED when absent)
    human_audit_ratings.csv  sample_id,rater_id,style,content,artifacts (after `audit-sheet`)
Training log (paths.log_dir/<train-run-id>.jsonl), JSON lines the training script writes:
    {"kind": "step", "g_loss", "d_loss", "cycle_loss", "identity_loss", "grad_norm"}
    {"kind": "nan"}                                       one per non-finite step
    {"kind": "train_summary", "param_count", "train_seconds", "images_per_sec", "peak_memory_gb"}
Missing fields are recorded as NOT_MEASURED, never guessed.

FID and MiFID reproduce the instructor's script exactly (same Inception, transforms, N_EVAL cap,
index pairing), and submission.csv is written in its format. The other metrics reuse those same
Inception features. Pretrained Inception/AlexNet are used for MEASUREMENT only; they never touch
generation.
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import config_arg_parser, describe_hardware, get_device, load_config  # noqa: E402
from common.metrics_io import MISSING, read_header, write_metrics  # noqa: E402
from common.run_logger import RunLogger, new_run_id  # noqa: E402

_INCEPTION_TF = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(),
                           T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def list_images(folder) -> list[Path]:
    return sorted((p for p in Path(folder).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")), key=str)


def matched(src_dir, out_dir, n: int) -> list[tuple[Path, Path]]:
    """(source, output) pairs that share a file name, first n by sorted name."""
    out = {p.name: p for p in list_images(out_dir)}
    pairs = [(s, out[s.name]) for s in list_images(src_dir) if s.name in out][:n]
    if not pairs:
        raise FileNotFoundError(f"no file names shared by {src_dir} and {out_dir}; outputs must "
                                "be named like their source image")
    return pairs


# ---- Inception features (identical to the instructor's script) ------------------------

def inception(device):
    import torchvision.models as models
    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                            transform_input=False)
    m.fc = nn.Identity()
    return m.to(device).eval()


@torch.no_grad()
def features(model, paths, device, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([_INCEPTION_TF(Image.open(p).convert("RGB"))
                         for p in paths[i:i + batch_size]]).to(device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))   # scipy >= 1.16 dropped the `disp` argument
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def fid_mifid(real: np.ndarray, gen: np.ndarray) -> tuple[float, float]:
    """Instructor's definition: counts matched, MiFID = mean cosine distance paired by index."""
    n = min(len(real), len(gen))
    real, gen = real[:n], gen[:n]
    fid = frechet_distance(real.mean(0), np.cov(real, rowvar=False),
                           gen.mean(0), np.cov(gen, rowvar=False))
    return fid, float(np.mean([cosine(r, g) for r, g in zip(real, gen)]))


# ---- The rest of the required metrics ---------------------------------------------------

def kid(real: np.ndarray, gen: np.ndarray, subset: int, n_subsets: int, seed: int) -> tuple:
    """Kernel Inception Distance: unbiased MMD^2, cubic polynomial kernel. Returns (mean, std).
    `subset` is clipped to the smaller set (300 Monet)."""
    rng = np.random.default_rng(seed)
    m, d = min(subset, len(real), len(gen)), real.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = real[rng.choice(len(real), m, replace=False)].astype(np.float64)
        y = gen[rng.choice(len(gen), m, replace=False)].astype(np.float64)
        a, b, c = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((a.sum() - np.trace(a) + b.sum() - np.trace(b)) / (m * (m - 1))
                    - 2 * c.mean())
    return float(np.mean(vals)), float(np.std(vals))


def prdc(real: np.ndarray, gen: np.ndarray, k: int) -> dict:
    """Precision, recall, density, coverage (Naeem et al. 2020) in Inception space."""
    r, g = torch.from_numpy(real), torch.from_numpy(gen)
    rad_r = torch.cdist(r, r).kthvalue(k + 1, dim=1).values   # k+1: distance 0 is the point itself
    rad_g = torch.cdist(g, g).kthvalue(k + 1, dim=1).values
    rg = torch.cdist(r, g)                                     # (real, gen)
    in_real = rg < rad_r[:, None]
    return {"gen_precision": in_real.any(0).float().mean().item(),
            "gen_recall": (rg < rad_g[None, :]).any(1).float().mean().item(),
            "density": in_real.sum(0).float().mean().item() / k,
            "coverage": (rg.min(1).values < rad_r).float().mean().item()}


@torch.no_grad()
def lpips_distance(pairs, device, net: str) -> float:
    """Mean LPIPS between each input and its translation."""
    import lpips
    fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    to_t = lambda p: (T.ToTensor()(Image.open(p).convert("RGB")) * 2 - 1).unsqueeze(0).to(device)
    return float(np.mean([fn(to_t(s), to_t(o)).item() for s, o in pairs]))


def content_cosine(model, pairs, device, batch_size: int) -> float:
    """Mean cosine similarity of Inception features, input vs translation."""
    a = features(model, [s for s, _ in pairs], device, batch_size)
    b = features(model, [o for _, o in pairs], device, batch_size)
    return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))


def cycle_l1(pairs) -> float:
    """Mean |x - G_BA(G_AB(x))| per pixel on the [0, 1] scale."""
    load = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return float(np.mean([np.abs(load(s) - load(o)).mean() for s, o in pairs]))


def parse_training_log(path) -> dict:
    """Column values from the training log. 'final' = mean over the last 10% of steps."""
    ev = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    steps = [e for e in ev if e["kind"] == "step"]
    out: dict = {}
    if steps:
        tail = steps[-max(1, len(steps) // 10):]
        for col, key in (("gen_loss_final", "g_loss"), ("disc_loss_final", "d_loss"),
                         ("cycle_loss_final", "cycle_loss"), ("identity_loss_final", "identity_loss")):
            if key in tail[0]:
                out[col] = float(np.mean([s[key] for s in tail]))
        if "grad_norm" in steps[0]:
            out["grad_norm_mean"] = float(np.mean([s["grad_norm"] for s in steps]))
        bad = sum(1 for s in steps for v in s.values()
                  if isinstance(v, float) and not math.isfinite(v))
        out["nan_count"] = bad + sum(1 for e in ev if e["kind"] == "nan")
    for e in ev:
        if e["kind"] == "train_summary":
            out.update(parameter_count=e.get("param_count"), training_time_s=e.get("train_seconds"),
                       images_per_sec=e.get("images_per_sec"), peak_memory_gb=e.get("peak_memory_gb"))
    return {k: v for k, v in out.items() if v is not None}


# ---- Blinded human audit ------------------------------------------------------------------

def audit_sheet(cfg: dict, seed: int) -> None:
    """Build the blinded rating sheet: a fixed, shuffled sample from both directions.

    Each sheet image is **source | translation** side by side, because "content" is
    one of the three rated axes and a rater cannot judge content preservation from
    the output alone. Protocol matches shreya_akotiya/src/human_audit.py so the two
    members' audits are comparable: 15 samples per direction, sampled from the first
    300 sorted predictions (the same set the instructor's evaluator scores), seed 42,
    shuffled, file names stripped.

    Direction convention (instructor's evaluator): A = Monet, B = photo.
      A2B = Monet -> photo  (sources: data/monet_jpg, predictions: outputs/pred_A2B)
      B2A = photo -> Monet  (sources: data/photo_jpg, predictions: outputs/pred_B2A)
    """
    from PIL import Image

    a = cfg["eval"]["human_audit"]
    out = Path(cfg["paths"]["output_dir"])
    data = Path(cfg["data"]["domain_a_dir"]).parent          # task3_gan/data
    first_n = int(a.get("sample_from_first_n", 300))
    per_dir = a["n_samples"] // 2
    rng = random.Random(a.get("seed", seed))

    directions = {"A2B": (Path(cfg["data"]["domain_b_dir"]).parent / "monet_jpg",
                          out / "pred_A2B"),
                  "B2A": (data / "photo_jpg", out / "pred_B2A")}

    picks = []
    for name, (src_dir, pred_dir) in directions.items():
        sources = {p.stem: p for p in list_images(src_dir)}
        preds = [p for p in list_images(pred_dir)[:first_n] if p.stem in sources]
        if len(preds) < per_dir:
            raise SystemExit(f"{pred_dir}: only {len(preds)} predictions have a matching source")
        picks += [(name, sources[p.stem], p) for p in rng.sample(preds, per_dir)]
    rng.shuffle(picks)

    sheet = out / "human_audit"
    sheet.mkdir(parents=True, exist_ok=True)
    with (out / "human_audit_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "direction", "source_file", "prediction_file"])
        for i, (d, src, pred) in enumerate(picks, 1):
            left = Image.open(src).convert("RGB").resize((256, 256))
            right = Image.open(pred).convert("RGB").resize((256, 256))
            pair = Image.new("RGB", (522, 256), "white")      # 10 px white gap
            pair.paste(left, (0, 0))
            pair.paste(right, (266, 0))
            pair.save(sheet / f"sample_{i:02d}.jpg", quality=95)
            w.writerow([f"sample_{i:02d}", d, src.name, pred.name])
    with (out / "human_audit_ratings.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "rater_id", *a["axes"]])
        for r in range(1, a["n_raters"] + 1):
            for i in range(1, len(picks) + 1):
                w.writerow([f"sample_{i:02d}", f"rater{r}"])
    print(f"wrote {len(picks)} blinded samples to {sheet}\n"
          "Left half = original input, right half = my model's translation.\n"
          "Give raters ONLY that folder and human_audit_ratings.csv (scores 1-5). "
          "Keep human_audit_manifest.csv until rating is done.")



def audit_summary(out: Path, axes: list) -> dict:
    """Per-direction mean score per axis, plus Cohen's kappa (mean over axes) and % agreement."""
    from sklearn.metrics import cohen_kappa_score
    import pandas as pd
    ratings = pd.read_csv(out / "human_audit_ratings.csv")
    manifest = pd.read_csv(out / "human_audit_manifest.csv")
    if ratings[axes].isna().any().any():
        raise ValueError("human_audit_ratings.csv still has empty scores")
    raters = sorted(ratings.rater_id.unique())
    if len(raters) != 2:
        raise ValueError(f"kappa needs exactly 2 raters, found {raters}")
    wide = {ax: ratings.pivot(index="sample_id", columns="rater_id", values=ax) for ax in axes}
    kappas = [cohen_kappa_score(w[raters[0]], w[raters[1]]) for w in wide.values()]
    agree = [float((w[raters[0]] == w[raters[1]]).mean()) for w in wide.values()]
    merged = ratings.merge(manifest, on="sample_id")
    res = {"kappa": float(np.mean(kappas)), "pct_agreement": float(np.mean(agree))}
    for d in ("A2B", "B2A"):
        res[d] = {ax: float(merged[merged.direction == d][ax].mean()) for ax in axes}
    return res


# ---- Entry point ----------------------------------------------------------------------------

def run_metrics(args, cfg: dict, device, log: RunLogger) -> int:
    e, out = cfg["eval"], Path(cfg["paths"]["output_dir"])
    n, bs = e["n_eval"], e["feature_batch_size"]
    real_a, real_b = cfg["data"]["domain_a_dir"], cfg["data"]["domain_b_dir"]   # Monet, photo
    net = inception(device)
    feat = lambda paths: features(net, paths, device, bs)

    # (name, real target set, source images, generated dir, cycle-reconstruction dir)
    directions = [("A2B", real_b, real_a, out / "pred_A2B", out / "cycle_A"),
                  ("B2A", real_a, real_b, out / "pred_B2A", out / "cycle_B")]
    train = parse_training_log(Path(cfg["paths"]["log_dir"]) / f"{args.train_run_id}.jsonl")
    audit = None
    if (out / "human_audit_ratings.csv").exists():
        try:
            audit = audit_summary(out, e["human_audit"]["axes"])
        except ValueError as exc:
            log.event("audit_skipped", reason=str(exc))
    hw = describe_hardware(device)
    hardware = f"{hw.get('gpu_name', hw['processor'])} / {device} / torch {torch.__version__}"

    header, rows, fids, mifids = read_header(cfg["paths"]["metrics_csv_path"]), [], [], []
    for name, target, src, gen_dir, cyc_dir in directions:
        real_f = feat(list_images(target)[:n])
        gen_paths = list_images(gen_dir)[:n]
        gen_f = feat(gen_paths)
        fid, mifid = fid_mifid(real_f, gen_f)
        kid_mean, kid_std = kid(real_f, gen_f, e["kid_subset_size"], e["kid_subsets"],
                                cfg["run"]["seed"])
        pairs = matched(src, gen_dir, n)
        row = {c: MISSING for c in header}
        row.update(direction=name, run_id=args.train_run_id, checkpoint=args.checkpoint,
                   fid=fid, kid=kid_mean, lpips=lpips_distance(pairs, device, e["lpips_net"]),
                   content_cosine=content_cosine(net, pairs, device, bs), hardware=hardware,
                   kaggle_public=args.kaggle_public, kaggle_private=args.kaggle_private,
                   leaderboard_rank=args.rank, **prdc(real_f, gen_f, e["nn_k"]), **train)
        if cyc_dir.is_dir():
            row["cycle_l1"] = cycle_l1(matched(src, cyc_dir, e["cycle_l1_samples"]))
        if audit:
            row.update(human_audit_style=audit[name]["style"], human_audit_content=audit[name]["content"],
                       human_audit_artifacts=audit[name]["artifacts"], inter_rater_kappa=audit["kappa"])
        row = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
        log.event("direction", direction=name, n_real=len(real_f), n_gen=len(gen_f),
                  mifid=round(mifid, 5), kid_std=round(kid_std, 6), **{k: row[k] for k in
                  ("fid", "kid", "lpips", "content_cosine", "cycle_l1")})
        rows.append(row)
        fids.append(fid)
        mifids.append(mifid)

    for row in rows:
        write_metrics(cfg["paths"]["metrics_csv_path"], row)
        write_metrics(cfg["paths"]["team_metrics_csv_path"], row)
    with Path(cfg["paths"]["submission_csv_path"]).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "FID", "MiFID"])
        w.writerow([1, float(np.mean(fids)), float(np.mean(mifids))])
    if args.zip:
        sub = list_images(out / "pred_B2A")
        with zipfile.ZipFile(out / "images.zip", "w", zipfile.ZIP_STORED) as z:
            for p in sub:
                z.write(p, p.name)
        log.event("zip", n_images=len(sub), path=str(out / "images.zip"))
    print(json.dumps({r["direction"]: {k: r[k] for k in ("fid", "kid", "gen_precision", "gen_recall",
                                                        "density", "coverage", "lpips",
                                                        "content_cosine", "cycle_l1")}
                      for r in rows}, indent=2))
    print(f"submission.csv: FID {np.mean(fids):.3f}  MiFID {np.mean(mifids):.4f}")
    return 0


def main() -> int:
    ap = config_arg_parser("Task 3 local evaluation")
    ap.add_argument("mode", choices=["metrics", "audit-sheet"])
    ap.add_argument("--train-run-id", help="run id of the training run (metrics)")
    ap.add_argument("--checkpoint", help="checkpoint file name behind these images (metrics)")
    ap.add_argument("--zip", action="store_true", help="also build outputs/images.zip from pred_B2A")
    ap.add_argument("--kaggle-public", default=MISSING)
    ap.add_argument("--kaggle-private", default=MISSING)
    ap.add_argument("--rank", default=MISSING)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode == "audit-sheet":
        audit_sheet(cfg, cfg["run"]["seed"])
        return 0
    if not (args.train_run_id and args.checkpoint):
        ap.error("metrics needs --train-run-id and --checkpoint")
    device = get_device(cfg["run"]["device"])
    log = RunLogger(cfg["paths"]["log_dir"], args.run_id or new_run_id(cfg["run"]["tag"] + "_eval"),
                    config=cfg, hardware=describe_hardware(device))
    try:
        rc = run_metrics(args, cfg, device, log)
        log.close(status="ok")
        return rc
    except Exception as exc:
        log.close(status="error", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
