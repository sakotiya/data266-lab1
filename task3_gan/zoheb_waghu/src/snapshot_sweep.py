"""Score every permanent snapshot of a training run with the leaderboard formula.

    python task3_gan/zoheb_waghu/src/snapshot_sweep.py --config <yaml> --train-run-id <training run id>

For each <run_id>_epochNNN.pt in checkpoint_dir: translate the first eval.n_eval images of each
domain (the set the instructor's script scores) and compute FID and MiFID per direction exactly as
evaluate_local.py does. One row per snapshot goes to outputs/snapshot_metrics.csv, with
leaderboard_proxy = (mean FID + mean MiFID) / 2. Translations are written to a temporary folder
and deleted; outputs/pred_* are never touched.
"""
from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parent))
sys.path.insert(0, str(SRC.parents[2]))

from common.config import config_arg_parser, get_device, load_config  # noqa: E402
from evaluate_local import features, fid_mifid, inception, list_images  # noqa: E402
from infer import translate  # noqa: E402
from model import build_networks  # noqa: E402

COLUMNS = ["epoch", "a2b_fid", "a2b_mifid", "b2a_fid", "b2a_mifid", "mean_fid", "mean_mifid",
           "leaderboard_proxy"]


def main() -> int:
    ap = config_arg_parser("Task 3 - score training snapshots")
    ap.add_argument("--train-run-id", required=True)
    ap.add_argument("--out", help="CSV path (default: <output_dir>/snapshot_metrics.csv)")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["run"]["device"])
    n, bs = cfg["eval"]["n_eval"], cfg["eval"]["feature_batch_size"]
    snaps = sorted(Path(cfg["paths"]["checkpoint_dir"]).glob(f"{args.train_run_id}_epoch*.pt"))
    if not snaps:
        raise FileNotFoundError(f"no {args.train_run_id}_epochNNN.pt snapshots in checkpoint_dir")

    net = inception(device)
    real_a = list_images(cfg["data"]["domain_a_dir"])[:n]          # Monet
    real_b = list_images(cfg["data"]["domain_b_dir"])[:n]          # photo
    feat_a, feat_b = features(net, real_a, device, bs), features(net, real_b, device, bs)
    nets = build_networks(cfg)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, files in (("src_A", real_a), ("src_B", real_b)):
            (tmp / name).mkdir()
            for p in files:
                shutil.copy(p, tmp / name / p.name)
        for snap in snaps:
            ck = torch.load(snap, map_location=device)
            for k in ("g_ab", "g_ba"):
                nets[k].load_state_dict(ck["nets"][k])
                nets[k].to(device).eval()
            translate(nets["g_ab"], nets["g_ba"], tmp / "src_A", tmp / "pred_A2B", tmp / "cycle_A", device, args.batch_size)
            translate(nets["g_ba"], nets["g_ab"], tmp / "src_B", tmp / "pred_B2A", tmp / "cycle_B", device, args.batch_size)
            a2b = fid_mifid(feat_b, features(net, list_images(tmp / "pred_A2B")[:n], device, bs))
            b2a = fid_mifid(feat_a, features(net, list_images(tmp / "pred_B2A")[:n], device, bs))
            mean_fid, mean_mifid = (a2b[0] + b2a[0]) / 2, (a2b[1] + b2a[1]) / 2
            rows.append([ck["epoch"] + 1, *a2b, *b2a, mean_fid, mean_mifid, (mean_fid + mean_mifid) / 2])
            print(f"epoch {rows[-1][0]:3d}  FID A2B {a2b[0]:.3f}  B2A {b2a[0]:.3f}  proxy {rows[-1][-1]:.3f}")
            del ck

    out = Path(args.out) if args.out else Path(cfg["paths"]["output_dir"]) / "snapshot_metrics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows([r[0], *(f"{v:.6f}" for v in r[1:])] for r in rows)
    print(f"wrote {len(rows)} rows to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
