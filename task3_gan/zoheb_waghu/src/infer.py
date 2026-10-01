"""Translate every image with a trained CycleGAN and write what evaluate_local.py reads.

    python task3_gan/zoheb_waghu/src/infer.py --config <yaml> --checkpoint <run>_final.pt

Monet  -> pred_A2B/ (photo)  and cycle_A/ (Monet -> photo -> Monet)
Photo  -> pred_B2A/ (Monet)  and cycle_B/ (photo -> Monet -> photo)
Files keep their source names. Only the two generators are used - no other model touches the images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[2]))

from common.config import config_arg_parser, get_device, load_config  # noqa: E402
from model import build_networks  # noqa: E402


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def translate(fwd, back, src_dir, pred_dir: Path, cycle_dir: Path, device, bs: int) -> int:
    pred_dir.mkdir(parents=True, exist_ok=True)
    cycle_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(src_dir).glob("*.jpg"))
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1)
                         for p in chunk]).float().div(127.5).sub(1).to(device)
        fake = fwd(x)
        for p, a, b in zip(chunk, to_uint8(fake), to_uint8(back(fake))):
            Image.fromarray(a).save(pred_dir / p.name, quality=95)
            Image.fromarray(b).save(cycle_dir / p.name, quality=95)
    return len(files)


def main() -> int:
    ap = config_arg_parser("Task 3 - inference")
    ap.add_argument("--checkpoint", required=True, help="file name in checkpoint_dir, or a path")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["run"]["device"])
    ck_path = Path(args.checkpoint)
    ck = torch.load(ck_path if ck_path.exists() else Path(cfg["paths"]["checkpoint_dir"]) / ck_path.name,
                    map_location=device)
    nets = build_networks(cfg)
    for k in ("g_ab", "g_ba"):
        nets[k].load_state_dict(ck["nets"][k])
        nets[k].to(device).eval()
    out = Path(cfg["paths"]["output_dir"])
    na = translate(nets["g_ab"], nets["g_ba"], cfg["data"]["domain_a_dir"], out / "pred_A2B", out / "cycle_A", device, args.batch_size)
    nb = translate(nets["g_ba"], nets["g_ab"], cfg["data"]["domain_b_dir"], out / "pred_B2A", out / "cycle_B", device, args.batch_size)
    print(f"wrote {na} Monet->photo and {nb} photo->Monet translations (+ cycle reconstructions) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
