"""Task 3 CycleGAN training. Config-driven; nothing here is a hyperparameter.

    python task3_gan/zoheb_waghu/src/train.py --config task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml
    ... --max-steps 300      short smoke/benchmark run (LR schedule still follows the full config)
    ... --resume <run>_last.pt   continue an interrupted run at the next epoch; use the SAME --run-id
                                 to keep one log (the image pools restart empty)
Checkpoints: <run>_last.pt after every epoch, <run>_epochNNN.pt every snapshot_interval_epochs,
<run>_final.pt (weights only) at the end. All under checkpoint_dir, all gitignored.

A = Monet, B = photo. G_AB: Monet -> photo, G_BA: photo -> Monet, D_A judges Monet, D_B photos.
Generator objective: adversarial (both directions) + lambda * cycle L1 + lambda * identity_weight * identity L1.
Discriminator objective: 0.5 * (real vs. past fakes) per domain. Logs use the events evaluate_local.py reads.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[2]))

from common.config import (config_arg_parser, describe_hardware, get_device,  # noqa: E402
                           load_config, set_seed)
from common.metrics_io import peak_memory_gb  # noqa: E402
from common.run_logger import RunLogger, new_run_id  # noqa: E402
from data import ImagePool, load_domain, random_batch  # noqa: E402
from model import build_networks  # noqa: E402


def lr_multiplier(epoch: int, t: dict) -> float:
    """1.0 for `epochs` epochs, then linear decay towards 0 over `epochs_decay` epochs."""
    return 1.0 - max(0, epoch - t["epochs"] + 1) / (t["epochs_decay"] + 1)


def grad_norm(params) -> float:
    return torch.nn.utils.clip_grad_norm_(params, float("inf")).item()   # measure only, never clips


def main() -> int:
    ap = config_arg_parser("Task 3 - train CycleGAN")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["run"]["seed"])
    device = get_device(cfg["run"]["device"])
    torch.backends.cudnn.benchmark = True                      # fixed input size, faster kernels
    d, m, ls, t = cfg["data"], cfg["model"], cfg["loss"], cfg["train"]
    paths = cfg["paths"]
    run_id = args.run_id or new_run_id(cfg["run"]["tag"] + ("_smoke" if args.max_steps else ""))
    log = RunLogger(paths["log_dir"], run_id, config=cfg, hardware=describe_hardware(device))
    out = Path(paths["output_dir"])
    (out / "train_samples").mkdir(parents=True, exist_ok=True)
    Path(paths["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    try:
        imgs_a, _ = load_domain(d["domain_a_dir"], d["load_size"], device)
        imgs_b, _ = load_domain(d["domain_b_dir"], d["load_size"], device)
        n_a, n_b, bs = len(imgs_a), len(imgs_b), t["batch_size"]
        n_epoch = n_b if d["epoch_definition"] == "domain_b" else n_a      # the larger domain sets an epoch
        steps_per_epoch = n_epoch // bs
        total_epochs = t["epochs"] + t["epochs_decay"]

        nets = {k: v.to(device) for k, v in build_networks(cfg).items()}
        g_params = itertools.chain(nets["g_ab"].parameters(), nets["g_ba"].parameters())
        d_params = itertools.chain(nets["d_a"].parameters(), nets["d_b"].parameters())
        params_g, params_d = list(g_params), list(d_params)
        opt_g = torch.optim.Adam(params_g, lr=t["lr"], betas=tuple(t["betas"]))
        opt_d = torch.optim.Adam(params_d, lr=t["lr"], betas=tuple(t["betas"]))
        n_params = sum(p.numel() for p in params_g + params_d)
        log.event("setup", n_a=n_a, n_b=n_b, steps_per_epoch=steps_per_epoch, total_epochs=total_epochs,
                  param_count=n_params, g_params=sum(p.numel() for p in params_g))

        gan_loss = nn.MSELoss() if ls["gan_mode"] == "lsgan" else nn.BCEWithLogitsLoss()
        adv = lambda pred, real: gan_loss(pred, torch.full_like(pred, 1.0 if real else 0.0))
        l1 = nn.L1Loss()
        pool_a, pool_b = ImagePool(ls["image_pool_size"]), ImagePool(ls["image_pool_size"])
        g_ab, g_ba, d_a, d_b = nets["g_ab"], nets["g_ba"], nets["d_a"], nets["d_b"]

        step, start_epoch = 0, 0
        if args.resume:
            ck = torch.load(args.resume, map_location=device)
            for k in nets:
                nets[k].load_state_dict(ck["nets"][k])
            opt_g.load_state_dict(ck["opt_g"])
            opt_d.load_state_dict(ck["opt_d"])
            step, start_epoch = ck["step"], ck["epoch"] + 1
            log.event("resume", checkpoint=args.resume, epoch=start_epoch, step=step)

        step0, train_t0, done = step, time.time(), False       # step0: steps done before this session
        for epoch in range(start_epoch, total_epochs):
            for opt in (opt_g, opt_d):
                for g in opt.param_groups:
                    g["lr"] = t["lr"] * lr_multiplier(epoch, t)
            order = torch.randperm(n_epoch).tolist()
            for k in range(steps_per_epoch):
                sel = order[k * bs:(k + 1) * bs]
                idx_a = sel if n_epoch == n_a else torch.randint(0, n_a, (bs,)).tolist()
                idx_b = sel if n_epoch == n_b else torch.randint(0, n_b, (bs,)).tolist()
                real_a = random_batch(imgs_a, idx_a, d["image_size"], d["random_flip"])
                real_b = random_batch(imgs_b, idx_b, d["image_size"], d["random_flip"])

                fake_b, fake_a = g_ab(real_a), g_ba(real_b)
                rec_a, rec_b = g_ba(fake_b), g_ab(fake_a)

                # ---- generators: fool both discriminators, reconstruct, keep colours on own domain
                for net in (d_a, d_b):
                    net.requires_grad_(False)
                l_adv = adv(d_b(fake_b), True) + adv(d_a(fake_a), True)
                c_a, c_b = l1(rec_a, real_a), l1(rec_b, real_b)
                i_a = i_b = torch.zeros((), device=device)
                if ls["lambda_identity"] > 0:
                    i_a, i_b = l1(g_ba(real_a), real_a), l1(g_ab(real_b), real_b)
                loss_g = (l_adv + ls["lambda_cycle_a"] * c_a + ls["lambda_cycle_b"] * c_b
                          + ls["lambda_identity"] * (ls["lambda_cycle_a"] * i_a + ls["lambda_cycle_b"] * i_b))
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                gn = grad_norm(params_g)
                opt_g.step()

                # ---- discriminators: real vs. past fakes from the pool
                for net in (d_a, d_b):
                    net.requires_grad_(True)
                fa, fb = pool_a.query(fake_a), pool_b.query(fake_b)
                loss_d = (0.5 * (adv(d_a(real_a), True) + adv(d_a(fa.detach()), False))
                          + 0.5 * (adv(d_b(real_b), True) + adv(d_b(fb.detach()), False)))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                step += 1

                if step % t["log_interval_steps"] == 0:
                    vals = dict(g_loss=loss_g.item(), d_loss=loss_d.item(),
                                cycle_loss=(c_a + c_b).item(),        # unweighted L1 sum, both directions
                                identity_loss=(i_a + i_b).item(), grad_norm=gn)
                    if not all(map(torch.isfinite, map(torch.tensor, vals.values()))):
                        log.event("nan", step=step)
                        raise RuntimeError(f"non-finite loss at step {step}: {vals}")
                    log.event("step", step=step, epoch=epoch,
                              lr=round(opt_g.param_groups[0]["lr"], 8), **{k2: round(v, 5) for k2, v in vals.items()})
                if step % t["sample_interval_steps"] == 0:
                    grid = torch.cat([torch.cat([real_a[:1], fake_b[:1], rec_a[:1]], 3),
                                      torch.cat([real_b[:1], fake_a[:1], rec_b[:1]], 3)], 2)
                    save_image(grid, out / "train_samples" / f"step_{step:07d}.jpg", normalize=True, value_range=(-1, 1))
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
            log.event("epoch", epoch=epoch, step=step, seconds=round(time.time() - train_t0, 1))
            if done or (epoch + 1) % t["checkpoint_interval_epochs"] == 0:
                # Full state (nets + optimisers) for --resume. Written to a temp file first, then
                # swapped in, so a crash mid-save cannot corrupt the previous checkpoint.
                ck = {"nets": {k2: v.state_dict() for k2, v in nets.items()}, "opt_g": opt_g.state_dict(),
                      "opt_d": opt_d.state_dict(), "epoch": epoch, "step": step, "config": cfg}
                ck_dir = Path(paths["checkpoint_dir"])
                torch.save(ck, ck_dir / f"{run_id}_last.tmp")
                (ck_dir / f"{run_id}_last.tmp").replace(ck_dir / f"{run_id}_last.pt")
                if (epoch + 1) % t["snapshot_interval_epochs"] == 0:      # permanent, never overwritten
                    torch.save(ck, ck_dir / f"{run_id}_epoch{epoch + 1:03d}.pt")
                log.event("checkpoint", epoch=epoch + 1, step=step)
            if done:
                break

        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - train_t0
        final = Path(paths["checkpoint_dir"]) / f"{run_id}_final.pt"
        torch.save({"nets": {k2: v.state_dict() for k2, v in nets.items()}, "epoch": epoch, "step": step,
                    "config": cfg, "run_id": run_id}, final)
        log.event("train_summary", param_count=n_params, train_seconds=round(seconds, 2),
                  images_per_sec=round((step - step0) * bs / seconds, 2),   # this session only
                  peak_memory_gb=round(peak_memory_gb(device), 4), steps=step, checkpoint=final.name)
        log.close(status="ok")
        print(f"done: {step} steps ({step - step0} this session) in {seconds:.0f}s -> {final.name}")
        return 0
    except Exception as exc:
        log.close(status="error", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
