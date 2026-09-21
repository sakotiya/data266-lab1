"""Task 2 training entry point.

    python task2_sentiment/zoheb_waghu/src/train.py --config .../m1_baseline_bilstm.yaml

Writes one metrics_report.csv row, a predictions CSV (for the error review),
and a raw log per run. The baseline's test predictions are reused by the
experimental runs for the paired McNemar test.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[2]))

from common.config import (config_arg_parser, describe_hardware, get_device,  # noqa: E402
                           load_config, set_seed)
from common.metrics_io import peak_memory_gb, write_metrics, write_team_row  # noqa: E402
from common.run_logger import RunLogger, new_run_id  # noqa: E402
from data import build_dataset, compute_slices, save_processed  # noqa: E402
from metrics import (bootstrap_ci, core_metrics, error_buckets,  # noqa: E402
                     expected_calibration_error, mcnemar_test, slice_metrics)
from models import build_model  # noqa: E402

CACHE_VERSION = "v1"

# team-agreed metrics_report.csv columns -> the keys this script produces.
# The team header omits the confusion-matrix cells and the per-slice columns the
# brief requires, so those go to metrics_report_extended.csv alongside it.
TEAM_SCHEMA = {
    "model_name": "model_name", "run_id": "run_id", "checkpoint": "checkpoint_id",
    "accuracy": "accuracy", "precision_macro": "precision_macro",
    "recall_macro": "recall_macro", "f1_macro": "f1_macro",
    "precision_micro": "precision_micro", "recall_micro": "recall_micro",
    "f1_micro": "f1_micro", "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted", "f1_weighted": "f1_weighted",
    "roc_auc": "roc_auc", "pr_auc": "pr_auc", "mcc": "mcc",
    "brier_score": "brier_score", "expected_calibration_error": "ece",
    "acc_ci95_low": "acc_ci_low", "acc_ci95_high": "acc_ci_high",
    "macro_f1_ci95_low": "f1_macro_ci_low", "macro_f1_ci95_high": "f1_macro_ci_high",
    "mcc_ci95_low": "mcc_ci_low", "mcc_ci95_high": "mcc_ci_high",
    "mcnemar_stat_vs_baseline": "mcnemar_stat_vs_baseline",
    "mcnemar_p_vs_baseline": "mcnemar_p_vs_baseline",
    "parameter_count": "param_count", "training_time_s": "train_seconds",
    "examples_per_sec": "examples_per_sec", "peak_memory_gb": "peak_memory_gb",
    "hardware": "hardware",
}

# config `monitor:` values -> core_metrics keys
MONITOR_KEYS = {"val_macro_f1": "f1_macro", "val_f1_macro": "f1_macro",
                "val_accuracy": "accuracy", "val_acc": "accuracy",
                "val_roc_auc": "roc_auc", "val_mcc": "mcc"}


def iterate(X, L, y, batch_size, device, shuffle, rng=None):
    idx = rng.permutation(len(y)) if shuffle else np.arange(len(y))
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        yield (torch.from_numpy(X[b]).to(device),
               torch.from_numpy(L[b]).to(device),
               torch.from_numpy(y[b]).float().to(device))


@torch.no_grad()
def predict(model, X, L, y, batch_size, device):
    model.eval()
    out = []
    for xb, lb, _ in iterate(X, L, y, batch_size, device, shuffle=False):
        out.append(torch.sigmoid(model(xb, lb)).float().cpu().numpy())
    return np.concatenate(out)


def lr_at(step, total, cfg):
    t = cfg["train"]
    if t.get("scheduler") != "cosine":
        return t["lr"]
    import math
    warm = t.get("warmup_steps", 0)
    if step < warm:
        return t["lr"] * (step + 1) / max(warm, 1)
    prog = (step - warm) / max(1, total - warm)
    return t["min_lr"] + 0.5 * (t["lr"] - t["min_lr"]) * (1 + math.cos(math.pi * min(prog, 1.0)))


def load_or_build(cfg, log):
    """Build the dataset once and cache it - all three models must consume the
    identical vocabulary and splits or the paired McNemar test is invalid."""
    cache = Path(cfg["data"]["cache_dir"])
    stamp = cache / f"ready_{CACHE_VERSION}.json"
    if stamp.exists():
        d = {}
        for split in ("train", "val", "test"):
            z = np.load(cache / f"{split}.npz")
            d[split] = (z["X"], z["L"], z["y"], None)
        d["stoi"] = json.loads((cache / "vocab.json").read_text())
        d["eda"] = json.loads((cache / "eda.json").read_text())
        d["feats"] = json.loads((cache / "feats.json").read_text())
        log.event("data_cache", status="hit", **{k: d["eda"][k] for k in
                                                 ("train_rows", "val_rows", "test_rows", "vocab_size")})
        return d
    bundle = build_dataset(cfg, log)
    save_processed(bundle, cfg["data"]["cache_dir"])
    (cache / "feats.json").write_text(json.dumps(
        {"test": bundle["test"][3], "val": bundle["val"][3]}))
    bundle["feats"] = {"test": bundle["test"][3], "val": bundle["val"][3]}
    stamp.write_text(json.dumps({"version": CACHE_VERSION}))
    log.event("data_cache", status="built", **{k: bundle["eda"][k] for k in
                                               ("train_rows", "val_rows", "test_rows", "vocab_size")})
    return bundle


def main() -> int:
    args = config_arg_parser("Task 2 - train a sentiment classifier").parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["run"]["seed"])
    device = get_device(cfg["run"]["device"])
    run_id = args.run_id or new_run_id(cfg["run"]["tag"])
    paths = cfg["paths"]
    out_dir = Path(paths["output_dir"])
    log = RunLogger(paths["log_dir"], run_id, config=cfg, hardware=describe_hardware(device))

    try:
        data = load_or_build(cfg, log)
        Xtr, Ltr, ytr, _ = data["train"]
        Xva, Lva, yva, _ = data["val"]
        Xte, Lte, yte, _ = data["test"]
        vocab_size = len(data["stoi"])
        log.event("eda", **{k: v for k, v in data["eda"].items() if not isinstance(v, list)})

        model = build_model(cfg, vocab_size).to(device)
        n_params = model.num_params()
        log.event("model", name=cfg["model"]["name"], param_count=n_params)

        t = cfg["train"]
        opt = (torch.optim.AdamW if t["optimizer"] == "adamw" else torch.optim.Adam)(
            model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
        lossf = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(cfg["run"]["seed"])
        steps_per_epoch = int(np.ceil(len(ytr) / t["batch_size"]))
        total_steps = steps_per_epoch * t["epochs"]

        best, bad_epochs, step = -np.inf, 0, 0
        ckpt = Path(paths["checkpoint_dir"]) / f"{run_id}_best.pt"
        history = []
        train_t0 = time.time()

        for epoch in range(t["epochs"]):
            model.train()
            ep_loss, nb = 0.0, 0
            for xb, lb, yb in iterate(Xtr, Ltr, ytr, t["batch_size"], device, True, rng):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total_steps, cfg)
                loss = lossf(model(xb, lb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                opt.step()
                ep_loss += loss.item()
                nb += 1
                step += 1

            val_probs = predict(model, Xva, Lva, yva, 256, device)
            vm = core_metrics(yva, val_probs)
            history.append({"epoch": epoch, "train_loss": ep_loss / nb, **{
                k: vm[k] for k in ("accuracy", "f1_macro", "roc_auc")}})
            log.event("epoch", epoch=epoch, train_loss=round(ep_loss / nb, 5),
                      val_acc=round(vm["accuracy"], 5), val_f1_macro=round(vm["f1_macro"], 5),
                      val_roc_auc=round(vm["roc_auc"], 5))

            if t["monitor"] not in MONITOR_KEYS:
                raise ValueError(f"unknown monitor {t['monitor']!r}; "
                                 f"expected one of {sorted(MONITOR_KEYS)}")
            score = vm[MONITOR_KEYS[t["monitor"]]]
            if score > best:
                best, bad_epochs = score, 0
                torch.save({"model": model.state_dict(), "config": cfg,
                            "vocab_size": vocab_size, "epoch": epoch, "run_id": run_id}, ckpt)
                log.event("checkpoint", epoch=epoch, **{t["monitor"]: round(score, 5)})
            else:
                bad_epochs += 1
                if bad_epochs >= t["early_stopping_patience"]:
                    log.event("early_stop", epoch=epoch, patience=t["early_stopping_patience"])
                    break

        train_seconds = time.time() - train_t0
        peak_mem = peak_memory_gb(device)

        # ---- final test evaluation, on the best checkpoint, ONCE -------------
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        te_probs = predict(model, Xte, Lte, yte, 256, device)
        # Two files on purpose: a stable per-tag name (the baseline lookup below
        # depends on it) and an immutable per-run name, so a second run of the
        # same config cannot silently overwrite the predictions a reported
        # metrics row was computed from.
        np.save(out_dir / f"test_probs_{cfg['run']['tag']}.npy", te_probs)
        np.save(out_dir / f"test_probs_{run_id}.npy", te_probs)

        m = core_metrics(yte, te_probs)
        m["ece"] = expected_calibration_error(yte, te_probs, cfg["eval"]["calibration_bins"])
        nb_boot = cfg["eval"]["bootstrap_samples"]
        ci = {s: bootstrap_ci(yte, te_probs, s, nb_boot, seed=cfg["run"]["seed"])
              for s in ("accuracy", "f1_macro", "mcc")}
        masks = compute_slices(data["feats"]["test"], cfg["eval"]["slices"])
        sl = slice_metrics(yte, te_probs, masks)

        # paired McNemar against the baseline's saved predictions
        base_probs_path = out_dir / "test_probs_t2_m1_baseline.npy"
        if cfg["run"]["tag"] != "t2_m1_baseline" and base_probs_path.exists():
            mc = mcnemar_test(yte, np.load(base_probs_path), te_probs)
        else:
            mc = {"statistic": "BASELINE", "pvalue": "BASELINE"}
        log.event("mcnemar", **mc)

        worst = min((k for k in sl if not np.isnan(sl[k]["f1_macro"])),
                    key=lambda k: sl[k]["f1_macro"], default=None)
        buckets = error_buckets(yte, te_probs, masks=masks, worst_slice=worst)
        (out_dir / f"error_buckets_{cfg['run']['tag']}.json").write_text(
            json.dumps({"worst_slice": worst, "buckets": buckets,
                        "slices": sl}, indent=2, default=str))
        (out_dir / f"history_{run_id}.json").write_text(json.dumps(history, indent=2))

        row = {
            "run_id": run_id, "model_name": cfg["model"]["name"],
            "config_path": cfg["_config_path"], "checkpoint_id": ckpt.name, "split": "test",
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()},
            "acc_ci_low": round(ci["accuracy"][0], 5), "acc_ci_high": round(ci["accuracy"][1], 5),
            "f1_macro_ci_low": round(ci["f1_macro"][0], 5), "f1_macro_ci_high": round(ci["f1_macro"][1], 5),
            "mcc_ci_low": round(ci["mcc"][0], 5), "mcc_ci_high": round(ci["mcc"][1], 5),
            "mcnemar_stat_vs_baseline": mc["statistic"] if isinstance(mc["statistic"], str) else round(mc["statistic"], 4),
            "mcnemar_p_vs_baseline": mc["pvalue"] if isinstance(mc["pvalue"], str) else f"{mc['pvalue']:.3e}",
            "slice_short_f1_macro": round(sl["short_reviews"]["f1_macro"], 5),
            "slice_short_error_rate": round(sl["short_reviews"]["error_rate"], 5),
            "slice_long_f1_macro": round(sl["long_reviews"]["f1_macro"], 5),
            "slice_long_error_rate": round(sl["long_reviews"]["error_rate"], 5),
            "slice_negation_f1_macro": round(sl["contains_negation"]["f1_macro"], 5),
            "slice_negation_error_rate": round(sl["contains_negation"]["error_rate"], 5),
            "slice_highpunct_f1_macro": round(sl["high_punctuation"]["f1_macro"], 5),
            "slice_highpunct_error_rate": round(sl["high_punctuation"]["error_rate"], 5),
            "param_count": n_params, "train_seconds": round(train_seconds, 2),
            "examples_per_sec": round(len(ytr) * len(history) / train_seconds, 2),
            "peak_memory_gb": round(peak_mem, 4), "device": str(device),
            "hardware": describe_hardware(device)["processor"],
        }
        write_metrics(paths["extended_metrics_csv_path"], row)
        write_team_row(paths["metrics_csv_path"], row, TEAM_SCHEMA,
                       overrides={"hardware": f"{describe_hardware(device)['processor']} / "
                                              f"{device} / torch {torch.__version__}"})
        log.event("metrics", **row)
        log.close(status="ok")
        print(json.dumps({k: row[k] for k in
                          ("run_id", "model_name", "accuracy", "f1_macro", "mcc",
                           "roc_auc", "ece", "param_count", "train_seconds")}, indent=2))
        return 0
    except Exception as exc:
        log.close(status="error", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
