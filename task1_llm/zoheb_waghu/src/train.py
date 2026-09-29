"""Task 1 training entry point. Config-driven; no hyperparameters live here.

    python task1_llm/zoheb_waghu/src/train.py --config task1_llm/zoheb_waghu/configs/gpt_baseline.yaml
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC.parents[2]))

from common.config import (config_arg_parser, describe_hardware, get_device,  # noqa: E402
                           load_config, set_seed)
from common.metrics_io import peak_memory_gb, write_metrics, write_team_row  # noqa: E402
from common.run_logger import RunLogger, new_run_id  # noqa: E402
from data import SequenceBatcher, build_splits, save_processed  # noqa: E402
from evaluate import (causal_mask_probe, eval_loss_and_accuracy,  # noqa: E402
                      generation_metrics, gradient_stability)


# team-agreed metrics_report.csv columns -> the keys this script produces
TEAM_SCHEMA = {
    "run_id": "run_id", "checkpoint": "checkpoint_id", "train_ce_loss": "train_loss",
    "val_ce_loss": "val_loss", "perplexity": "perplexity",
    "bits_per_character": "bits_per_char", "generalization_gap": "generalization_gap",
    "top1_next_char_accuracy": "top1_next_char_acc", "distinct_1": "distinct_1",
    "distinct_2": "distinct_2", "distinct_3": "distinct_3",
    "repeated_4gram_rate": "repeated_4gram_rate", "grad_norm_mean": "grad_norm_mean",
    "grad_norm_max": "grad_norm_max", "loss_spikes": "loss_spikes",
    "nan_count": "nan_count", "parameter_count": "param_count",
    "train_tokens_per_sec": "train_tokens_per_sec",
    "gen_tokens_per_sec": "gen_tokens_per_sec", "peak_memory_gb": "peak_memory_gb",
    "total_training_time_s": "total_train_seconds", "hardware": "hardware",
}


def lr_at(step: int, cfg: dict) -> float:
    """Linear warm-up then cosine decay to min_lr (workplan 1.3)."""
    t = cfg["train"]
    warm, lr, min_lr = t["warmup_steps"], t["lr"], t["min_lr"]
    total = t["_total_steps"]
    if step < warm:
        return lr * (step + 1) / warm
    if t["schedule"] != "cosine":
        return lr
    progress = (step - warm) / max(1, total - warm)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def main() -> int:
    args = config_arg_parser("Task 1 - train GPT from scratch").parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["run"]["seed"])
    device = get_device(cfg["run"]["device"])
    run_id = args.run_id or new_run_id(cfg["run"]["tag"])
    paths = cfg["paths"]
    log = RunLogger(paths["log_dir"], run_id, config=cfg, hardware=describe_hardware(device))
    t_start = time.time()

    try:
        # ---- data -------------------------------------------------------
        splits = build_splits(cfg)
        stats = save_processed(splits, cfg["data"]["cache_dir"])
        log.event("data", **stats)
        vocab = splits["vocab"]

        bs, blk = cfg["train"]["batch_size"], cfg["data"]["block_size"]
        train_b = SequenceBatcher(splits["train_ids"], blk, bs, device, cfg["run"]["seed"])
        val_b = SequenceBatcher(splits["val_ids"], blk, bs, device, cfg["run"]["seed"])

        # One epoch = one pass over every training sequence. With 100K sequences
        # at batch 32 that is 3,125 optimiser steps, so no override is needed;
        # `steps_per_epoch` stays supported only for smoke configs.
        steps_per_epoch = (cfg["train"].get("steps_per_epoch")
                           or cfg["data"]["train_sequences"] // cfg["train"]["batch_size"])
        total_steps = steps_per_epoch * cfg["train"]["epochs"]
        cfg["train"]["_total_steps"] = total_steps
        tokens_per_epoch = steps_per_epoch * bs * blk
        log.event("schedule", steps_per_epoch=steps_per_epoch, total_steps=total_steps,
                  tokens_per_epoch=tokens_per_epoch,
                  train_sequences=cfg["data"]["train_sequences"],
                  corpus_passes_per_epoch=round(tokens_per_epoch / len(splits["train_ids"]), 2),
                  nonoverlapping_batches_per_epoch=train_b.batches_per_epoch)

        # ---- model ------------------------------------------------------
        from model import GPT
        m = cfg["model"]
        model = GPT(vocab.size, m["n_layer"], m["n_head"], m["n_embd"], blk,
                    m["dropout"], m["bias"], m["tie_weights"]).to(device)
        n_params = model.num_params()
        log.event("model", param_count=n_params, vocab_size=vocab.size)

        probe = causal_mask_probe(model, vocab.size, blk, device)
        log.event("causal_mask_probe", **probe)
        if not probe["passed"]:
            raise RuntimeError(f"causal mask leaks information: {probe}")

        t = cfg["train"]
        opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=tuple(t["betas"]),
                                weight_decay=t["weight_decay"])

        # ---- train ------------------------------------------------------
        history, grad_norms, step_losses = [], [], []
        best_val, step = float("inf"), 0
        ckpt_path = Path(paths["checkpoint_dir"]) / f"{run_id}_best.pt"
        train_t0 = time.time()

        for epoch in range(t["epochs"]):
            model.train()
            ep_loss, ep_batches = 0.0, 0
            for _ in range(steps_per_epoch):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, cfg)
                x, y = train_b.sample()
                _, loss = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                opt.step()

                lv = loss.item()
                if math.isnan(lv) or math.isinf(lv):
                    log.event("nan", step=step)
                    raise RuntimeError(f"non-finite loss at step {step}")
                ep_loss += lv
                ep_batches += 1
                step_losses.append(lv)
                grad_norms.append(float(gn))
                if step % t["log_interval_steps"] == 0:
                    log.event("step", step=step, epoch=epoch, loss=round(lv, 4),
                              lr=round(lr_at(step, cfg), 6), grad_norm=round(float(gn), 4))
                step += 1

            train_loss = ep_loss / max(ep_batches, 1)
            val = eval_loss_and_accuracy(model, val_b)
            history.append({"epoch": epoch, "train_loss": train_loss, **val})
            log.event("epoch", epoch=epoch, train_loss=round(train_loss, 4),
                      val_loss=round(val["loss"], 4), val_ppl=round(val["perplexity"], 3),
                      val_bpc=round(val["bits_per_char"], 4),
                      val_top1=round(val["top1_next_char_acc"], 4))
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save({"model": model.state_dict(), "config": cfg,
                            "vocab_size": vocab.size, "epoch": epoch,
                            "val_loss": val["loss"], "run_id": run_id}, ckpt_path)
                log.event("checkpoint", path=str(ckpt_path), epoch=epoch,
                          val_loss=round(val["loss"], 4))

        train_seconds = time.time() - train_t0
        tokens_seen = step * bs * blk
        peak_mem = peak_memory_gb(device)

        # ---- final eval on the best checkpoint --------------------------
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        final_val = eval_loss_and_accuracy(model, val_b)
        final_train = eval_loss_and_accuracy(model, train_b)

        # ---- generation -------------------------------------------------
        from generate import run_generation
        samples, gen_tps = run_generation(model, vocab, cfg, device)
        Path(paths["sample_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(paths["sample_dir"]) / f"samples_{run_id}.txt").write_text(
            "\n\n".join(f"### {s['label']} | prompt={s['prompt']!r}\n{s['text']}"
                        for s in samples))
        gen_m = generation_metrics([s["text"] for s in samples],
                                   cfg["eval"]["ngram_max"], cfg["eval"]["repeat_ngram_n"])
        log.event("generation", n_samples=len(samples),
                  gen_tokens_per_sec=round(gen_tps, 2), **{k: round(v, 4) for k, v in gen_m.items()})

        stab = gradient_stability(grad_norms, step_losses)
        hist_path = Path(paths["output_dir"]) / f"history_{run_id}.json"
        hist_path.write_text(json.dumps(
            {"history": history, "step_losses": step_losses,
             "grad_norms": grad_norms, "causal_probe": probe}, indent=2))

        row = {
            "run_id": run_id, "model_name": cfg["run"]["tag"],
            "config_path": cfg["_config_path"], "checkpoint_id": ckpt_path.name,
            "train_loss": round(final_train["loss"], 5),
            "val_loss": round(final_val["loss"], 5),
            "perplexity": round(final_val["perplexity"], 4),
            "bits_per_char": round(final_val["bits_per_char"], 5),
            "generalization_gap": round(final_val["loss"] - final_train["loss"], 5),
            "top1_next_char_acc": round(final_val["top1_next_char_acc"], 5),
            "distinct_1": round(gen_m["distinct_1"], 5),
            "distinct_2": round(gen_m["distinct_2"], 5),
            "distinct_3": round(gen_m["distinct_3"], 5),
            "repeated_4gram_rate": round(gen_m["repeated_4gram_rate"], 5),
            "grad_norm_mean": round(stab["grad_norm_mean"], 5),
            "grad_norm_max": round(stab["grad_norm_max"], 5),
            "loss_spikes": stab["loss_spikes"], "nan_count": 0,
            "param_count": n_params,
            "train_tokens_per_sec": round(tokens_seen / train_seconds, 2),
            "gen_tokens_per_sec": round(gen_tps, 2),
            "peak_memory_gb": round(peak_mem, 4),
            "total_train_seconds": round(train_seconds, 2),
            "device": str(device),
            "hardware": describe_hardware(device)["processor"],
        }
        write_metrics(paths["extended_metrics_csv_path"], row)
        write_team_row(paths["metrics_csv_path"], row, TEAM_SCHEMA,
                       overrides={"hardware": f"{describe_hardware(device)['processor']} / "
                                              f"{device} / torch {torch.__version__}"})
        log.event("metrics", **row)
        log.close(status="ok", wall=round(time.time() - t_start, 2))
        print(json.dumps(row, indent=2))
        return 0
    except Exception as exc:
        log.close(status="error", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
