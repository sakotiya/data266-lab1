"""Raw, append-only run logging.

The workplan grades an unedited log file per training run as the evidence trail,
so this writer never rewrites or prunes: one JSONL line per step/epoch event plus
a human-readable .log mirror. Do not clean these files up after a run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id(tag: str) -> str:
    return f"{tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class RunLogger:
    """Usage:
        log = RunLogger(log_dir, run_id, config=cfg, hardware=hw)
        log.event("step", step=1, loss=4.21, grad_norm=1.03)
        log.close(status="ok")
    """

    def __init__(self, log_dir: str | Path, run_id: str,
                 config: dict[str, Any] | None = None,
                 hardware: dict[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = (self.dir / f"{run_id}.jsonl").open("a", buffering=1)
        self.text = (self.dir / f"{run_id}.log").open("a", buffering=1)
        self.t0 = time.time()
        self.event(
            "run_start",
            run_id=run_id,
            utc=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_commit(),
            python=sys.version.split()[0],
            argv=" ".join(sys.argv),
            config=config or {},
            hardware=hardware or {},
        )

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"kind": kind, "t": round(time.time() - self.t0, 3), **fields}
        self.jsonl.write(json.dumps(rec, default=str) + "\n")
        flat = " ".join(f"{k}={v}" for k, v in fields.items() if k != "config")
        self.text.write(f"[{rec['t']:>9.3f}s] {kind:<12} {flat}\n")

    def close(self, status: str = "ok", **fields: Any) -> None:
        self.event("run_end", status=status, wall_seconds=round(time.time() - self.t0, 3), **fields)
        self.jsonl.close()
        self.text.close()
