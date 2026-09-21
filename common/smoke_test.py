"""Harness smoke test - no torch, no training, no data.

Proves the three things every run depends on: configs load and inherit correctly,
the run logger writes an append-only trail, and the metrics writer enforces the
schema. Run from the repo root: `python common/smoke_test.py`
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import REPO_ROOT, load_config  # noqa: E402
from common.metrics_io import write_metrics  # noqa: E402
from common.run_logger import RunLogger, new_run_id  # noqa: E402

CONFIGS = [
    "task1_llm/zoheb_waghu/configs/gpt_baseline.yaml",
    "task2_sentiment/zoheb_waghu/configs/m1_baseline_bilstm.yaml",
    "task2_sentiment/zoheb_waghu/configs/m2_cnn_multikernel.yaml",
    "task2_sentiment/zoheb_waghu/configs/m3_bilstm_attention.yaml",
    "task3_gan/zoheb_waghu/configs/cyclegan_baseline.yaml",
]


def check_configs() -> None:
    for rel in CONFIGS:
        cfg = load_config(REPO_ROOT / rel)
        assert cfg["run"]["member"] == "zoheb_waghu", rel
        for key in ("log_dir", "checkpoint_dir", "output_dir"):
            assert Path(cfg["paths"][key]).is_dir(), f"{rel}: {key} missing"
        assert Path(cfg["paths"]["metrics_csv_path"]).is_file(), rel
        print(f"  ok  {rel}  (tag={cfg['run']['tag']})")

    shared = load_config(REPO_ROOT / "task2_sentiment/zoheb_waghu/configs/_shared.yaml")
    m3 = load_config(REPO_ROOT / "task2_sentiment/zoheb_waghu/configs/m3_bilstm_attention.yaml")
    assert m3["embedding"]["dim"] == 256 and shared["embedding"]["dim"] == 128, "override failed"
    assert m3["preprocess"] == shared["preprocess"], "inheritance dropped shared preprocessing"
    print("  ok  task2 `extends:` inheritance (child overrides, parent preserved)")


def check_logger() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_id = new_run_id("smoke")
        log = RunLogger(tmp, run_id, config={"a": 1}, hardware={"device": "cpu"})
        log.event("step", step=1, loss=4.2, grad_norm=1.1)
        log.close(status="ok")
        lines = (Path(tmp) / f"{run_id}.jsonl").read_text().strip().split("\n")
        assert [__import__("json").loads(x)["kind"] for x in lines] == \
            ["run_start", "step", "run_end"], lines
        assert (Path(tmp) / f"{run_id}.log").read_text().count("\n") == 3
    print("  ok  run logger (jsonl + text mirror, 3 events)")


def check_metrics_writer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "m.csv"
        csv_path.write_text("run_id,accuracy,f1_macro\n")
        write_metrics(csv_path, {"run_id": "r1", "accuracy": 0.9, "f1_macro": 0.88})
        for bad, why in [({"run_id": "r2"}, "missing column"),
                         ({"run_id": "r3", "accuracy": 1, "f1_macro": 1, "oops": 1}, "unknown column")]:
            try:
                write_metrics(csv_path, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"writer accepted a row with a {why}")
        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1 and rows[0]["run_id"] == "r1", rows
    print("  ok  metrics writer (accepts complete rows, rejects incomplete and unknown)")


def check_schemas() -> None:
    # Team-agreed schemas (these must match every member's files exactly) plus my
    # own extended tables, which carry the columns the team schema omits.
    expected = {"task1_llm/zoheb_waghu/metrics_report.csv": 22,
                "task2_sentiment/zoheb_waghu/metrics_report.csv": 31,
                "task3_gan/zoheb_waghu/full_metrics_report.csv": 30,
                "task1_llm/zoheb_waghu/metrics_report_extended.csv": 25,
                "task2_sentiment/zoheb_waghu/metrics_report_extended.csv": 46}
    for rel, n in expected.items():
        header = next(csv.reader((REPO_ROOT / rel).open()))
        assert len(header) == n, f"{rel}: {len(header)} columns, expected {n}"
        assert len(set(header)) == n, f"{rel}: duplicate column names"
        print(f"  ok  {rel}  ({n} metric columns)")


if __name__ == "__main__":
    for name, fn in [("configs", check_configs), ("logger", check_logger),
                     ("metrics writer", check_metrics_writer), ("schemas", check_schemas)]:
        print(f"{name}:")
        fn()
    print("\nall harness checks passed")
