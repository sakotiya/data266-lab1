"""Metric collection into the per-task metrics_report.csv.

The CSV schema is fixed per task by the workplan's "metrics to report" list.
write_metrics() refuses to write a row that is missing a required column, so a
half-finished evaluation fails loudly instead of producing a gap in the report.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

MISSING = "NOT_MEASURED"


def read_header(csv_path: str | Path) -> list[str]:
    with Path(csv_path).open() as fh:
        return next(csv.reader(fh))


def write_metrics(csv_path: str | Path, row: dict[str, Any]) -> None:
    """Append one model's metrics. Every header column must be present in `row`."""
    csv_path = Path(csv_path)
    header = read_header(csv_path)
    missing = [c for c in header if c not in row]
    if missing:
        raise ValueError(f"metrics row is missing required columns: {missing}")
    unknown = [k for k in row if k not in header]
    if unknown:
        raise ValueError(f"metrics row has columns not in {csv_path.name}: {unknown}")
    with csv_path.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=header).writerow(row)


def write_team_row(csv_path: str | Path, row: dict, mapping: dict,
                   overrides: dict | None = None) -> None:
    """Write the same run into the TEAM-agreed schema.

    The team fixed one column set per task so every member's numbers line up
    (workplan section 0). My own table carries extra columns the team schema
    omits, so each run writes both: the team file for comparison, the extended
    file for the metrics the brief requires but the team header lacks.
    `mapping` is {team_column: my_column}; it must cover the team header exactly.
    """
    header = read_header(csv_path)
    missing = set(header) - set(mapping)
    if missing:
        raise ValueError(f"mapping does not cover team columns: {sorted(missing)}")
    extra = set(mapping) - set(header)
    if extra:
        raise ValueError(f"mapping has columns absent from {Path(csv_path).name}: {sorted(extra)}")
    overrides = overrides or {}
    out = {c: overrides.get(c, row.get(mapping[c], MISSING)) for c in header}
    with Path(csv_path).open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=header).writerow(out)


def peak_memory_gb(device: "torch.device") -> float:
    """Peak memory in GB. CUDA reports a true peak; MPS has no peak counter, so
    it reports driver-allocated memory at call time (poll it at the loop's high
    point) and CPU falls back to process max RSS. Say which in the report."""
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
