"""build_timing_table — tabulate bench_device.py timing records as the Table 3 markdown.

Reads the JSON records written by ``scripts/exps/bench_device.py --json-out`` (one file per
particle count S, as ``bash scripts/run_experiments.sh --timing`` produces them) and writes a
markdown table in the format of assets/runs/analysis/ablations/timing/h100_timing.md, whose
rows the evidence register parses: ``| S | run time (s) | steps/s | peak GPU MB | s/puzzle |``
with run time to 2 decimals, steps/s to 1, peak memory as an integer ("-" when the device does
not report it) and seconds per puzzle to 3.

Usage
-----
    python scripts/analysis/build_timing_table.py --device cuda --device-label H100 \
        "assets/runs/analysis/ablations/timing/bench_sudoku_cuda_S*.json" \
        --out assets/runs/analysis/ablations/timing/h100_timing.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from tqdm import tqdm

PROTOCOL_KEYS = ("steps", "batch_size", "n_batches", "n_puzzles")


def load_records(patterns: list[str], device: str | None) -> list[dict]:
    """Load the records of every matching JSON file, keeping one device."""
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)} | {p for p in patterns if Path(p).exists()})
    if not paths:
        raise FileNotFoundError(f"no timing records match {patterns}")
    records: list[dict] = []
    for path in tqdm(paths, desc="Reading timing records"):
        payload = json.loads(Path(path).read_text())
        records.extend(payload if isinstance(payload, list) else [payload])
    devices = sorted({r["device"] for r in records})
    if device is None:
        if len(devices) != 1:
            raise ValueError(f"records cover devices {devices}; pass --device to pick one")
        device = devices[0]
    records = [r for r in records if r["device"] == device]
    if not records:
        raise ValueError(f"no records for device {device!r} (found {devices})")
    by_s: dict[int, dict] = {}
    for r in records:
        if r["particles"] in by_s:
            raise ValueError(f"two records for S={r['particles']} on {device}; remove the stale one")
        by_s[r["particles"]] = r
    for key in PROTOCOL_KEYS:
        if len({r[key] for r in records}) != 1:
            raise ValueError(f"records disagree on {key}: {sorted({r[key] for r in records})}")
    return [by_s[s] for s in sorted(by_s)]


def render_table(records: list[dict], device_label: str) -> str:
    """Render the records as the h100_timing.md markdown."""
    first = records[0]
    lines = [
        f"# {device_label} timing and memory against particle count S (paper Table 3)",
        "",
        f"`bench_device.py` on {device_label}, N={first['steps']} steps, "
        f"batch {first['batch_size']} × {first['n_batches']} batches = {first['n_puzzles']} puzzles per run.",
        "",
        "| S | run time (s) | steps/s | peak GPU MB | s/puzzle |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        peak = f"{r['peak_mb']:.0f}" if r["peak_mb"] is not None else "-"
        lines.append(
            f"| {r['particles']} | {r['elapsed_s']:.2f} | {r['steps_per_s']:.1f} | {peak} | {r['s_per_puzzle']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tabulate bench_device.py --json-out records as the Table 3 markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("records", nargs="+", help="JSON files or glob patterns written by bench_device.py --json-out.")
    ap.add_argument("--device", default=None, help="Device to tabulate (default: the only device in the records).")
    ap.add_argument(
        "--device-label", default=None, help="Device name for the header (default: the device string in upper case)."
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = load_records(args.records, args.device)
    label = args.device_label or records[0]["device"].upper()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_table(records, label))
    print(f"Wrote {args.out} ({len(records)} rows, {label})")


if __name__ == "__main__":
    main()
