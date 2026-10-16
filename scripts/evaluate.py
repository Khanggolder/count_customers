from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_counts(path: Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            clip = row.get("clip") or row.get("clip_dir") or "single"
            if "direction" in row:
                key = (clip, row["direction"].upper())
                counts[key] = counts.get(key, 0) + 1
            else:
                for field, direction in (("entering_gt", "IN"), ("exiting_gt", "OUT"), ("in", "IN"), ("out", "OUT")):
                    if row.get(field) not in (None, ""):
                        counts[(clip, direction)] = int(row[field])
    return counts


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    predicted, ground_truth = load_counts(args.pred), load_counts(args.gt)
    common = sorted(set(predicted) & set(ground_truth))
    if not common:
        raise ValueError("Không có clip/direction chung giữa prediction và ground truth")
    rows = [{"clip": key[0], "direction": key[1], "predicted": predicted[key], "ground_truth": ground_truth[key], "absolute_error": abs(predicted[key] - ground_truth[key])} for key in common]
    report = {"count_pairs": len(rows), "mae": sum(row["absolute_error"] for row in rows) / len(rows), "rows": rows, "reid_metrics": None, "reid_note": "NA nếu không có identity ground truth"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

