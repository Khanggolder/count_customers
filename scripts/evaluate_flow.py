"""Evaluate motion event timing; passenger accuracy stays N/A without count estimates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_motion(events: list[dict], labels: list[dict] | None, tolerance_s: float = 1.0) -> dict:
    if not labels:
        return {"motion_precision": None, "motion_recall": None, "passenger_mae": None,
                "passenger_count_accuracy": "N/A: no labeled ground truth"}
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    predictions = [(row["direction"].upper(), (float(row["start_time_s"]) + float(row["end_time_s"])) / 2)
                   for row in events]
    truth = [(row["direction"].upper(), float(row["event_time_s"])) for row in labels]
    unmatched = truth.copy()
    matched = 0
    for direction, timestamp in predictions:
        choices = [(abs(timestamp - value), index) for index, (label_direction, value) in enumerate(unmatched)
                   if label_direction == direction and abs(timestamp - value) <= tolerance_s]
        if choices:
            unmatched.pop(min(choices)[1])
            matched += 1
    return {"motion_precision": matched / len(predictions) if predictions else (1.0 if not truth else 0.0),
            "motion_recall": matched / len(truth) if truth else 1.0,
            "matched_motion_events": matched, "predicted_motion_events": len(predictions),
            "labeled_events": len(truth), "passenger_mae": None,
            "passenger_count_accuracy": "N/A: uncalibrated motion events have no passenger count"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--tolerance-s", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("runs/flow_evaluation.json"))
    args = parser.parse_args()
    report = evaluate_motion(_rows(args.events), _rows(args.labels) if args.labels and args.labels.is_file()
                             else None, args.tolerance_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
