"""Count/event evaluator. No ground truth means metrics are N/A, never invented."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_by_clip(rows: list[dict]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        clip = row.get("clip") or row.get("clip_dir") or "single"
        if row.get("direction"):
            key = (clip, row["direction"].upper())
            counts[key] = counts.get(key, 0) + 1
        else:
            for field, direction in (
                ("entering_gt", "IN"),
                ("exiting_gt", "OUT"),
                ("in", "IN"),
                ("out", "OUT"),
            ):
                if row.get(field) not in (None, ""):
                    counts[(clip, direction)] = int(row[field])
    return counts


def _time(row: dict) -> float | None:
    value = row.get("timestamp_s") or row.get("source_pts")
    return float(value) if value not in (None, "") else None


def evaluate(pred: list[dict], gt: list[dict] | None, tolerance_s: float = 1.0) -> dict:
    if gt is None or not gt:
        return {
            "mae": None,
            "mae_in": None,
            "mae_out": None,
            "net_mae": None,
            "count_accuracy_in": None,
            "count_accuracy_out": None,
            "precision": None,
            "recall": None,
            "model_validation": "NO_GROUND_TRUTH_NA",
            "rows": [],
        }
    if any(row.get("delta") == "-1" and not row.get("crossing_direction") for row in pred):
        return {
            "mae": None, "mae_in": None, "mae_out": None, "net_mae": None,
            "count_accuracy_in": None, "count_accuracy_out": None,
            "precision": None, "recall": None,
            "model_validation": "LEGACY_AMBIGUOUS_DIRECTION_NA", "rows": [],
        }
    predicted, truth = count_by_clip(pred), count_by_clip(gt)
    keys = sorted(set(predicted) | set(truth))
    rows = [
        {
            "clip": clip,
            "direction": direction,
            "predicted": predicted.get((clip, direction), 0),
            "ground_truth": truth.get((clip, direction), 0),
            "absolute_error": abs(predicted.get((clip, direction), 0) - truth.get((clip, direction), 0)),
        }
        for clip, direction in keys
    ]
    if not rows:
        return {
            "mae": None,
            "precision": None,
            "recall": None,
            "model_validation": "NO_COMMON_LABELS_NA",
            "rows": [],
        }
    predicted_events = [row for row in pred if row.get("direction")]
    truth_events = [row for row in gt if row.get("direction")]
    timed = bool(truth_events) and all(_time(row) is not None for row in predicted_events + truth_events)
    precision = recall = None
    matched = 0
    if timed:
        for key in keys:
            remaining = sorted(
                [
                    (row, _time(row))
                    for row in truth_events
                    if ((row.get("clip") or row.get("clip_dir") or "single"), row.get("direction", "").upper())
                    == key
                ],
                key=lambda item: item[1],
            )
            for prediction, prediction_time in sorted(
                [
                    (row, _time(row))
                    for row in predicted_events
                    if ((row.get("clip") or row.get("clip_dir") or "single"), row.get("direction", "").upper())
                    == key
                ],
                key=lambda item: item[1],
            ):
                choices = [
                    (abs(value - prediction_time), index)
                    for index, (label, value) in enumerate(remaining)
                    if abs(value - prediction_time) <= tolerance_s
                    and (not prediction.get("camera") or not label.get("camera") or prediction["camera"].upper() == label["camera"].upper())
                    and (not prediction.get("crossing_direction") or not label.get("crossing_direction") or prediction["crossing_direction"] == label["crossing_direction"])
                ]
                if choices:
                    _, index = min(choices)
                    remaining.pop(index)
                    matched += 1
        precision = matched / len(predicted_events) if predicted_events else (1.0 if not truth_events else 0.0)
        recall = matched / len(truth_events) if truth_events else 1.0
    return {
        "mae": sum(row["absolute_error"] for row in rows) / len(rows),
        "mae_in": _direction_mae(rows, "IN"),
        "mae_out": _direction_mae(rows, "OUT"),
        "net_mae": _net_mae(predicted, truth),
        "count_accuracy_in": _exact_rate(rows, "IN"),
        "count_accuracy_out": _exact_rate(rows, "OUT"),
        "precision": precision,
        "recall": recall,
        "matched_events": matched if timed else None,
        "tolerance_s": tolerance_s if timed else None,
        "model_validation": "TIMED_EVENT_MATCH" if timed else "COUNT_ONLY",
        "rows": rows,
    }


def _direction_mae(rows: list[dict], direction: str) -> float | None:
    values = [row["absolute_error"] for row in rows if row["direction"] == direction]
    return sum(values) / len(values) if values else None


def _exact_rate(rows: list[dict], direction: str) -> float | None:
    values = [row["absolute_error"] for row in rows if row["direction"] == direction]
    return sum(error == 0 for error in values) / len(values) if values else None


def _net_mae(predicted: dict[tuple[str, str], int], truth: dict[tuple[str, str], int]) -> float:
    clips = {clip for clip, _ in predicted | truth}
    return sum(
        abs(
            predicted.get((clip, "IN"), 0) - predicted.get((clip, "OUT"), 0)
            - truth.get((clip, "IN"), 0) + truth.get((clip, "OUT"), 0)
        )
        for clip in clips
    ) / len(clips)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--tolerance-s", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    if args.tolerance_s < 0:
        parser.error("--tolerance-s must be non-negative")
    report = evaluate(
        read_rows(args.pred), read_rows(args.gt) if args.gt and args.gt.is_file() else None, args.tolerance_s
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
