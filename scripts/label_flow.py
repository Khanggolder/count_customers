"""Manual REAR passage labels. One key press marks an event, then enter group size."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def label(video: Path, output: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    rows = []
    index = -1
    paused = False
    frame = None
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break
                index += 1
            display = frame.copy()
            cv2.putText(display, "O=OUT I=IN SPACE=pause Q=save/quit", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            cv2.imshow("REAR manual passage labels", display)
            key = cv2.waitKey(0 if paused else max(1, round(1000 / fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key in (ord("o"), ord("i")):
                direction = "OUT" if key == ord("o") else "IN"
                count = input(f"{direction} at {index / fps:.2f}s: passenger_count (blank if uncertain): ").strip()
                if count and (not count.isdecimal() or int(count) < 1):
                    print("Invalid count; annotation skipped")
                    continue
                rows.append({"video_id": video.name, "event_time_s": round(index / fps, 4),
                             "direction": direction, "passenger_count": count,
                             "notes": input("notes (optional): ").strip()})
    finally:
        capture.release()
        cv2.destroyAllWindows()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, ["video_id", "event_time_s", "direction", "passenger_count", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Saved {label(args.video, args.output)} manual events to {args.output}")
