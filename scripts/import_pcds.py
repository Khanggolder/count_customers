from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
import zipfile
from pathlib import Path

COUNT_PATTERNS = {
    "entering": re.compile(r"(?:entering|enter|in)\s*[:=]?\s*(\d+)", re.IGNORECASE),
    "exiting": re.compile(r"(?:exiting|exit|out)\s*[:=]?\s*(\d+)", re.IGNORECASE),
}


def parse_label(path: Path) -> tuple[int | None, int | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, int | None] = {}
    for key, pattern in COUNT_PATTERNS.items():
        match = pattern.search(text)
        values[key] = int(match.group(1)) if match else None
    if values["entering"] is None and values["exiting"] is None:
        numbers = [int(item) for item in re.findall(r"\d+", text)]
        if len(numbers) >= 2:
            values["entering"], values["exiting"] = numbers[-2:]
    return values["entering"], values["exiting"]


def parse_pcds_rows(path: Path) -> list[dict[str, object]]:
    """Parse documented PCDS rows: DepthVideoName, Entering, Exiting, VideoType."""
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        name = parts[0]
        if not name or name.replace(".", "", 1).lstrip("-").isdigit():
            continue
        rows.append(
            {
                "clip": name,
                "entering_gt": int(parts[1]),
                "exiting_gt": int(parts[2]),
                "video_type": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
            }
        )
    return rows


def scan(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in root.rglob("label.txt"):
        folder = label.parent
        lowered = folder.name.lower()
        camera = "front" if "front" in lowered else ("back" if "back" in lowered or "rear" in lowered else "unknown")
        videos = sorted(path for path in folder.rglob("*") if path.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov"})
        detailed = parse_pcds_rows(label)
        if detailed:
            by_name = {path.name.lower(): path for path in videos}
            for item in detailed:
                named = by_name.get(str(item["clip"]).lower())
                rows.append(
                    {
                        "clip": item["clip"],
                        "clip_dir": str(folder),
                        "camera": camera,
                        "entering_gt": item["entering_gt"],
                        "exiting_gt": item["exiting_gt"],
                        "video_type": item["video_type"],
                        "label_file": str(label),
                        "video_files": str(named) if named else "",
                    }
                )
        else:
            entering, exiting = parse_label(label)
            rows.append({"clip": folder.name, "clip_dir": str(folder), "camera": camera, "entering_gt": entering, "exiting_gt": exiting, "video_type": None, "label_file": str(label), "video_files": "|".join(str(path) for path in videos)})
    return rows


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    temp: tempfile.TemporaryDirectory[str] | None = None
    root = args.input
    if args.input.is_file():
        if not zipfile.is_zipfile(args.input):
            raise ValueError("Input file phải là ZIP hợp lệ")
        temp = tempfile.TemporaryDirectory(prefix="busreid_pcds_")
        root = Path(temp.name)
        with zipfile.ZipFile(args.input) as archive:
            for item in archive.infolist():
                target = (root / item.filename).resolve()
                if root.resolve() not in target.parents and target != root.resolve():
                    raise ValueError("ZIP chứa đường dẫn không an toàn")
            archive.extractall(root)
    try:
        rows = scan(root)
        manifest = args.output / "pcds_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["clip", "clip_dir", "camera", "entering_gt", "exiting_gt", "video_type", "label_file", "video_files"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Đã lập manifest {len(rows)} clip: {manifest}")
        return 0 if rows else 3
    finally:
        if temp:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

