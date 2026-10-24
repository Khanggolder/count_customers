from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import cv2


def save_upload(upload: BinaryIO, filename: str, destination: Path) -> Path:
    if Path(filename).suffix.lower() != ".mp4":
        raise ValueError("Chỉ hỗ trợ upload MP4")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{uuid.uuid4().hex}.mp4"
    with path.open("wb") as output:
        while block := upload.read(1024 * 1024):
            output.write(block)
    if path.stat().st_size == 0:
        path.unlink()
        raise ValueError("Video upload rỗng")
    return path


def resolve_source(
    upload: BinaryIO | None,
    filename: str | None,
    path: str,
    rtsp: str,
    destination: Path,
    webcam_index: int | None = None,
) -> str | None:
    selected = sum(bool(item) for item in (upload, path.strip(), rtsp.strip())) + (webcam_index is not None)
    if selected > 1:
        raise ValueError("Mỗi camera chỉ chọn một nguồn: upload, đường dẫn, webcam hoặc RTSP")
    if webcam_index is not None:
        if webcam_index < 0:
            raise ValueError("Webcam index phải không âm")
        return f"webcam:{webcam_index}"
    if upload is not None:
        return str(save_upload(upload, filename or "", destination))
    if rtsp.strip():
        if not rtsp.strip().lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("Nguồn RTSP phải bắt đầu bằng rtsp:// hoặc rtsps://")
        return rtsp.strip()
    if path.strip():
        selected_path = Path(path.strip()).expanduser()
        if not selected_path.is_file() or selected_path.suffix.lower() != ".mp4":
            raise ValueError(f"Không tìm thấy MP4: {selected_path}")
        return str(selected_path.resolve())
    return None


def first_frame(path: str) -> bytes | None:
    if path.lower().startswith(("rtsp://", "rtsps://")):
        return None
    capture = cv2.VideoCapture(path)
    try:
        ok, frame = capture.read()
        if not ok:
            return None
        height, width = frame.shape[:2]
        if width > 640:
            frame = cv2.resize(frame, (640, round(height * 640 / width)))
        encoded, image = cv2.imencode(".jpg", frame)
        return image.tobytes() if encoded else None
    finally:
        capture.release()


def new_job_dir(root: Path, kind: str = "video_lab") -> Path:
    if kind not in {"video_lab", "ai_counting"}:
        raise ValueError("Loại phiên không hợp lệ")
    output = root / "runs" / kind / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True)
    return output


def start_job(
    root: Path,
    output: Path,
    sources: dict[str, str],
    *,
    trip_id: str | None = None,
    telemetry_file: str = "",
    capture_start: str = "",
    synchronized: bool = False,
    max_frames: int = 0,
    hidden_layers: set[str] | None = None,
    session_only: bool = False,
    start_frame: int = 0,
) -> subprocess.Popen:
    if not sources:
        raise ValueError("Chọn ít nhất một video FRONT hoặc REAR")
    if session_only and (trip_id or telemetry_file or capture_start or synchronized):
        raise ValueError("Phiên đếm độc lập không nhận Trip hoặc telemetry")
    if start_frame and not session_only:
        raise ValueError("Chọn frame bắt đầu chỉ hỗ trợ Test ROI độc lập")
    output.mkdir(parents=True, exist_ok=True)
    config_snapshot = output / "cameras_snapshot.yaml"
    shutil.copy2(root / "configs" / "cameras.yaml", config_snapshot)
    command = [
        sys.executable,
        str(root / "scripts" / ("run_count_session.py" if session_only else "run_count.py")),
        "--config",
        str(config_snapshot),
        "--output",
        str(output),
        "--stop-file" if session_only else "--cancel-file",
        str(output / ("stop.request" if session_only else "cancel.request")),
    ]
    for camera, source in sources.items():
        command.extend([f"--{camera}", source])
    if trip_id:
        command.extend(["--trip-id", trip_id])
    if telemetry_file:
        command.extend(["--telemetry-json", telemetry_file])
    if capture_start:
        command.extend(["--recording-start-time", capture_start])
    if synchronized:
        command.append("--telemetry-synchronized")
    if max_frames:
        command.extend(["--max-frames", str(max_frames)])
    if start_frame:
        command.extend(["--start-frame", str(start_frame)])
    for layer in sorted(hidden_layers or set()):
        command.append(f"--hide-{layer}")
    log = (output / "process.log").open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        log.close()


def camera_status(output: Path, camera: str, returncode: int | None) -> dict:
    progress = output / f"{camera}_progress.json"
    if progress.is_file():
        try:
            return json.loads(progress.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "status": "QUEUED" if returncode is None else "CANCELLED" if returncode == 2 else "FAILED",
        "frames": 0,
        "events": 0,
        "counts": {},
        "total_frames": None,
        "elapsed_s": 0.0,
        "processing_fps": 0.0,
    }


def event_timeline(output: Path, cameras: list[str]) -> list[dict]:
    rows = []
    for camera in cameras:
        path = output / f"{camera}_events.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    rows.append(
                        {
                            "timestamp": row["capture_time_utc"] or f"{float(row['timestamp_s']):.2f}s video",
                            "camera": camera.upper(),
                            "track_id": row["track_id"],
                            "direction": row["direction"],
                            "frame_index": int(row["frame_index"]),
                            "confidence": row["confidence"],
                            "event_status": "UNASSIGNED",
                            "timestamp_s": float(row["timestamp_s"]),
                            "crop_path": row["crop_path"],
                            "created_at": row.get("created_at", ""),
                        }
                    )
    return sorted(rows, key=lambda row: (row["timestamp_s"], row["camera"]))
