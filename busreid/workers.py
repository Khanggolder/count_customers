from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event, Lock

from .config import AppConfig
from .pipeline import CameraPipeline, OverlayOptions, PipelineCancelled, PipelineResult


@dataclass(slots=True)
class WorkerStatus:
    state: str = "QUEUED"
    detail: str | None = None


@dataclass(slots=True)
class DualCameraResult:
    results: dict[str, PipelineResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    cancelled: bool = False


class DualCameraRunner:
    """Run available camera sources concurrently; one failure never blocks the other."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cancel_event = Event()
        self.status = {name: WorkerStatus() for name in ("front", "rear")}
        self._lock = Lock()

    def cancel(self) -> None:
        self.cancel_event.set()

    def _run_one(
        self,
        camera: str,
        source: str,
        output_dir: Path,
        max_frames: int | None,
        recording_start_time: datetime | None,
        overlay: OverlayOptions | None,
        cancel_file: Path | None,
        stop_file: Path | None,
        start_frame: int,
    ) -> PipelineResult:
        with self._lock:
            self.status[camera].state = "RUNNING"
        if self.cancel_event.is_set():
            raise RuntimeError("cancelled")
        options = {"max_frames": max_frames}
        if start_frame:
            options["start_frame"] = start_frame
        if recording_start_time is not None:
            options["recording_start_time"] = recording_start_time
        if overlay is not None:
            options["overlay"] = overlay
        if cancel_file is not None:
            options["cancel_file"] = cancel_file
        if stop_file is not None:
            options["stop_file"] = stop_file
        result = CameraPipeline(self.config.cameras[camera], self.config).run(source, output_dir, **options)
        with self._lock:
            self.status[camera].state = result.status if hasattr(result, "status") else "COMPLETED"
        return result

    def run(
        self,
        sources: dict[str, str],
        output_dir: str | Path,
        max_frames: int | None = None,
        recording_start_time: datetime | None = None,
        overlay: OverlayOptions | None = None,
        cancel_file: Path | None = None,
        stop_file: Path | None = None,
        start_frame: int = 0,
    ) -> DualCameraResult:
        valid = {
            camera: source for camera, source in sources.items() if source and camera in self.config.cameras
        }
        outcome = DualCameraResult()
        if not valid:
            return outcome
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=len(valid), thread_name_prefix="bus-camera") as pool:
            futures: dict[Future, str] = {
                pool.submit(
                    self._run_one,
                    camera,
                    source,
                    destination,
                    max_frames,
                    recording_start_time,
                    overlay,
                    cancel_file,
                    stop_file,
                    start_frame,
                ): camera
                for camera, source in valid.items()
            }
            for future in as_completed(futures):
                camera = futures[future]
                try:
                    outcome.results[camera] = future.result()
                except PipelineCancelled:
                    outcome.cancelled = True
                    with self._lock:
                        self.status[camera] = WorkerStatus("CANCELLED")
                except Exception as exc:  # isolate worker failures and report them
                    outcome.errors[camera] = f"{type(exc).__name__}: {exc}"
                    with self._lock:
                        self.status[camera] = WorkerStatus("FAILED", outcome.errors[camera])
        outcome.cancelled = outcome.cancelled or self.cancel_event.is_set()
        return outcome
