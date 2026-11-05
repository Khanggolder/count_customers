from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .business import validate_passenger_direction
from .types import Direction


@dataclass(slots=True)
class CameraConfig:
    name: str
    role: str
    event_direction: Direction
    source: str | int | None
    zone_a: list[tuple[float, float]]
    zone_b: list[tuple[float, float]]
    min_stable_frames: int = 2
    debounce_frames: int = 12
    state_ttl_frames: int = 90
    anchor_mode: str = "bottom_center"
    time_offset_ms: int = 0


@dataclass(slots=True)
class DetectorConfig:
    model: str = "yolo11n.pt"
    confidence: float = 0.35
    iou: float = 0.5
    device: str = "auto"
    tracker: str = "bytetrack.yaml"


@dataclass(slots=True)
class RearOccupancyConfig:
    model: str = "11n-head.pt"
    confidence: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"
    roi: list[tuple[float, float]] = field(
        default_factory=lambda: [(0, 0), (1, 0), (1, 1), (0, 1)]
    )
    ignore: list[list[tuple[float, float]]] = field(default_factory=list)
    smoothing_window: int = 1


@dataclass(slots=True)
class AppConfig:
    database_url: str
    detector: DetectorConfig
    cameras: dict[str, CameraConfig]
    rear_occupancy: RearOccupancyConfig = field(default_factory=RearOccupancyConfig)


def _polygon(value: list[list[float]]) -> list[tuple[float, float]]:
    if len(value) < 3:
        raise ValueError("Mỗi vùng ROI cần ít nhất 3 điểm")
    points = [(float(x), float(y)) for x, y in value]
    if any(x < 0 or x > 1 or y < 0 or y > 1 for x, y in points):
        raise ValueError("Tọa độ ROI phải normalized trong [0, 1]")
    return points


def load_config(path: str | Path = "configs/cameras.yaml") -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cameras: dict[str, CameraConfig] = {}
    for name, value in raw["cameras"].items():
        if value["role"].upper() != name.upper():
            raise ValueError(f"Camera {name} role must match its FRONT/REAR identity")
        validate_passenger_direction(name, value["event_direction"])
        cameras[name] = CameraConfig(
            name=name,
            role=value["role"],
            event_direction=Direction(value["event_direction"]),
            source=value.get("source"),
            zone_a=_polygon(value["zone_a"]),
            zone_b=_polygon(value["zone_b"]),
            min_stable_frames=int(value.get("min_stable_frames", 2)),
            debounce_frames=int(value.get("debounce_frames", 12)),
            state_ttl_frames=int(value.get("state_ttl_frames", 90)),
            anchor_mode=value.get("anchor_mode", "bottom_center"),
            time_offset_ms=int(value.get("time_offset_ms", 0)),
        )
    occupancy_raw = raw.get("rear_occupancy", {})
    occupancy = RearOccupancyConfig(
        model=str(occupancy_raw.get("model", "11n-head.pt")),
        confidence=float(occupancy_raw.get("confidence", 0.35)),
        iou=float(occupancy_raw.get("iou", 0.5)),
        imgsz=int(occupancy_raw.get("imgsz", 640)),
        device=str(occupancy_raw.get("device", "auto")),
        roi=_polygon(occupancy_raw.get("roi", [[0, 0], [1, 0], [1, 1], [0, 1]])),
        ignore=[_polygon(points) for points in occupancy_raw.get("ignore", [])],
        smoothing_window=int(occupancy_raw.get("smoothing_window", 1)),
    )
    from .roi import validate_polygon

    validate_polygon(occupancy.roi)
    for polygon in occupancy.ignore:
        validate_polygon(polygon)
    if not 0 < occupancy.confidence <= 1 or not 0 < occupancy.iou <= 1:
        raise ValueError("Head Detection confidence/IoU must be in (0, 1]")
    if occupancy.imgsz < 32 or occupancy.smoothing_window < 1:
        raise ValueError("Head Detection imgsz/window must be positive")
    return AppConfig(
        database_url=raw.get("database_url", "sqlite:///data/busstop_flow.db"),
        detector=DetectorConfig(**raw.get("detector", {})),
        cameras=cameras,
        rear_occupancy=occupancy,
    )
