from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .business import expected_direction
from .types import CrossingDirection, Direction


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
class AppConfig:
    database_url: str
    detector: DetectorConfig
    cameras: dict[str, CameraConfig]


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
        if Direction(value["event_direction"]) != expected_direction(name, CrossingDirection.A_TO_B):
            raise ValueError(f"Camera {name} A_TO_B direction conflicts with calibration")
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
    return AppConfig(
        database_url=raw.get("database_url", "sqlite:///data/busstop_flow.db"),
        detector=DetectorConfig(**raw.get("detector", {})),
        cameras=cameras,
    )
