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
class RearHeadOutConfig:
    """Independent experimental head crossing zones; never writes business events."""

    model: str = "11n-head.pt"
    tracker: str = "bytetrack.yaml"
    confidence: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"
    roi: list[tuple[float, float]] = field(default_factory=lambda: [(0, 0), (1, 0), (1, 1), (0, 1)])
    zone_a: list[tuple[float, float]] = field(default_factory=list)
    zone_b: list[tuple[float, float]] = field(default_factory=list)
    min_stable_frames: int = 2
    state_ttl_frames: int = 90
    id_switch_grace_frames: int = 5
    id_switch_max_distance: float = 0.08


@dataclass(slots=True)
class RearFlowOutConfig:
    """Experimental motion events, never authoritative passenger counts."""

    enabled: bool = False
    roi: list[tuple[float, float]] = field(default_factory=lambda: [(0, 0), (1, 0), (1, 1), (0, 1)])
    counting_line: list[tuple[float, float]] = field(default_factory=lambda: [(0.2, 0.5), (0.8, 0.5)])
    out_direction: tuple[float, float] = (0.0, 1.0)
    flow_method: str = "farneback"
    stabilization: bool = False
    clahe: bool = False
    gaussian_blur: bool = True
    min_motion_magnitude: float = 1.0
    smoothing_window: int = 5
    min_active_fraction: float = 0.08
    min_event_frames: int = 2
    max_gap_frames: int = 3
    debounce_frames: int = 6


@dataclass(slots=True)
class AppConfig:
    database_url: str
    detector: DetectorConfig
    cameras: dict[str, CameraConfig]
    rear_occupancy: RearOccupancyConfig = field(default_factory=RearOccupancyConfig)
    rear_head_out: RearHeadOutConfig = field(default_factory=RearHeadOutConfig)
    rear_flow_out: RearFlowOutConfig = field(default_factory=RearFlowOutConfig)


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
    head_raw = raw.get("rear_head_out", {})
    head = RearHeadOutConfig(
        model=str(head_raw.get("model", occupancy.model)),
        tracker=str(head_raw.get("tracker", "bytetrack.yaml")),
        confidence=float(head_raw.get("confidence", occupancy.confidence)),
        iou=float(head_raw.get("iou", occupancy.iou)),
        imgsz=int(head_raw.get("imgsz", occupancy.imgsz)),
        device=str(head_raw.get("device", occupancy.device)),
        roi=_polygon(head_raw.get("roi", [[0, 0], [1, 0], [1, 1], [0, 1]])),
        zone_a=_polygon(head_raw.get("zone_a", cameras["rear"].zone_a)),
        zone_b=_polygon(head_raw.get("zone_b", cameras["rear"].zone_b)),
        min_stable_frames=int(head_raw.get("min_stable_frames", 2)),
        state_ttl_frames=int(head_raw.get("state_ttl_frames", 90)),
        id_switch_grace_frames=int(head_raw.get("id_switch_grace_frames", 5)),
        id_switch_max_distance=float(head_raw.get("id_switch_max_distance", 0.08)),
    )
    from .roi import validate_flow_geometry, validate_head_out_zones, validate_polygon

    validate_polygon(occupancy.roi)
    for polygon in occupancy.ignore:
        validate_polygon(polygon)
    if not 0 < occupancy.confidence <= 1 or not 0 < occupancy.iou <= 1:
        raise ValueError("Head Detection confidence/IoU must be in (0, 1]")
    if occupancy.imgsz < 32 or occupancy.smoothing_window < 1:
        raise ValueError("Head Detection imgsz/window must be positive")
    validate_head_out_zones(head.roi, head.zone_a, head.zone_b)
    if not 0 < head.confidence <= 1 or not 0 < head.iou <= 1 or head.imgsz < 32:
        raise ValueError("Head OUT confidence/IoU/imgsz invalid")
    if min(head.min_stable_frames, head.state_ttl_frames, head.id_switch_grace_frames) < 1:
        raise ValueError("Head OUT frame limits must be positive")
    if not 0 < head.id_switch_max_distance <= 1:
        raise ValueError("Head OUT ID switch distance must be in (0, 1]")
    if head.tracker not in {"bytetrack.yaml", "botsort.yaml"}:
        raise ValueError("Head OUT tracker must be bytetrack.yaml or botsort.yaml")
    flow_raw = raw.get("rear_flow_out", {})
    flow = RearFlowOutConfig(
        enabled=bool(flow_raw.get("enabled", False)),
        roi=_polygon(flow_raw.get("roi", [[0, 0], [1, 0], [1, 1], [0, 1]])),
        counting_line=[tuple(map(float, point)) for point in flow_raw.get("counting_line", [[.2, .5], [.8, .5]])],
        out_direction=tuple(map(float, flow_raw.get("out_direction", [0, 1]))),
        flow_method=str(flow_raw.get("flow_method", "farneback")),
        stabilization=bool(flow_raw.get("stabilization", False)),
        clahe=bool(flow_raw.get("clahe", False)),
        gaussian_blur=bool(flow_raw.get("gaussian_blur", True)),
        min_motion_magnitude=float(flow_raw.get("min_motion_magnitude", 1.0)),
        smoothing_window=int(flow_raw.get("smoothing_window", 5)),
        min_active_fraction=float(flow_raw.get("min_active_fraction", .08)),
        min_event_frames=int(flow_raw.get("min_event_frames", 2)),
        max_gap_frames=int(flow_raw.get("max_gap_frames", 3)),
        debounce_frames=int(flow_raw.get("debounce_frames", 6)),
    )
    validate_flow_geometry(flow.roi, flow.counting_line, flow.out_direction)
    if flow.flow_method != "farneback" or flow.min_motion_magnitude <= 0:
        raise ValueError("REAR flow method or motion threshold invalid")
    if not 0 < flow.min_active_fraction <= 1 or min(flow.smoothing_window, flow.min_event_frames,
                                                    flow.max_gap_frames, flow.debounce_frames) < 1:
        raise ValueError("REAR flow temporal limits invalid")
    return AppConfig(
        database_url=raw.get("database_url", "sqlite:///data/busstop_flow.db"),
        detector=DetectorConfig(**raw.get("detector", {})),
        cameras=cameras,
        rear_occupancy=occupancy,
        rear_head_out=head,
        rear_flow_out=flow,
    )
