"""Geometry and persistence for camera ROIs (coordinates are normalized)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .config import load_config

Point = tuple[float, float]


def to_normalized(point: tuple[int, int], width: int, height: int) -> Point:
    if width < 1 or height < 1 or not (0 <= point[0] < width and 0 <= point[1] < height):
        raise ValueError("Điểm nằm ngoài frame")
    return (point[0] / width, point[1] / height)


def to_pixels(point: Point, width: int, height: int) -> tuple[int, int]:
    return (min(round(point[0] * width), width - 1), min(round(point[1] * height), height - 1))


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return abs(_orientation(a, b, p)) < 1e-10 and all(
        min(start, end) - 1e-10 <= value <= max(start, end) + 1e-10
        for start, end, value in zip(a, b, p, strict=False)
    )


def _intersects(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    if (ab_c > 1e-10 and ab_d < -1e-10 or ab_c < -1e-10 and ab_d > 1e-10) and (
        cd_a > 1e-10 and cd_b < -1e-10 or cd_a < -1e-10 and cd_b > 1e-10
    ):
        return True
    return any(
        _on_segment(start, end, point) for start, end, point in ((a, b, c), (a, b, d), (c, d, a), (c, d, b))
    )


def _inside(point: Point, polygon: list[Point]) -> bool:
    if any(_on_segment(polygon[i], polygon[(i + 1) % len(polygon)], point) for i in range(len(polygon))):
        return True
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=False):
        if (a[1] > y) != (b[1] > y):
            edge_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < edge_x:
                inside = not inside
    return inside


def validate_polygon(points: list[Point]) -> None:
    if len(points) < 3:
        raise ValueError("Mỗi vùng cần ít nhất 3 đỉnh")
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
        raise ValueError("Đỉnh nằm ngoài frame hoặc không thuộc [0, 1]")
    if len({tuple(point) for point in points}) != len(points):
        raise ValueError("Polygon có đỉnh trùng nhau")
    edges = list(zip(points, points[1:] + points[:1], strict=False))
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i or j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _intersects(a, b, c, d):
                raise ValueError("Polygon tự cắt")
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1], strict=False)))
    if area < 1e-8:
        raise ValueError("Polygon có diện tích bằng 0")


def validate_zones(zone_a: list[Point], zone_b: list[Point]) -> None:
    validate_polygon(zone_a)
    validate_polygon(zone_b)
    for a, b in zip(zone_a, zone_a[1:] + zone_a[:1], strict=False):
        for c, d in zip(zone_b, zone_b[1:] + zone_b[:1], strict=False):
            if _intersects(a, b, c, d):
                raise ValueError("Vùng A và B chồng lấn hoặc chạm nhau")
    if _inside(zone_a[0], zone_b) or _inside(zone_b[0], zone_a):
        raise ValueError("Vùng A và B chồng lấn")


def _edge_inside_polygon(start: Point, end: Point, polygon: list[Point]) -> bool:
    """Check every interval split by boundary intersections (also handles concave ROI)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = dx * dx + dy * dy
    cuts = [0.0, 1.0]
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=False):
        ex, ey = b[0] - a[0], b[1] - a[1]
        ux, uy = a[0] - start[0], a[1] - start[1]
        denominator = dx * ey - dy * ex
        if abs(denominator) > 1e-12:
            t = (ux * ey - uy * ex) / denominator
            u = (ux * dy - uy * dx) / denominator
            if -1e-10 <= t <= 1 + 1e-10 and -1e-10 <= u <= 1 + 1e-10:
                cuts.append(max(0.0, min(1.0, t)))
        elif abs(ux * dy - uy * dx) < 1e-12:
            cuts.extend(max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / span))
                        for point in (a, b))
    ordered = sorted(set(cuts))
    return all(_inside((start[0] + dx * (left + right) / 2,
                        start[1] + dy * (left + right) / 2), polygon)
               for left, right in zip(ordered, ordered[1:], strict=False) if right - left > 1e-10)


def validate_head_out_zones(roi: list[Point], zone_a: list[Point], zone_b: list[Point]) -> None:
    """Head centers must traverse a valid ROI with separate A and B polygons."""
    validate_polygon(roi)
    validate_zones(zone_a, zone_b)  # No overlap or touching: a transition region must exist.
    for label, zone in (("A", zone_a), ("B", zone_b)):
        if any(not _inside(point, roi) for point in zone):
            raise ValueError(f"Zone {label} phải nằm trong ROI")
        if any(not _edge_inside_polygon(a, b, roi)
               for a, b in zip(zone, zone[1:] + zone[:1], strict=False)):
            raise ValueError(f"Zone {label} phải nằm hoàn toàn trong ROI")


def validate_flow_geometry(roi: list[Point], line: list[Point], direction: Point) -> None:
    """The OUT arrow is an explicit side, independent of line point order."""
    import math

    validate_polygon(roi)
    if len(line) != 2 or any(len(point) != 2 for point in line):
        raise ValueError("Counting Line cần đúng hai điểm")
    if any(not all(math.isfinite(value) and 0 <= value <= 1 for value in point) for point in line):
        raise ValueError("Counting Line nằm ngoài frame")
    if line[0] == line[1] or not _edge_inside_polygon(line[0], line[1], roi):
        raise ValueError("Counting Line phải nằm hoàn toàn trong ROI")
    if len(direction) != 2 or not all(math.isfinite(value) for value in direction):
        raise ValueError("Hướng OUT không hợp lệ")
    dx, dy = line[1][0] - line[0][0], line[1][1] - line[0][1]
    cross = dx * direction[1] - dy * direction[0]
    if abs(cross) < 1e-6 or math.hypot(*direction) < 1e-6:
        raise ValueError("Hướng OUT phải cắt ngang Counting Line")


def save_flow_geometry(path: Path, roi: list[Point], line: list[Point], direction: Point) -> Path:
    validate_flow_geometry(roi, line, direction)
    load_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    backup = path.with_name(f"{path.name}.rear_flow_out.bak")
    _write_yaml(backup, raw)
    raw.setdefault("rear_flow_out", {}).update({
        "enabled": True, "roi": [list(point) for point in roi],
        "counting_line": [list(point) for point in line],
        "out_direction": list(direction),
    })
    _write_yaml(path, raw)
    return backup


def restore_flow_geometry(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.rear_flow_out.bak")
    if not backup.is_file():
        raise FileNotFoundError("Chưa có bản sao lưu REAR Optical Flow")
    load_config(backup)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    previous = yaml.safe_load(backup.read_text(encoding="utf-8"))
    recovery = path.with_name(f"{path.name}.rear_flow_out.pre-restore.bak")
    _write_yaml(recovery, raw)
    raw["rear_flow_out"] = previous.get("rear_flow_out", {})
    _write_yaml(path, raw)
    return recovery


def _write_yaml(path: Path, raw: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        load_config(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_zones(path: Path, camera: str, zone_a: list[Point], zone_b: list[Point]) -> Path:
    camera = camera.lower()
    if camera not in {"front", "rear"}:
        raise ValueError("Chỉ hỗ trợ FRONT hoặc REAR")
    validate_zones(zone_a, zone_b)
    load_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if camera not in raw["cameras"]:
        raise ValueError("Camera không có trong cấu hình")
    raw["cameras"][camera]["zone_a"] = [list(point) for point in zone_a]
    raw["cameras"][camera]["zone_b"] = [list(point) for point in zone_b]
    backup = path.with_name(f"{path.name}.{camera}.bak")
    previous = yaml.safe_load(path.read_text(encoding="utf-8"))
    _write_yaml(backup, previous)
    # Keep the legacy whole-file snapshot for recovery, but UI restore is camera-scoped.
    _write_yaml(path.with_name(path.name + ".bak"), previous)
    _write_yaml(path, raw)
    return backup


def restore_backup(path: Path, camera: str) -> Path:
    camera = camera.lower()
    if camera not in {"front", "rear"}:
        raise ValueError("Chỉ hỗ trợ FRONT hoặc REAR")
    backup = path.with_name(f"{path.name}.{camera}.bak")
    if not backup.is_file():
        backup = path.with_name(path.name + ".bak")
    if not backup.is_file():
        raise FileNotFoundError("Chưa có bản sao lưu ROI")
    load_config(backup)
    load_config(path)
    previous = yaml.safe_load(backup.read_text(encoding="utf-8"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if camera not in previous["cameras"] or camera not in raw["cameras"]:
        raise ValueError("Camera không có trong cấu hình")
    zone_a = previous["cameras"][camera]["zone_a"]
    zone_b = previous["cameras"][camera]["zone_b"]
    validate_zones(zone_a, zone_b)
    recovery = path.with_name(f"{path.name}.{camera}.pre-restore.bak")
    _write_yaml(recovery, raw)
    raw["cameras"][camera]["zone_a"] = zone_a
    raw["cameras"][camera]["zone_b"] = zone_b
    _write_yaml(path, raw)
    return recovery


_HEAD_POLYGONS = ("roi", "zone_a", "zone_b")


def save_head_out_zones(path: Path, roi: list[Point], zone_a: list[Point], zone_b: list[Point]) -> Path:
    """Atomically save only the three head polygons and a scoped backup."""
    validate_head_out_zones(roi, zone_a, zone_b)
    load_config(path)
    previous = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    head = raw.setdefault("rear_head_out", {})
    head.update({key: [list(point) for point in points]
                 for key, points in zip(_HEAD_POLYGONS, (roi, zone_a, zone_b), strict=True)})
    backup = path.with_name(f"{path.name}.rear_head_out.bak")
    _write_yaml(backup, previous)
    _write_yaml(path, raw)
    return backup


def restore_head_out_backup(path: Path) -> Path:
    """Restore only head polygons, preserving later model and person settings."""
    backup = path.with_name(f"{path.name}.rear_head_out.bak")
    if not backup.is_file():
        raise FileNotFoundError("Chưa có bản sao lưu ROI Head Tracking")
    load_config(backup)
    load_config(path)
    previous = yaml.safe_load(backup.read_text(encoding="utf-8"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "rear_head_out" not in previous:
        raise ValueError("Bản sao lưu không có vùng REAR Head Tracking")
    selected = previous["rear_head_out"]
    validate_head_out_zones(*(selected[key] for key in _HEAD_POLYGONS))
    recovery = path.with_name(f"{path.name}.rear_head_out.pre-restore.bak")
    _write_yaml(recovery, raw)
    raw.setdefault("rear_head_out", {}).update({key: selected[key] for key in _HEAD_POLYGONS})
    _write_yaml(path, raw)
    return recovery
