"""Geometry and persistence for camera ROIs (coordinates are normalized)."""

from __future__ import annotations

import os
import shutil
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
    if len(set(points)) != len(points):
        raise ValueError("Polygon có đỉnh trùng nhau")
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1], strict=False)))
    if area < 1e-8:
        raise ValueError("Polygon có diện tích bằng 0")
    edges = list(zip(points, points[1:] + points[:1], strict=False))
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges):
            if j <= i or j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _intersects(a, b, c, d):
                raise ValueError("Polygon tự cắt")


def validate_zones(zone_a: list[Point], zone_b: list[Point]) -> None:
    validate_polygon(zone_a)
    validate_polygon(zone_b)
    for a, b in zip(zone_a, zone_a[1:] + zone_a[:1], strict=False):
        for c, d in zip(zone_b, zone_b[1:] + zone_b[:1], strict=False):
            if _intersects(a, b, c, d):
                raise ValueError("Vùng A và B chồng lấn hoặc chạm nhau")
    if _inside(zone_a[0], zone_b) or _inside(zone_b[0], zone_a):
        raise ValueError("Vùng A và B chồng lấn")


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
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    _write_yaml(path, raw)
    return backup


def restore_backup(path: Path) -> None:
    backup = path.with_name(path.name + ".bak")
    if not backup.is_file():
        raise FileNotFoundError("Chưa có bản sao lưu ROI")
    load_config(backup)
    raw = yaml.safe_load(backup.read_text(encoding="utf-8"))
    _write_yaml(path, raw)
