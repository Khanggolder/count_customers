"""One mapping shared by CV, API, replay, assignment and exports."""

from __future__ import annotations

from .types import CrossingDirection, Direction

PASSENGER_DIRECTION = {
    ("FRONT", CrossingDirection.A_TO_B): Direction.IN,
    ("FRONT", CrossingDirection.B_TO_A): Direction.OUT,
    ("REAR", CrossingDirection.A_TO_B): Direction.OUT,
    ("REAR", CrossingDirection.B_TO_A): Direction.IN,
}


def expected_direction(camera: str, crossing_direction: str | CrossingDirection) -> Direction:
    try:
        crossing = CrossingDirection(crossing_direction)
        return PASSENGER_DIRECTION[(camera.upper(), crossing)]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid camera/crossing: {camera}/{crossing_direction}") from exc


def occupancy_change(direction: str | Direction) -> int:
    try:
        return 1 if Direction(direction) == Direction.IN else -1
    except ValueError as exc:
        raise ValueError(f"Invalid passenger direction: {direction}") from exc


def validate_passenger_direction(
    camera: str,
    direction: str | Direction,
    crossing_direction: str | CrossingDirection | None = None,
    occupancy_delta: int | None = None,
) -> None:
    if camera.upper() not in {"FRONT", "REAR"}:
        raise ValueError(f"Invalid camera: {camera}")
    try:
        actual = Direction(direction)
    except ValueError as exc:
        raise ValueError(f"Invalid passenger direction: {direction}") from exc
    if crossing_direction is not None and actual != expected_direction(camera, crossing_direction):
        raise ValueError("Crossing direction conflicts with passenger direction")
    if occupancy_delta is not None and occupancy_delta != occupancy_change(actual):
        raise ValueError("Occupancy delta conflicts with passenger direction")
