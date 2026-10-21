"""Mandatory boarding/alighting policy shared by every event ingestion path."""

from __future__ import annotations

from .types import Direction

ALLOWED_DIRECTIONS = {"FRONT": Direction.IN, "REAR": Direction.OUT}


def validate_passenger_direction(camera: str, direction: str | Direction) -> None:
    expected = ALLOWED_DIRECTIONS.get(camera.upper())
    actual = direction.value if isinstance(direction, Direction) else direction
    if expected is None or actual != expected.value:
        raise ValueError(
            f"Invalid passenger direction: {camera.upper()} accepts only "
            f"{expected.value if expected else 'FRONT/REAR camera'}; received {actual}"
        )
