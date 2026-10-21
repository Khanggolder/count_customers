from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(slots=True)
class SourceClock:
    recording_start_time: datetime
    offset_ms: int = 0
    _last_time: datetime | None = None
    _generation_base_s: float = 0.0
    _last_pts_s: float = 0.0

    def __post_init__(self) -> None:
        if self.recording_start_time.tzinfo is None:
            raise ValueError("recording_start_time must include timezone")
        self.recording_start_time = self.recording_start_time.astimezone(timezone.utc)

    def map_pts(self, pts_s: float) -> datetime:
        if pts_s < 0:
            raise ValueError("PTS must be non-negative")
        effective = self._generation_base_s + pts_s
        value = self.recording_start_time + timedelta(seconds=effective, milliseconds=self.offset_ms)
        if self._last_time is not None and value <= self._last_time:
            value = self._last_time + timedelta(microseconds=1)
        self._last_pts_s = pts_s
        self._last_time = value
        return value

    def reconnect(self) -> None:
        self._generation_base_s += self._last_pts_s
        self._last_pts_s = 0.0
