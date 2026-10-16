from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OccupancySnapshot:
    value: int | None
    warning: str | None


class OccupancyEstimator:
    def __init__(self, initial: int = 0, paired_sources: bool = True) -> None:
        self.initial = initial
        self.paired_sources = paired_sources

    def estimate(self, count_in: int, count_out: int) -> OccupancySnapshot:
        if not self.paired_sources:
            return OccupancySnapshot(None, "Không tính occupancy vì hai nguồn không cùng chuyến (UNPAIRED)")
        value = self.initial + count_in - count_out
        warning = None
        if value < 0:
            warning = "Occupancy âm: kiểm tra đồng bộ nguồn, số khách ban đầu hoặc sự kiện bỏ sót"
        return OccupancySnapshot(value, warning)

