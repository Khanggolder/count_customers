from datetime import datetime, timezone

from busreid.timebase import SourceClock


def test_offset_and_reconnect_stay_monotonic() -> None:
    clock = SourceClock(datetime(2026, 1, 1, tzinfo=timezone.utc), offset_ms=500)
    assert clock.map_pts(1).timestamp() == datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() + 1.5
    before = clock.map_pts(2)
    clock.reconnect()
    after = clock.map_pts(0)
    assert after > before
