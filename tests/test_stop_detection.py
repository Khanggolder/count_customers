from __future__ import annotations

from datetime import datetime, timedelta, timezone

from busreid.stop_detection import StopPoint, StopVisitDetector, TelemetryPoint, assign_visit

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
STOP = StopPoint("S1", 1, 10.0, 106.0, 50)


def sample(
    second: int, lat: float | None, speed: float | None, door: bool | None, quality: str = "GOOD"
) -> TelemetryPoint:
    return TelemetryPoint(
        START + timedelta(seconds=second), lat, 106.0 if lat is not None else None, speed, door, door, quality
    )


def test_fast_near_stop_is_not_serving() -> None:
    rows = [sample(i, 10.0, 20.0, False) for i in range(5)]
    assert StopVisitDetector([STOP]).detect(rows) == []


def test_zero_speed_outside_range_is_not_stop() -> None:
    rows = [sample(i, 10.01, 0.0, True) for i in range(6)]
    assert StopVisitDetector([STOP]).detect(rows) == []


def test_traffic_pause_near_stop_with_closed_door_is_not_confirmed() -> None:
    rows = [sample(i, 10.0, 0.0, False) for i in range(6)]
    assert StopVisitDetector([STOP]).detect(rows) == []


def test_near_slow_open_door_stable_confirms_and_assigns() -> None:
    rows = [sample(i, 10.0, 0.0, True) for i in range(5)] + [sample(5, 10.001, 15.0, False)]
    visits = StopVisitDetector([STOP]).detect(rows)
    assert len(visits) == 1
    assert visits[0].status == "CONFIRMED"
    assert visits[0].departed_at == START + timedelta(seconds=5)
    assert assign_visit(START + timedelta(seconds=2), visits) == "S1"
    assert assign_visit(START + timedelta(seconds=20), visits) is None


def test_door_sensor_absent_is_uncertain_not_confirmed() -> None:
    rows = [sample(i, 10.0, 0.0, None) for i in range(5)] + [sample(5, 10.001, 15.0, None)]
    visits = StopVisitDetector([STOP]).detect(rows)
    assert visits[0].status == "UNCERTAIN"
    assert assign_visit(START + timedelta(seconds=2), visits) is None


def test_stale_or_missing_gps_does_not_confirm() -> None:
    rows = [sample(i, None, 0.0, True, "MISSING") for i in range(6)]
    assert StopVisitDetector([STOP]).detect(rows) == []


def test_missing_gps_during_open_door_does_not_depart() -> None:
    rows = [sample(i, 10.0, 0.0, True) for i in range(4)]
    rows += [sample(i, None, 0.0, True, "MISSING") for i in range(4, 8)]
    rows += [sample(8, 10.0, 0.0, True), sample(9, 10.001, 15.0, False)]
    visits = StopVisitDetector([STOP]).detect(rows)
    assert len(visits) == 1
    assert visits[0].departed_at == START + timedelta(seconds=9)


def test_stale_far_gps_during_open_door_does_not_depart() -> None:
    rows = [sample(i, 10.0, 0.0, True) for i in range(4)]
    rows += [sample(4, 10.01, 15.0, True, "STALE")]
    visits = StopVisitDetector([STOP]).detect(rows)
    assert visits[0].departed_at is None


def test_stop_sequence_and_repeated_stop_are_bounded() -> None:
    stop2 = StopPoint("S2", 2, 10.002, 106.0, 50)
    rows = [sample(i, 10.0, 0.0, True) for i in range(4)]
    rows += [sample(4, 10.001, 15.0, False)]
    rows += [sample(i, 10.0, 0.0, True) for i in range(5, 10)]  # repeated S1 must not create another visit
    rows += [
        TelemetryPoint(START + timedelta(seconds=i), 10.002, 106.0, 0.0, True, True) for i in range(10, 14)
    ]
    rows += [TelemetryPoint(START + timedelta(seconds=14), 10.003, 106.0, 15.0, False, False)]
    assert [visit.stop_id for visit in StopVisitDetector([STOP, stop2]).detect(rows)] == ["S1", "S2"]
