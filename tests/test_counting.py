from __future__ import annotations

from busreid.counting import DirectionCounter
from busreid.types import Detection, Direction

SHAPE = (100, 100, 3)
ZONE_A = [(0, 0), (1, 0), (1, 0.45), (0, 0.45)]
ZONE_B = [(0, 0.55), (1, 0.55), (1, 1), (0, 1)]


def det(track: str, y: int) -> Detection:
    return Detection(track, (40, y - 20, 60, y), 0.9)


def feed(counter: DirectionCounter, track: str, ys: list[int]) -> list:
    return [event for index, y in enumerate(ys) if (event := counter.update(det(track, y), SHAPE, index, index / 10))]


def test_outside_to_inside_counted_once() -> None:
    counter = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, 2, 5)
    events = feed(counter, "front:track00001", [20, 20, 50, 50, 80, 80, 80])
    assert len(events) == 1
    assert events[0].direction == Direction.IN


def test_standing_at_threshold_is_not_counted() -> None:
    counter = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, 3, 5)
    assert feed(counter, "front:track00001", [50] * 20) == []


def test_reverse_direction_emits_out() -> None:
    counter = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, 2, 2)
    events = feed(counter, "front:track00001", [80, 80, 50, 50, 20, 20])
    assert [event.direction for event in events] == [Direction.OUT]


def test_oscillation_debounced() -> None:
    counter = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, 1, 10)
    events = feed(counter, "front:track00001", [20, 80, 20, 80, 20])
    assert len(events) == 1


def test_multiple_people_are_independent() -> None:
    counter = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, 1, 1)
    events = []
    for frame, y in enumerate((20, 80)):
        for track in ("front:track00001", "front:track00002"):
            event = counter.update(det(track, y), SHAPE, frame, float(frame))
            if event:
                events.append(event)
    assert len(events) == 2
    assert {event.track_id for event in events} == {"front:track00001", "front:track00002"}


def test_camera_namespaces_do_not_collide() -> None:
    front = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, 1, 1)
    rear = DirectionCounter("rear", ZONE_A, ZONE_B, Direction.OUT, 1, 1)
    assert feed(front, "front:track00001", [20, 80])[0].direction == Direction.IN
    assert feed(rear, "rear:track00001", [20, 80])[0].direction == Direction.OUT


def test_lost_track_does_not_create_false_crossing() -> None:
    counter = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, 1, 1)
    counter.update(det("front:old", 20), SHAPE, 0, 0)
    counter.prune(set())
    assert counter.update(det("front:new", 80), SHAPE, 10, 1) is None

