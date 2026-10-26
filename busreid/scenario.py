from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

STOP_NAMES = [
    "Bến Thành B",
    "Trường Ernst Thalmann",
    "KTX Trần Hưng Đạo",
    "Rạp Hưng Đạo",
    "Bệnh viện Răng Hàm Mặt",
    "Trần Đình Xu",
    "Tổng Cty Samco",
    "Chợ Nanci",
]

# FRONT-IN, FRONT-OUT, REAR-IN, REAR-OUT at each stop.
FLOW_MATRIX = [
    (4, 1, 0, 0),
    (3, 0, 1, 2),
    (2, 0, 0, 3),
    (5, 1, 1, 2),
    (1, 0, 0, 4),
    (2, 0, 0, 3),
    (1, 0, 0, 3),
    (0, 0, 0, 8),
]


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_scenario(seed: int = 42, kind: str = "normal") -> dict:
    """Build a deterministic, fully labelled scenario; seed is recorded for reproducibility."""
    if kind not in {"normal", "negative_occupancy", "incomplete_camera"}:
        raise ValueError("Unknown scenario kind")
    start = datetime(2026, 10, 19, 1, 0, tzinfo=timezone.utc)  # 08:00 Asia/Ho_Chi_Minh
    base_lat, base_lon = 10.77175, 106.69835
    stops = []
    for index, name in enumerate(STOP_NAMES, start=1):
        stops.append(
            {
                "stop_id": f"SIM-R01-S{index:02d}",
                "sequence": index,
                "name": name,
                "latitude": round(base_lat - index * 0.00125, 6),
                "longitude": round(base_lon - index * 0.00105, 6),
                "radius_m": 45.0,
                "coordinate_source": "SYNTHETIC_ROUTE_GEOMETRY_NOT_REAL_STOP_COORDINATES",
                "name_source_url": "https://xe-buyt.com/ben-thanh-ben-xe-cho-lon.html",
                "retrieved_date": "2026-10-19",
                "verified": False,
            }
        )

    telemetry: list[dict] = []
    events: list[dict] = []
    visits_truth: list[dict] = []
    cursor = start
    # Explicit negative controls: zero speed outside geofence, door open outside geofence, missing GPS.
    telemetry.extend(
        [
            {
                "event_time_utc": _iso(cursor),
                "latitude": base_lat,
                "longitude": base_lon,
                "speed_kmh": 0.0,
                "door_front": False,
                "door_rear": False,
                "quality": "GOOD",
                "case": "TRAFFIC_JAM_OUTSIDE_STOP",
            },
            {
                "event_time_utc": _iso(cursor + timedelta(seconds=1)),
                "latitude": base_lat,
                "longitude": base_lon,
                "speed_kmh": 0.0,
                "door_front": True,
                "door_rear": False,
                "quality": "GOOD",
                "case": "DOOR_OPEN_OUTSIDE_STOP",
            },
            {
                "event_time_utc": _iso(cursor + timedelta(seconds=2)),
                "latitude": None,
                "longitude": None,
                "speed_kmh": 0.0,
                "door_front": None,
                "door_rear": None,
                "quality": "MISSING",
                "case": "GPS_MISSING",
            },
        ]
    )
    cursor += timedelta(seconds=5)

    for stop_index, (stop, flows) in enumerate(zip(stops, FLOW_MATRIX, strict=True)):
        lat, lon = stop["latitude"], stop["longitude"]
        # Red-light-like pause near the next stop: short, door closed, therefore never confirmed.
        for offset, speed in ((0, 1.0), (1, 0.0), (2, 12.0)):
            telemetry.append(
                {
                    "event_time_utc": _iso(cursor + timedelta(seconds=offset)),
                    "latitude": lat + 0.00028,
                    "longitude": lon,
                    "speed_kmh": speed,
                    "door_front": False,
                    "door_rear": False,
                    "quality": "GOOD",
                    "case": "RED_LIGHT_NEAR_STOP",
                }
            )
        cursor += timedelta(seconds=4)
        arrived = cursor
        for offset in range(0, 6):
            telemetry.append(
                {
                    "event_time_utc": _iso(cursor + timedelta(seconds=offset)),
                    "latitude": lat + (0.00001 if offset % 2 else 0),
                    "longitude": lon,
                    "speed_kmh": 1.2 if offset == 0 else 0.0,
                    "door_front": offset < 5,
                    "door_rear": offset < 5,
                    "quality": "NOISY" if offset == 1 else "GOOD",
                    "case": "SERVING_STOP",
                }
            )
        departed = cursor + timedelta(seconds=7)
        telemetry.append(
            {
                "event_time_utc": _iso(cursor + timedelta(seconds=6)),
                "latitude": lat,
                "longitude": lon,
                "speed_kmh": 2.0,
                "door_front": False,
                "door_rear": False,
                "quality": "GOOD",
                "case": "DOORS_CLOSED",
            }
        )
        telemetry.append(
            {
                "event_time_utc": _iso(departed),
                "latitude": lat - 0.00075,
                "longitude": lon - 0.00075,
                "speed_kmh": 18.0,
                "door_front": False,
                "door_rear": False,
                "quality": "GOOD",
                "case": "DEPARTING",
            }
        )
        visits_truth.append(
            {
                "stop_id": stop["stop_id"],
                "arrived_at": _iso(arrived),
                "departed_at": _iso(departed),
                "status": "CONFIRMED",
            }
        )
        event_time = arrived + timedelta(seconds=4)
        for camera, crossing, direction, count in (
            ("front", "A_TO_B", "IN", flows[0]),
            ("front", "B_TO_A", "OUT", flows[1]),
            ("rear", "B_TO_A", "IN", flows[2]),
            ("rear", "A_TO_B", "OUT", flows[3]),
        ):
            if kind == "incomplete_camera" and camera == "rear":
                continue
            for number in range(count):
                event_offset_ms = (
                    500
                    if stop_index == 3 and camera == "front" and direction == "OUT" and number == 0
                    else number * 10
                )
                uid = f"SIM-v3-{kind}-{stop_index + 1:02d}-{camera}-{direction}-{number + 1:03d}"
                events.append(
                    {
                        "event_uuid": uid,
                        "camera": camera,
                        "crossing_direction": crossing,
                        "direction": direction,
                        "occupancy_delta": 1 if direction == "IN" else -1,
                        "track_id": (
                            f"front:sim:{stop_index + 1}:return:1"
                            if stop_index == 3 and camera == "front" and number == 0
                            else f"{camera}:sim:{stop_index + 1}:{direction}:{number + 1}"
                        ),
                        "video_frame": None,
                        "source_pts": (event_time - start).total_seconds() + event_offset_ms / 1000,
                        "event_time_utc": _iso(event_time + timedelta(milliseconds=event_offset_ms)),
                        "source_type": "SIMULATED_EVENT_GENERATOR",
                        "source_confidence": 1.0,
                        "provenance": "SIMULATED_EVENTS",
                        "ground_truth_stop_id": stop["stop_id"],
                    }
                )
        cursor += timedelta(seconds=15)

    events.append(
        {
            "event_uuid": f"SIM-v3-{kind}-UNASSIGNED-CONTROL",
            "camera": "front",
            "crossing_direction": "A_TO_B",
            "direction": "IN",
            "occupancy_delta": 1,
            "track_id": "front:sim:unassigned",
            "video_frame": None,
            "source_pts": 3.0,
            "event_time_utc": _iso(start + timedelta(seconds=3)),
            "source_type": "SIMULATED_EVENT_GENERATOR",
            "source_confidence": 1.0,
            "provenance": "SIMULATED_EVENTS",
            "ground_truth_stop_id": None,
        }
    )
    expected = []
    initial_occupancy = 8 if kind != "negative_occupancy" else 0
    occupancy = initial_occupancy
    for stop, flows in zip(stops, FLOW_MATRIX, strict=True):
        front_in, front_out, rear_in, rear_out = flows
        if kind == "incomplete_camera":
            rear_in = rear_out = 0
        total_in = front_in + rear_in
        total_out = front_out + rear_out
        occupancy += total_in - total_out
        expected.append(
            {
                "stop_id": stop["stop_id"],
                "front_in": front_in,
                "front_out": front_out,
                "rear_in": rear_in,
                "rear_out": rear_out,
                "total_in": total_in,
                "total_out": total_out,
                "onboard_after_stop": occupancy if kind != "incomplete_camera" else None,
            }
        )

    return {
        "schema_version": "3.0",
        "label": "DỮ LIỆU MÔ PHỎNG / SIMULATED EVENTS",
        "seed": seed,
        "route": {
            "route_id": "SIM-R01",
            "name": "Tuyến 01 – kịch bản nghiên cứu",
            "direction": "Bến xe Chợ Lớn",
            "source": "https://xe-buyt.com/ben-thanh-ben-xe-cho-lon.html; official RouteFinding returned HTTP 403 during verification",
            "retrieved_date": "2026-10-19",
            "verified": False,
        },
        "trip": {
            "trip_id": {
                "normal": "SIM-R01-v3-20261019-0800",
                "negative_occupancy": "SIM-R01-v3-NEGATIVE-CONTROL",
                "incomplete_camera": "SIM-R01-v3-INCOMPLETE-CAMERA",
            }[kind],
            "vehicle_id": "SIM-BUS-01",
            "start_time_utc": _iso(start),
            "end_time_utc": _iso(cursor),
            "mode": "UNPAIRED" if kind == "incomplete_camera" else "SYNTHETIC_PAIRED",
            "initial_occupancy": initial_occupancy,
            "status": "COMPLETED",
            "coverage_complete": kind != "incomplete_camera",
            "provenance": "SIMULATED",
        },
        "stops": stops,
        "telemetry": telemetry,
        "passenger_events": events,
        "camera_roles": ["front"] if kind == "incomplete_camera" else ["front", "rear"],
        "ground_truth": {
            "stop_visits": visits_truth,
            "flow": expected,
            "unassigned_event_uuids": [f"SIM-v3-{kind}-UNASSIGNED-CONTROL"],
            "non_event_controls": ["APPROACH_AND_TURN_BACK_BEFORE_CROSSING"],
            "non_event_movements": [
                {
                    "camera": "front",
                    "track_id": "front:sim:approach-return",
                    "sequence": ["OUTSIDE", "A", "OUTSIDE"],
                    "expected_events": 0,
                    "reason": "Never reached zone B",
                }
            ],
            "completed_return_pair": {
                "stop_id": stops[3]["stop_id"],
                "track_id": "front:sim:4:return:1",
                "directions": ["IN", "OUT"],
                "expected_events": 2,
            },
        },
        "disclaimer": "Not real GPS, not AI output, and not a claim about Route 01 operations.",
    }


def write_scenario(path: str | Path, seed: int = 42) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_scenario(seed), ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
