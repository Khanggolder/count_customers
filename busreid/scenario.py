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

FLOW_MATRIX = [
    (4, 0, 1, 0),
    (3, 1, 0, 2),
    (2, 0, 1, 3),
    (5, 1, 0, 2),
    (1, 2, 1, 4),
    (2, 0, 0, 3),
    (1, 1, 2, 3),
    (0, 2, 1, 8),
]


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_scenario(seed: int = 42, kind: str = "normal") -> dict:
    """Build a deterministic, fully labelled scenario; seed is recorded for reproducibility."""
    if kind not in {"normal", "negative_occupancy"}:
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
        for camera, direction, count in (
            ("front", "IN", flows[0]),
            ("front", "OUT", flows[1]),
            ("rear", "IN", flows[2]),
            ("rear", "OUT", flows[3]),
        ):
            for number in range(count):
                uid = f"SIM-{kind}-{stop_index + 1:02d}-{camera}-{direction}-{number + 1:03d}"
                events.append(
                    {
                        "event_uuid": uid,
                        "camera": camera,
                        "direction": direction,
                        "track_id": f"{camera}:sim:{stop_index + 1}:{direction}:{number + 1}",
                        "video_frame": None,
                        "source_pts": (event_time - start).total_seconds() + number * 0.01,
                        "event_time_utc": _iso(event_time + timedelta(milliseconds=number * 10)),
                        "source_type": "SIMULATED_EVENT_GENERATOR",
                        "source_confidence": 1.0,
                        "provenance": "SIMULATED_EVENTS",
                        "ground_truth_stop_id": stop["stop_id"],
                    }
                )
        cursor += timedelta(seconds=15)

    events.append(
        {
            "event_uuid": f"SIM-{kind}-UNASSIGNED-CONTROL",
            "camera": "front",
            "direction": "IN",
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
    initial_occupancy = 8 if kind == "normal" else 0
    occupancy = initial_occupancy
    for stop, flows in zip(stops, FLOW_MATRIX, strict=True):
        total_in, total_out = flows[0] + flows[2], flows[1] + flows[3]
        occupancy += total_in - total_out
        expected.append(
            {
                "stop_id": stop["stop_id"],
                "front_in": flows[0],
                "front_out": flows[1],
                "rear_in": flows[2],
                "rear_out": flows[3],
                "total_in": total_in,
                "total_out": total_out,
                "onboard_after_stop": occupancy,
            }
        )

    return {
        "schema_version": "1.0",
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
            "trip_id": "SIM-R01-20261019-0800" if kind == "normal" else "SIM-R01-NEGATIVE-CONTROL",
            "vehicle_id": "SIM-BUS-01",
            "start_time_utc": _iso(start),
            "end_time_utc": _iso(cursor),
            "mode": "SYNTHETIC_PAIRED",
            "initial_occupancy": initial_occupancy,
            "status": "COMPLETED",
            "coverage_complete": True,
            "provenance": "SIMULATED",
        },
        "stops": stops,
        "telemetry": telemetry,
        "passenger_events": events,
        "ground_truth": {
            "stop_visits": visits_truth,
            "flow": expected,
            "unassigned_event_uuids": [f"SIM-{kind}-UNASSIGNED-CONTROL"],
        },
        "disclaimer": "Not real GPS, not AI output, and not a claim about Route 01 operations.",
    }


def write_scenario(path: str | Path, seed: int = 42) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_scenario(seed), ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
