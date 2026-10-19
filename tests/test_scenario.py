from __future__ import annotations

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import PassengerEventModel, TelemetrySampleModel
from busreid.scenario import build_scenario
from busreid.service import flow_rows, replay_scenario


def test_complete_scenario_exact_counts_and_idempotency(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'scenario.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    scenario = build_scenario()
    with factory() as db:
        first = replay_scenario(db, scenario)
        replay_scenario(db, scenario)
        flow = flow_rows(db, first["trip_id"])
        assert len(flow) == 8
        for actual, expected in zip(flow, scenario["ground_truth"]["flow"], strict=True):
            for key, value in expected.items():
                assert actual[key] == value
            assert actual["visit_status"] == "CONFIRMED"
        assert all(row["warning"] is None and row["onboard_after_stop"] >= 0 for row in flow)
        assert db.scalar(select(func.count()).select_from(PassengerEventModel)) == len(
            scenario["passenger_events"]
        )
        assert db.scalar(select(func.count()).select_from(TelemetrySampleModel)) == len(scenario["telemetry"])
        unassigned = list(
            db.scalars(
                select(PassengerEventModel).where(PassengerEventModel.assignment_status == "UNASSIGNED")
            )
        )
        assert [row.event_uuid for row in unassigned] == ["SIM-normal-UNASSIGNED-CONTROL"]


def test_negative_occupancy_is_separate_control() -> None:
    scenario = build_scenario(kind="negative_occupancy")
    assert scenario["ground_truth"]["flow"][-1]["onboard_after_stop"] < 0
    assert build_scenario()["ground_truth"]["flow"][-1]["onboard_after_stop"] >= 0


def test_negative_control_replays_separately_and_warns(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'controls.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        normal_id = replay_scenario(db, build_scenario())["trip_id"]
        negative_id = replay_scenario(db, build_scenario(kind="negative_occupancy"))["trip_id"]
        assert normal_id != negative_id
        assert flow_rows(db, normal_id)[-1]["warning"] is None
        assert flow_rows(db, negative_id)[-1]["warning"] == "NEGATIVE_OCCUPANCY_REVIEW_REQUIRED"


def test_synthetic_provenance_is_never_real_ai() -> None:
    scenario = build_scenario()
    assert all(item["provenance"] == "SIMULATED_EVENTS" for item in scenario["passenger_events"])
    assert scenario["trip"]["provenance"] == "SIMULATED"
