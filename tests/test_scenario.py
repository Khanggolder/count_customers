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
        assert flow[-1]["warning"] == "NEGATIVE_OCCUPANCY_REVIEW_REQUIRED"
        assert db.scalar(select(func.count()).select_from(PassengerEventModel)) == len(
            scenario["passenger_events"]
        )
        assert db.scalar(select(func.count()).select_from(TelemetrySampleModel)) == len(scenario["telemetry"])
        unassigned = list(
            db.scalars(
                select(PassengerEventModel).where(PassengerEventModel.assignment_status == "UNASSIGNED")
            )
        )
        assert [row.event_uuid for row in unassigned] == ["SIM-UNASSIGNED-CONTROL"]


def test_synthetic_provenance_is_never_real_ai() -> None:
    scenario = build_scenario()
    assert all(item["provenance"] == "SIMULATED_EVENTS" for item in scenario["passenger_events"])
    assert scenario["trip"]["provenance"] == "SIMULATED"
