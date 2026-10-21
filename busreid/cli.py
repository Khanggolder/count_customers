from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.database import Base, SessionLocal, engine
from busreid.scenario import write_scenario
from busreid.service import flow_rows, load_scenario, replay_scenario


def demo_main() -> int:
    parser = argparse.ArgumentParser(description="Generate and replay the offline BusStop Flow AI demo")
    parser.add_argument("--scenario", type=Path, default=Path("data/sample/generated/scenario.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    write_scenario(args.scenario)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        result = replay_scenario(db, load_scenario(args.scenario))
        flow = flow_rows(db, result["trip_id"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "flow_by_stop.json").write_text(
        json.dumps(flow, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {**result, "flow_output": str(args.output / "flow_by_stop.json")}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(demo_main())
