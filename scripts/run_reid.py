from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.database import (  # noqa: E402
    Base,
    SessionLocal,  # noqa: E402
    engine,
)
from backend.models import CountEventModel, ReIDMatchModel, SessionModel  # noqa: E402
from busreid.config import load_config  # noqa: E402
from busreid.matching import MatchingEngine  # noqa: E402
from busreid.reid import ReIDEncoder  # noqa: E402
from busreid.types import EmbeddingRecord  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--mode", choices=["bus-unpaired", "paired-gt", "technical-benchmark"], required=True)
    parser.add_argument("--config", default="configs/cameras.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    encoder = ReIDEncoder(config.reid.model, config.reid.checkpoint)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        session = db.get(SessionModel, args.session)
        if session is None:
            raise ValueError(f"Không thấy session {args.session}")
        if args.mode == "paired-gt" and session.mode != "PAIRED":
            raise ValueError("paired-gt yêu cầu session PAIRED và identity ground truth được xác minh")
        events = db.query(CountEventModel).filter(CountEventModel.session_id == args.session, CountEventModel.crop_path.is_not(None)).all()
        records: list[tuple[EmbeddingRecord, CountEventModel]] = []
        for event in events:
            image = cv2.imread(event.crop_path)
            if image is None:
                continue
            record = EmbeddingRecord(event.id, event.track_id, event.timestamp_s, encoder.aggregate([image]), event.passenger_id, Path(event.crop_path))
            records.append((record, event))
        entered = [record for record, event in records if event.direction == "IN"]
        exited = [record for record, event in records if event.direction == "OUT"]
        decisions = MatchingEngine(config.reid.threshold, config.reid.ambiguity_margin).match_batch(entered, exited, paired_and_verified=args.mode == "paired-gt")
        for decision in decisions:
            db.add(ReIDMatchModel(session_id=args.session, in_event_id=decision.in_event_id, out_event_id=decision.out_event_id, similarity=decision.similarity, status=decision.status.value, reason=decision.reason))
        db.commit()
    print(json.dumps([{"out_event_id": item.out_event_id, "in_event_id": item.in_event_id, "status": item.status.value, "similarity": item.similarity, "reason": item.reason} for item in decisions], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

