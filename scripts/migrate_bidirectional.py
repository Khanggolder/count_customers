"""Explicit, backed-up SQLite schema migration for bidirectional passenger events."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def migrate(database: Path) -> Path:
    if not database.is_file():
        raise FileNotFoundError(database)
    backup = database.with_name(
        f"{database.name}.before-bidirectional-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.bak"
    )
    with sqlite3.connect(database) as source, sqlite3.connect(backup) as destination:
        source.backup(destination)
    try:
        with sqlite3.connect(database) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(passenger_events)")}
            if not columns:
                raise ValueError("Database has no passenger_events table")
            connection.execute("BEGIN IMMEDIATE")
            if "crossing_direction" not in columns:
                connection.execute("ALTER TABLE passenger_events ADD COLUMN crossing_direction VARCHAR(16)")
            if "occupancy_delta" not in columns:
                connection.execute("ALTER TABLE passenger_events ADD COLUMN occupancy_delta INTEGER")
            if "direction_status" not in columns:
                connection.execute("ALTER TABLE passenger_events ADD COLUMN direction_status VARCHAR(32)")
            # Old valid forward-only events can be inferred from the former application policy.
            connection.execute(
                """UPDATE passenger_events
                   SET crossing_direction = 'A_TO_B',
                       occupancy_delta = CASE direction WHEN 'IN' THEN 1 ELSE -1 END,
                       direction_status = 'INFERRED_LEGACY'
                   WHERE crossing_direction IS NULL AND direction_status IS NULL
                     AND ((camera = 'FRONT' AND direction = 'IN')
                       OR (camera = 'REAR' AND direction = 'OUT'))"""
            )
            # Never reinterpret old reverse/delta records without a reliable crossing label.
            connection.execute(
                """UPDATE passenger_events SET direction_status = 'REVIEW_REQUIRED'
                   WHERE direction_status IS NULL"""
            )
    except Exception:
        # The backup remains available for manual recovery; never delete user data.
        raise
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite .db path (stop API/workers first)")
    args = parser.parse_args()
    backup = migrate(args.database.resolve())
    print(f"Migration complete. Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
