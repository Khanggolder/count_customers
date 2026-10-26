"""Legacy SQLite migration is backed up and never invents reverse directions."""

from __future__ import annotations

import sqlite3

from scripts.migrate_bidirectional import migrate


def test_migration_backs_up_and_classifies_legacy_events(tmp_path) -> None:
    database = tmp_path / "legacy.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE passenger_events (id INTEGER PRIMARY KEY, camera TEXT, direction TEXT, event_uuid TEXT)"
        )
        connection.executemany(
            "INSERT INTO passenger_events (camera, direction, event_uuid) VALUES (?, ?, ?)",
            [("FRONT", "IN", "a"), ("REAR", "OUT", "b"), ("FRONT", "OUT", "c"), ("REAR", "IN", "d")],
        )
    backup = migrate(database)
    assert backup.is_file()
    with sqlite3.connect(backup) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(passenger_events)")}
        assert "crossing_direction" not in columns
        assert connection.execute("SELECT COUNT(*) FROM passenger_events").fetchone()[0] == 4
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT event_uuid, crossing_direction, occupancy_delta, direction_status "
            "FROM passenger_events ORDER BY event_uuid"
        ).fetchall()
    assert rows == [
        ("a", "A_TO_B", 1, "INFERRED_LEGACY"),
        ("b", "A_TO_B", -1, "INFERRED_LEGACY"),
        ("c", None, None, "REVIEW_REQUIRED"),
        ("d", None, None, "REVIEW_REQUIRED"),
    ]
    migrate(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM passenger_events").fetchone()[0] == 4
