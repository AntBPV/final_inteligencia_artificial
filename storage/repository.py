import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable


class CountRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.database_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS count_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL CHECK(event_type IN ('entrada', 'salida')),
                occupancy INTEGER NOT NULL CHECK(occupancy >= 0)
            );

            CREATE INDEX IF NOT EXISTS idx_count_events_occurred_at
            ON count_events(occurred_at);

            CREATE INDEX IF NOT EXISTS idx_count_events_session
            ON count_events(session_id);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def save_event(
        self,
        session_id: str,
        track_id: int,
        event_type: str,
        occupancy: int,
        occurred_at: str | None = None,
    ) -> int:
        if event_type not in {"entrada", "salida"}:
            raise ValueError("event_type debe ser entrada o salida")
        if occupancy < 0:
            raise ValueError("occupancy no puede ser negativo")

        timestamp = occurred_at or datetime.now().astimezone().isoformat(timespec="seconds")
        cursor = self.connection.execute(
            """
            INSERT INTO count_events(session_id, occurred_at, track_id, event_type, occupancy)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, timestamp, track_id, event_type, occupancy),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def export_csv(self, output_path: str | Path) -> int:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.connection.execute(
            """
            SELECT id, session_id, occurred_at, track_id, event_type, occupancy
            FROM count_events
            ORDER BY occurred_at ASC, id ASC
            """
        ).fetchall()

        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["id", "session_id", "occurred_at", "track_id", "event_type", "occupancy"]
            )
            for row in rows:
                writer.writerow(
                    [row["id"], row["session_id"], row["occurred_at"],
                     row["track_id"], row["event_type"], row["occupancy"]]
                )
        return len(rows)
