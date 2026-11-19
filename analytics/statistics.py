from datetime import datetime, timedelta
import sqlite3
from typing import Any


class OccupancyStatistics:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def total_entries(self, start: datetime | None = None) -> int:
        row = self._counts("entrada", start)
        return int(row["total"])

    def total_exits(self, start: datetime | None = None) -> int:
        row = self._counts("salida", start)
        return int(row["total"])

    def _counts(self, event_type: str, start: datetime | None) -> sqlite3.Row:
        if start is None:
            return self.connection.execute(
                "SELECT COUNT(*) AS total FROM count_events WHERE event_type = ?",
                (event_type,),
            ).fetchone()

        return self.connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM count_events
            WHERE event_type = ? AND occurred_at >= ?
            """,
            (event_type, start.astimezone().isoformat(timespec="seconds")),
        ).fetchone()

    def current_occupancy(self) -> int:
        row = self.connection.execute(
            """
            SELECT occupancy
            FROM count_events
            ORDER BY datetime(occurred_at) DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        return int(row["occupancy"]) if row else 0

    def maximum_occupancy(self, start: datetime | None = None) -> int:
        if start is None:
            row = self.connection.execute(
                "SELECT COALESCE(MAX(occupancy), 0) AS maximum FROM count_events"
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                SELECT COALESCE(MAX(occupancy), 0) AS maximum
                FROM count_events
                WHERE occurred_at >= ?
                """,
                (start.astimezone().isoformat(timespec="seconds"),),
            ).fetchone()
        return int(row["maximum"])

    def summarize_today(self) -> dict[str, int]:
        now = datetime.now().astimezone()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return {
            "entradas": self.total_entries(start),
            "salidas": self.total_exits(start),
            "ocupacion": self.current_occupancy(),
            "ocupacion_maxima": self.maximum_occupancy(start),
        }

    def summarize_last_hour(self) -> dict[str, int]:
        start = datetime.now().astimezone() - timedelta(hours=1)
        return {
            "entradas": self.total_entries(start),
            "salidas": self.total_exits(start),
            "ocupacion": self.current_occupancy(),
            "ocupacion_maxima": self.maximum_occupancy(start),
        }
