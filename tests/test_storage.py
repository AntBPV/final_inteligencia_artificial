from datetime import datetime
from pathlib import Path
import tempfile

from analytics.assistant import OccupancyAssistant
from analytics.statistics import OccupancyStatistics
from storage.repository import CountRepository


def test_repository_and_statistics() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "test.sqlite3"
        repository = CountRepository(database)

        repository.save_event(
            session_id="session-test",
            track_id=1,
            event_type="entrada",
            occupancy=1,
            occurred_at="2026-11-19T12:00:00-05:00",
        )
        repository.save_event(
            session_id="session-test",
            track_id=2,
            event_type="entrada",
            occupancy=2,
            occurred_at="2026-11-19T12:01:00-05:00",
        )
        repository.save_event(
            session_id="session-test",
            track_id=1,
            event_type="salida",
            occupancy=1,
            occurred_at="2026-11-19T12:02:00-05:00",
        )

        statistics = OccupancyStatistics(repository.connection)
        assert statistics.total_entries() == 2
        assert statistics.total_exits() == 1
        assert statistics.current_occupancy() == 1
        assert statistics.maximum_occupancy() == 2

        csv_path = Path(directory) / "events.csv"
        assert repository.export_csv(csv_path) == 3
        assert csv_path.exists()
        assert "event_type" in csv_path.read_text(encoding="utf-8")

        repository.close()


def test_assistant_answers_occupancy() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "assistant.sqlite3"
        repository = CountRepository(database)
        repository.save_event(
            session_id="assistant-test",
            track_id=1,
            event_type="entrada",
            occupancy=1,
        )
        assistant = OccupancyAssistant(
            OccupancyStatistics(repository.connection)
        )

        answer = assistant.ask("cuantas personas hay ahora")

        assert "1" in answer
        assert "personas" in answer
        repository.close()


def test_assistant_now_query_is_not_interpreted_as_last_hour() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "assistant-now.sqlite3"
        repository = CountRepository(database)
        repository.save_event(
            session_id="assistant-test",
            track_id=1,
            event_type="entrada",
            occupancy=1,
        )
        assistant = OccupancyAssistant(
            OccupancyStatistics(repository.connection)
        )

        answer = assistant.ask("cuantas personas hay ahora")

        assert "ultima hora" not in answer.lower()
        assert "última hora" not in answer.lower()
        assert "personas" in answer
        repository.close()


def test_transformer_uses_supported_intent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "assistant-transformer.sqlite3"
        repository = CountRepository(database)
        repository.save_event(
            session_id="assistant-test",
            track_id=1,
            event_type="entrada",
            occupancy=1,
        )
        assistant = OccupancyAssistant(
            OccupancyStatistics(repository.connection)
        )

        answer = assistant.ask("dime cuantos visitantes entraron hoy")

        assert assistant.last_intent is not None
        assert assistant.last_intent.intent in {
            "occupancy_now",
            "entries_today",
            "exits_today",
            "maximum_occupancy",
            "last_hour",
            "daily_summary",
        }
        assert "Retroalimentacion:" in answer
        repository.close()
