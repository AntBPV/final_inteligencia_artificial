from datetime import datetime

from .intent_transformer import IntentTransformer, IntentResult
from .statistics import OccupancyStatistics


class OccupancyAssistant:
    def __init__(self, statistics: OccupancyStatistics) -> None:
        self.statistics = statistics
        self.intent_transformer = IntentTransformer()
        self.last_intent: IntentResult | None = None
        self.confidence_threshold = 0.48

    def _feedback(self) -> str:
        if self.last_intent is None:
            return ""
        intent_name = self.last_intent.intent
        confidence = self.last_intent.confidence
        return (
            f"\nRetroalimentacion: la consulta fue interpretada como "
            f"{intent_name} con confianza {confidence:.0%}."
        )

    def ask(self, question: str) -> str:
        text = question.lower().strip()

        if not text:
            return "Escribe una consulta, por ejemplo: cuantas personas hay ahora."

        result = self.intent_transformer.predict(text)
        self.last_intent = result
        if result.confidence < self.confidence_threshold:
            return (
                "No tengo suficiente confianza para interpretar la consulta."
                "\nPrueba con: cuantas personas hay ahora, entradas hoy, "
                "salidas hoy, ocupacion maxima, resumen de hoy o ultima hora."
                + self._feedback()
            )

        if result.intent == "last_hour":
            summary = self.statistics.summarize_last_hour()
            return (
                f"En la ultima hora hubo {summary['entradas']} entradas y "
                f"{summary['salidas']} salidas. Actualmente hay "
                f"{summary['ocupacion']} personas y la ocupacion maxima "
                f"registrada fue {summary['ocupacion_maxima']}."
                + self._feedback()
            )

        if result.intent == "maximum_occupancy":
            maximum = self.statistics.maximum_occupancy()
            return (
                f"La ocupacion maxima historica registrada fue {maximum} personas."
                + self._feedback()
            )

        if result.intent == "entries_today":
            total = self.statistics.total_entries(
                datetime.now().astimezone().replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            )
            return f"Hoy se registraron {total} entradas." + self._feedback()

        if result.intent == "exits_today":
            total = self.statistics.total_exits(
                datetime.now().astimezone().replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            )
            return f"Hoy se registraron {total} salidas." + self._feedback()

        if result.intent == "occupancy_now":
            summary = self.statistics.summarize_today()
            return (
                f"Actualmente hay {summary['ocupacion']} personas. "
                f"Hoy hay {summary['entradas']} entradas y "
                f"{summary['salidas']} salidas."
                + self._feedback()
            )

        if result.intent == "daily_summary":
            summary = self.statistics.summarize_today()
            return (
                f"Hoy: {summary['entradas']} entradas, "
                f"{summary['salidas']} salidas, "
                f"{summary['ocupacion']} personas actualmente y "
                f"{summary['ocupacion_maxima']} como ocupacion maxima."
                + self._feedback()
            )

        return (
            "No entiendo la consulta. Prueba: cuantas personas hay ahora, "
            "entradas hoy, salidas hoy, ocupacion maxima o resumen de la ultima hora."
            + self._feedback()
        )
