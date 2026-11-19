from datetime import datetime

from .statistics import OccupancyStatistics


class OccupancyAssistant:
    def __init__(self, statistics: OccupancyStatistics) -> None:
        self.statistics = statistics

    def ask(self, question: str) -> str:
        text = question.lower().strip()

        if not text:
            return "Escribe una consulta, por ejemplo: cuantas personas hay ahora."

        if any(
            phrase in text
            for phrase in (
                "ultima hora",
                "última hora",
                "ultimos 60",
                "últimos 60",
                "última 1 hora",
                "ultima 1 hora",
            )
        ):
            summary = self.statistics.summarize_last_hour()
            return (
                f"En la ultima hora hubo {summary['entradas']} entradas y "
                f"{summary['salidas']} salidas. Actualmente hay "
                f"{summary['ocupacion']} personas y la ocupacion maxima "
                f"registrada fue {summary['ocupacion_maxima']}."
            )

        if any(word in text for word in ("maxima", "máxima", "pico")):
            maximum = self.statistics.maximum_occupancy()
            return f"La ocupacion maxima historica registrada fue {maximum} personas."

        if "entrada" in text:
            if "hoy" in text or "dia" in text or "día" in text:
                total = self.statistics.total_entries(
                    datetime.now().astimezone().replace(
                        hour=0,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )
                )
                return f"Hoy se registraron {total} entradas."
            return f"Se han registrado {self.statistics.total_entries()} entradas."

        if "salida" in text:
            if "hoy" in text or "dia" in text or "día" in text:
                total = self.statistics.total_exits(
                    datetime.now().astimezone().replace(
                        hour=0,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )
                )
                return f"Hoy se registraron {total} salidas."
            return f"Se han registrado {self.statistics.total_exits()} salidas."

        if any(word in text for word in ("cuantas", "cuántas", "ocupacion", "ocupación", "ahora")):
            summary = self.statistics.summarize_today()
            return (
                f"Actualmente hay {summary['ocupacion']} personas. "
                f"Hoy hay {summary['entradas']} entradas y "
                f"{summary['salidas']} salidas."
            )

        if any(word in text for word in ("hoy", "dia", "día", "resumen")):
            summary = self.statistics.summarize_today()
            return (
                f"Hoy: {summary['entradas']} entradas, "
                f"{summary['salidas']} salidas, "
                f"{summary['ocupacion']} personas actualmente y "
                f"{summary['ocupacion_maxima']} como ocupacion maxima."
            )

        return (
            "No entiendo la consulta. Prueba: cuantas personas hay ahora, "
            "entradas hoy, salidas hoy, ocupacion maxima o resumen de la ultima hora."
        )
