import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from device_support import get_device_statuses, resolve_device
from analytics.assistant import OccupancyAssistant
from analytics.statistics import OccupancyStatistics
from counting.line_counter import LineCounter
from storage.repository import CountRepository
from tracking.person_tracker import PersonTracker
from visualization.counting_renderer import draw_counting_overlay
from visualization.tracking_renderer import draw_tracks


st.set_page_config(
    page_title="Conteo de personas",
    page_icon="👥",
    layout="wide",
)

st.title("Sistema de deteccion, seguimiento y ocupacion")
st.caption("Demostracion interactiva basada en YOLO, ByteTrack/BoT-SORT y SQLite.")

with st.sidebar:
    st.header("Configuracion")
    source_option = st.radio(
        "Fuente de video",
        ("Camara web", "Archivo de video"),
        index=1,
    )

    video_path = ""
    if source_option == "Archivo de video":
        video_path = st.text_input(
            "Ruta del video",
            value="resources/people-walking.mp4",
        )
    else:
        camera_index = st.number_input(
            "Indice de camara",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

    confidence = st.slider(
        "Umbral de confianza",
        min_value=0.20,
        max_value=0.80,
        value=0.35,
        step=0.05,
    )
    tracker_name = st.selectbox(
        "Algoritmo",
        (
            "tracking/bytetrack_stable.yaml",
            "tracking/botsort_stable.yaml",
        ),
    )
    confirmation = st.slider(
        "Fotogramas de confirmacion",
        min_value=1,
        max_value=10,
        value=3,
    )
    resolution = st.selectbox(
        "Resolucion de inferencia",
        (640, 512, 416, 320),
        index=2,
    )
    inference_interval = st.slider(
        "Procesar 1 de cada N fotogramas",
        min_value=1,
        max_value=5,
        value=1,
    )
    device_option = st.selectbox(
        "Dispositivo",
        ("auto", "cpu", "cuda", "dml", "openvino"),
    )
    line_x1 = st.slider("Linea X inicial", 0.0, 1.0, 0.0)
    line_y1 = st.slider("Linea Y inicial", 0.0, 1.0, 0.50)
    line_x2 = st.slider("Linea X final", 0.0, 1.0, 1.0)
    line_y2 = st.slider("Linea Y final", 0.0, 1.0, 0.50)
    reverse_direction = st.checkbox("Invertir entrada y salida")
    database_path = st.text_input(
        "Base SQLite",
        value="data/occupancy.sqlite3",
    )

    start_demo = st.button("Iniciar demostracion", type="primary")
    stop_demo = st.button("Detener demostracion")

video_column, metrics_column = st.columns([3, 1])
with video_column:
    video_placeholder = st.empty()
with metrics_column:
    occupancy_metric = st.metric("Ocupacion actual", 0)
    entries_metric = st.metric("Entradas", 0)
    exits_metric = st.metric("Salidas", 0)
    fps_metric = st.metric("FPS", "0")
    recent_placeholder = st.empty()

tabs = st.tabs(["Eventos", "Estadisticas", "Asistente"])
with tabs[0]:
    events_placeholder = st.empty()
with tabs[1]:
    statistics_placeholder = st.empty()
with tabs[2]:
    question = st.text_input("Consulta", key="assistant_question")
    ask_button = st.button("Consultar")
    assistant_answer = st.empty()


if "run_demo" not in st.session_state:
    st.session_state.run_demo = False
if start_demo:
    st.session_state.run_demo = True
if stop_demo:
    st.session_state.run_demo = False


def get_recent_events(repository: CountRepository, limit: int = 20) -> pd.DataFrame:
    rows = repository.connection.execute(
        """
        SELECT occurred_at, track_id, event_type, occupancy
        FROM count_events
        ORDER BY datetime(occurred_at) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=["occurred_at", "track_id", "event_type", "occupancy"],
    )


if ask_button and question:
    repository = CountRepository(database_path)
    assistant = OccupancyAssistant(OccupancyStatistics(repository.connection))
    assistant_answer.info(assistant.ask(question))
    repository.close()


repository = CountRepository(database_path)

today_statistics = OccupancyStatistics(repository.connection)
today = today_statistics.summarize_today()
statistics_placeholder.json(today)
events_placeholder.dataframe(get_recent_events(repository), use_container_width=True)


if st.session_state.run_demo:
    source: str | int = (
        int(camera_index) if source_option == "Camara web" else video_path
    )

    if source_option == "Archivo de video":
        if not Path(video_path).exists():
            video_placeholder.error(f"No existe el video: {video_path}")
            st.session_state.run_demo = False
            repository.close()
            st.stop()

    try:
        resolved_device = resolve_device(device_option)
        tracker = PersonTracker(
            model_name="yolo11n.pt",
            confidence=confidence,
            device=resolved_device,
            tracker=tracker_name,
            trajectory_length=30,
            imgsz=resolution,
        )
    except Exception as error:
        video_placeholder.error(f"No se pudo iniciar el seguimiento: {error}")
        repository.close()
        st.stop()

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        video_placeholder.error("No se pudo abrir la fuente seleccionada.")
        repository.close()
        st.stop()

    counter: LineCounter | None = None
    recent_events: list[str] = []
    last_tracks = []
    inference_frame_index = 0
    session_id = uuid.uuid4().hex
    frame_count = 0
    started_at = time.perf_counter()

    try:
        while st.session_state.run_demo:
            success, frame = capture.read()
            if not success:
                video_placeholder.info("La fuente finalizo.")
                break

            inference_frame_index += 1
            if inference_frame_index % inference_interval == 0:
                last_tracks = tracker.update(frame)
            tracks = last_tracks
            rendered = draw_tracks(frame, tracks, tracker)

            if counter is None:
                height, width = frame.shape[:2]
                counter = LineCounter(
                    frame_width=width,
                    frame_height=height,
                    line_start=(line_x1, line_y1),
                    line_end=(line_x2, line_y2),
                    confirmation_frames=confirmation,
                    reverse_direction=reverse_direction,
                )

            events = counter.update(tracks)
            for event in events:
                label = f"{event.event_type.upper()} - ID {event.track_id}"
                recent_events.append(label)
                repository.save_event(
                    session_id=session_id,
                    track_id=event.track_id,
                    event_type=event.event_type,
                    occupancy=event.occupancy,
                )

            rendered = draw_counting_overlay(
                rendered,
                counter,
                recent_events[-5:],
            )
            rgb_frame = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            occupancy_metric.metric("Ocupacion actual", counter.occupancy)
            entries_metric.metric("Entradas", counter.entries)
            exits_metric.metric("Salidas", counter.exits)

            frame_count += 1
            elapsed = max(time.perf_counter() - started_at, 0.001)
            fps_metric.metric("FPS", f"{frame_count / elapsed:.1f}")
            recent_placeholder.write(
                "\n".join(reversed(recent_events[-5:]))
                if recent_events
                else "Sin eventos recientes"
            )

            events_placeholder.dataframe(
                get_recent_events(repository),
                use_container_width=True,
            )
            statistics_placeholder.json(
                OccupancyStatistics(repository.connection).summarize_today()
            )
    finally:
        capture.release()
        st.session_state.run_demo = False

repository.close()
