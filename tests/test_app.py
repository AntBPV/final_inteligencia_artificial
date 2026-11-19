from pathlib import Path


def test_app_file_exists() -> None:
    app = Path("app.py")
    assert app.exists()
    assert "streamlit" in app.read_text(encoding="utf-8")
