"""Exercise synthetic inputs through the actual Streamlit application."""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_synthetic_ui_calculates_two_valves():
    root=Path(__file__).resolve().parents[1]
    before=Path.cwd()
    try:
        os.chdir(root)
        app=AppTest.from_file(str(root/'app.py'),default_timeout=90).run()
        assert not app.exception
        tanks=next(widget for widget in app.selectbox if widget.label=='탱크 규격')
        assert list(tanks.options)==['가상 소형 탱크','가상 중형 탱크','직접 입력']
        app.button[0].click().run(timeout=90)
        assert not app.exception
        # A simulated pressure-limit failure is an engineering result, not a UI failure.
        assert all('부적합' in error.value for error in app.error)
        assert len(app.get('download_button')) >= 2
    finally:
        os.chdir(before)
