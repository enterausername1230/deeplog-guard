"""Session semantics and read-only DB contract regression checks."""
import sqlite3

import pandas as pd
import pytest

from logguard.config import ROOT
from logguard.dashboard.data import demo_data, read_database, session_summary


def test_session_or_and_model_isolation():
    logs = pd.DataFrame({
        "id": [1, 2, 3], "timestamp": pd.to_datetime(["2026-10-12"] * 3, utc=True),
        "block_id": ["blk_1"] * 3, "model_name": ["AE", "AE", "LSTM"],
        "anomaly_score": [.8, .1, .2], "is_anomaly": [True, False, False],
    })
    result = session_summary(logs).set_index("model_name")
    assert bool(result.loc["AE", "is_anomaly"])
    assert not bool(result.loc["LSTM", "is_anomaly"])
    assert result.loc["AE", "windows"] == 2


def test_sqlite_contract_and_missing_db(tmp_path):
    path = tmp_path / "logs.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "src/logguard/dashboard/schema.sql").read_text())
        conn.execute("INSERT INTO predictions VALUES (1, ?, 'blk_1', 1, 'AE', .8, .6, 1, 'raw', 10)",
                     ("2026-10-12T08:00:00+00:00",))
    before = path.read_bytes()
    logs, alerts = read_database(path, 100)
    assert bool(logs.iloc[0]["is_anomaly"])
    assert alerts.empty
    assert path.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError):
        read_database(tmp_path / "missing.sqlite3", 100)
    assert not (tmp_path / "missing.sqlite3").exists()


def test_demo_data_is_explicit_and_has_anomalies():
    templates = pd.read_csv(ROOT / "data/interim/hdfs_templates.csv")
    logs, alerts = demo_data(templates, 42)
    assert len(session_summary(logs)) == 90
    assert session_summary(logs)["is_anomaly"].sum() == 9
    assert set(alerts["status"]) == {"sent", "failed"}
    assert logs["model_name"].str.startswith("Demo").all()


def test_streamlit_demo_filters_and_missing_db():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src/logguard/dashboard/app.py"), default_timeout=20).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.checkbox[0].check().run()
    assert not app.exception
    assert app.metric[0].value == app.metric[1].value
    app.text_input[1].set_value("nonexistent-block").run()
    assert not app.exception
    assert app.metric[0].value == "0"
    app.radio[0].set_value("SQLite").run()
    assert not app.exception
    assert any("SQLite 파일이 없습니다" in item.value for item in app.warning)


def test_streamlit_sqlite_mode(tmp_path):
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "live.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "src/logguard/dashboard/schema.sql").read_text())
        conn.execute("INSERT INTO predictions VALUES (1, ?, 'blk_live', 1, 'AE', .8, .6, 1, 'raw', 10)",
                     ("2026-10-12T08:00:00+00:00",))
        conn.execute("INSERT INTO alerts VALUES (1, ?, 'blk_live', 'AE', 'sent', 'test')",
                     ("2026-10-12T08:00:00+00:00",))
    app = AppTest.from_file(str(ROOT / "src/logguard/dashboard/app.py"), default_timeout=20).run()
    app.radio[0].set_value("SQLite").run()
    app.text_input[0].set_value(str(path)).run()
    app.multiselect[0].set_value(["AE"]).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert app.metric[1].value == "1"
    assert app.metric[3].value == "1"
