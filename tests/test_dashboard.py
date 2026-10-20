"""Session semantics and read-only DB contract regression checks.

대시보드 테스트 (원작성: 이동준 / 2026-10-20 통합 DB 구조 반영: 옥상민)
    DB 구조는 백엔드와 같은 src/logguard/api/schema.sql 을 쓴다 (예전 dashboard/schema.sql 은 통합되어 삭제).
"""
import sqlite3

import pandas as pd
import pytest

from logguard.config import ROOT
from logguard.dashboard.data import demo_data, explain, read_database, session_summary

SCHEMA = ROOT / "src/logguard/api/schema.sql"     # 백엔드와 공유하는 단일 DB 규격


def _insert_log(conn, block, score, anomalous, model="AE"):
    """백엔드가 저장하는 것과 같은 모양의 판정 행 1개."""
    conn.execute(
        "INSERT INTO logs (timestamp, log_date, log_time, level, component, block_id, event_id, event_index,"
        " model_name, anomaly_score, threshold, is_anomaly, reason, details, latency_ms, status, raw_log)"
        " VALUES (?, '081109', '203615', 'INFO', 'dfs.DataNode', ?, 1, 3, ?, ?, .5, ?, ?, NULL, 10, ?, 'raw')",
        ("2026-10-12T08:00:00+00:00", block, model, score, int(anomalous),
         "model" if anomalous else "normal", "anomaly" if anomalous else "normal"))


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
        conn.executescript(SCHEMA.read_text())
        _insert_log(conn, "blk_1", .8, True)
        # 모델 미연결 행(점수 없음)은 대시보드가 읽지 않아야 한다
        conn.execute("INSERT INTO logs (timestamp, log_date, log_time, level, component, block_id, event_index,"
                     " status, raw_log) VALUES ('2026-10-12T08:00:01+00:00', '081109', '203615', 'INFO', 'c',"
                     " 'blk_2', 1, 'pending_model', 'raw')")
    before = path.read_bytes()
    logs, alerts, closed = read_database(path, 100)
    assert len(logs) == 1 and bool(logs.iloc[0]["is_anomaly"])
    assert alerts.empty and closed.empty
    assert path.read_bytes() == before
    with pytest.raises(sqlite3.OperationalError):
        read_database(tmp_path / "missing.sqlite3", 100)
    assert not (tmp_path / "missing.sqlite3").exists()


def test_demo_data_is_explicit_and_has_anomalies():
    templates = pd.read_csv(ROOT / "data/interim/hdfs_templates.csv")
    logs, alerts, closed = demo_data(templates, 42)
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
        conn.executescript(SCHEMA.read_text())
        _insert_log(conn, "blk_live", .8, True)
        conn.execute("INSERT INTO alerts (timestamp, block_id, model_name, status, message)"
                     " VALUES (?, 'blk_live', 'AE', 'sent', 'test')", ("2026-10-12T08:00:00+00:00",))
    app = AppTest.from_file(str(ROOT / "src/logguard/dashboard/app.py"), default_timeout=20).run()
    app.radio[0].set_value("SQLite").run()
    app.text_input[0].set_value(str(path)).run()
    app.multiselect[0].set_value(["AE"]).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert app.metric[1].value == "1"
    assert app.metric[3].value == "1"


def test_session_close_result_makes_truncated_session_anomalous():
    """윈도우 판정은 모두 정상이어도, 세션 종료(EOS) 판정이 이상이면 세션은 이상 (중간에 끊긴 세션)."""
    logs = pd.DataFrame({
        "id": [1, 2], "timestamp": pd.to_datetime(["2026-10-12"] * 2, utc=True),
        "block_id": ["blk_cut", "blk_ok"], "model_name": ["AE", "AE"],
        "anomaly_score": [.1, .1], "is_anomaly": [False, False]})
    closed = pd.DataFrame({"block_id": ["blk_cut"], "model_name": ["AE"], "is_anomaly": [True],
                           "close_reason": ["manual"]})
    result = session_summary(logs, closed).set_index("block_id")
    assert bool(result.loc["blk_cut", "is_anomaly"]) and bool(result.loc["blk_cut", "is_closed"])
    assert not bool(result.loc["blk_ok", "is_anomaly"]) and not bool(result.loc["blk_ok", "is_closed"])


def test_explain_reasons():
    """판정 근거 JSON → 한 줄 설명."""
    ae = '{"worst": {"actual": {"template": "A"}, "reconstructed": {"template": "B"}}}'
    assert explain(ae, "model") == "실제: A  →  모델 예상: B"
    assert explain(None, "unknown_event") == "학습에 없던 이벤트 등장"
    assert explain(None, "normal") == ""
