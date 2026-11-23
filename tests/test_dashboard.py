"""Session semantics and read-only DB contract regression checks.

대시보드 테스트 (원작성: 이동준 / 2026-10-20 통합 DB 구조 반영: 옥상민)
    DB 구조는 백엔드와 같은 src/logguard/api/schema.sql 을 쓴다 (예전 dashboard/schema.sql 은 통합되어 삭제).
"""
import sqlite3

import pandas as pd
import pytest

from logguard.config import ROOT
from logguard.dashboard.data import (alerts_since, default_source, demo_data, explain, read_database, session_summary,
                                     trend_frequency)

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


def test_streamlit_demo_filters_and_missing_db(tmp_path):
    """데모 모드 필터 + DB 파일이 없을 때 경고.

    (2026-10-21 수정) 기본 DB 경로가 백엔드 DB(artifacts/api/logguard.sqlite)와 같아져서, start.sh 로 서버를 한 번
    실행한 컴퓨터에서는 기본 경로에 DB 가 존재한다. 그래서 "없는 DB" 를 검사할 때는 기본값에 기대지 않고
    임시 폴더의 존재하지 않는 경로를 직접 넣는다 (팀원 PC 상태와 상관없이 같은 결과).
    """
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
    app.text_input[0].set_value(str(tmp_path / "missing.sqlite3")).run()     # 존재하지 않는 DB 경로
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


def test_slack_link_opens_sqlite_filtered_to_block():
    """Slack 링크(?block=...)로 열면 데모가 아니라 SQLite 모드 + 그 블록 검색 상태로 시작한다."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "src/logguard/dashboard/app.py"), default_timeout=20)
    app.query_params["block"] = "blk_-7"
    app.run()
    assert not app.exception
    assert app.radio[0].value == "SQLite" and app.text_input[1].value == "blk_-7"


def test_trend_frequency_fits_span():
    """점수 추이 그래프 묶음 간격: 짧은 재생(30초)은 1초 단위, 긴 구간은 점이 너무 많지 않게 (2026-11-20 추가)"""
    def span(seconds):
        start = pd.Timestamp("2026-11-20T00:00:00Z")
        return pd.Series([start, start + pd.Timedelta(seconds=seconds)])
    assert trend_frequency(span(30)) == "1s"          # 2만 줄 빠른 시연 ≈ 30초 → 점 30개
    assert trend_frequency(span(3600)) == "30s"       # 1시간 → 점 120개
    assert trend_frequency(span(86400)) == "15min"    # 하루 → 점 96개
    assert trend_frequency(pd.Series([pd.Timestamp("2026-11-20T00:00:00Z")])) == "1min"   # 점 1개면 기본값


def test_default_source_prefers_real_db():
    """처음 고르는 데이터 소스: 환경변수 → Slack 링크 → DB 파일 존재 → 데모 (2026-11-20 추가)"""
    assert default_source("sqlite", "", False) == "SQLite"          # start.sh 는 환경변수로 SQLite 지정
    assert default_source("demo", "blk_1", True) == "데모 데이터"   # 환경변수가 가장 우선
    assert default_source(None, "blk_1", False) == "SQLite"         # Slack 링크로 열림
    assert default_source(None, "", True) == "SQLite"               # 서버를 돌려 DB 가 있음
    assert default_source(None, "", False) == "데모 데이터"         # DB 없이 화면만 볼 때


def test_alerts_after_last_log_are_kept():
    """마지막 로그보다 늦게 만들어진 알림(요약은 최대 20초 늦음)도 조회 기간 안이면 보여야 한다 (2026-11-23 수정)."""
    alerts = pd.DataFrame({"timestamp": ["2026-11-23T05:00:00+00:00", "2026-11-23T05:30:20+00:00", "2026-11-23T04:00:00+00:00"],
                           "status": ["sent", "sent", "sent"]})
    cutoff = pd.Timestamp("2026-11-23T04:30:00Z")          # 조회 시작 (마지막 로그 05:30:00 의 1시간 전)
    kept = alerts_since(alerts, cutoff)
    assert len(kept) == 2                                    # 05:30:20 (마지막 로그 뒤) 포함, 04:00 은 기간 밖
