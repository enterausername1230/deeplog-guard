"""Run: python -m streamlit run src/logguard/dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

# Support direct Streamlit invocation before editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from logguard.config import ROOT, load_config
from logguard.dashboard.data import demo_data, evaluation_results, read_database, session_summary

st.set_page_config(page_title="DeepLog Guard", page_icon="🛡️", layout="wide")
cfg = load_config()
dcfg = cfg["dashboard"]
st.markdown("""<style>
.block-container {max-width:1500px;padding-top:2rem;}
[data-testid="stMetric"] {border:1px solid #334155;border-radius:12px;padding:16px;}
</style>""", unsafe_allow_html=True)
st.title("🛡️ DeepLog Guard")
st.caption("서버 로그 이상 탐지 · Block ID 세션 모니터링")

with st.sidebar:
    st.header("모니터링 설정")
    source = st.radio("데이터 소스", ["데모 데이터", "SQLite"])
    db_path = st.text_input("SQLite 경로", value=dcfg["database_path"], disabled=source != "SQLite")
    period = st.selectbox("조회 기간", [15, 30, 60, 120, 1440],
                          index=[15, 30, 60, 120, 1440].index(dcfg["default_period_minutes"]),
                          format_func=lambda value: f"최근 {value}분")
    block_search = st.text_input("Block ID 검색", placeholder="blk_...")
    anomalies_only = st.checkbox("이상 세션만 보기")
    auto_refresh = st.toggle("자동 갱신", value=True)
    st.button("지금 새로고침")
    st.caption("실시간 지표는 조회 구간에서 관측한 세션 기준입니다.")


@st.fragment(run_every=dcfg["refresh_seconds"] if auto_refresh else None)
def monitor():
    templates = pd.read_csv(cfg["paths"]["interim_dir"] / "hdfs_templates.csv")
    if source == "데모 데이터":
        logs, alerts = demo_data(templates, cfg["seed"])
        st.info("데모 모드 · 로그, 스코어, 지연, 알림은 합성 데이터입니다. 실제 모델 성능은 ‘모델 평가’에서 확인하세요.")
    else:
        path = Path(db_path)
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            st.warning("SQLite 파일이 없습니다. 데모 모드로 화면을 확인하거나 백엔드 DB 경로를 입력하세요.")
            return
        try:
            logs, alerts = read_database(path, dcfg["max_rows"])
        except Exception as exc:
            st.error(f"DB 조회 실패: {exc}")
            st.caption("필요한 테이블과 필드는 src/logguard/dashboard/schema.sql을 참고하세요.")
            return
        st.caption(f"읽기 전용 SQLite · 최신 {dcfg['max_rows']:,}개 윈도우까지 조회")
        if len(logs) == dcfg["max_rows"]:
            st.warning("조회 상한에 도달했습니다. 세션 이력이 일부 잘릴 수 있습니다.")
    if logs.empty:
        st.info("아직 추론 기록이 없습니다.")
        return

    # Filter relative to the latest observed timestamp so historical replay remains inspectable.
    anchor = logs["timestamp"].max()
    cutoff = anchor - pd.Timedelta(minutes=period)
    models = st.multiselect("모델", sorted(logs["model_name"].unique()),
                            default=sorted(logs["model_name"].unique()), key="models")
    observed = logs[(logs["timestamp"] >= cutoff) & logs["model_name"].isin(models)]
    if block_search:
        observed = observed[observed["block_id"].str.contains(block_search, regex=False, na=False)]
    # Keep OR decisions across loaded history, even when the anomalous window precedes the time filter.
    history = session_summary(logs)
    active_keys = observed[["model_name", "block_id"]].drop_duplicates()
    sessions = history.merge(active_keys, on=["model_name", "block_id"])
    if anomalies_only:
        sessions = sessions[sessions["is_anomaly"]]
        observed = observed.merge(sessions[["model_name", "block_id"]], on=["model_name", "block_id"])
    if not alerts.empty:
        alerts = alerts.copy()
        alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], utc=True)
        alerts = alerts[alerts["timestamp"].between(cutoff, anchor)]
        alerts = alerts.merge(sessions[["model_name", "block_id"]], on=["model_name", "block_id"])

    st.caption(f"기준 시각: {anchor.tz_convert('Asia/Seoul'):%Y-%m-%d %H:%M:%S} KST · 최신 기록 시각 기준으로 조회")
    n_anomaly = int(sessions["is_anomaly"].sum())
    cards = st.columns(4)
    cards[0].metric("관측 세션 · 모델별", f"{len(sessions):,}")
    cards[1].metric("이상 세션", f"{n_anomaly:,}")
    cards[2].metric("이상 세션 비율", f"{n_anomaly / len(sessions):.1%}" if len(sessions) else "—")
    cards[3].metric("Slack 전송 성공", f"{int((alerts['status'] == 'sent').sum()):,}")
    st.caption("같은 Block ID도 모델별로 각각 집계합니다. 세션 내 윈도우 하나라도 이상이면 이상 세션입니다.")

    live, detail, notifications, evaluation = st.tabs(["실시간 모니터링", "세션 상세", "알림 이력", "모델 평가"])
    with live:
        if observed.empty:
            st.info("필터에 맞는 기록이 없습니다.")
        else:
            st.subheader("이상 스코어와 임계치")
            for name, group in observed.groupby("model_name"):
                st.caption(name)
                trend = group.set_index("timestamp")[["anomaly_score", "threshold"]].resample("1min").max()
                st.line_chart(trend, x_label="시간 (브라우저 현지 시각)", y_label="모델 스코어")
            latency = observed["latency_ms"].dropna()
            st.caption(f"관측 윈도우 {len(observed):,}개 · 추론 지연 p95 " +
                       (f"{latency.quantile(.95):.1f} ms" if len(latency) else "측정값 없음"))
            st.subheader("최근 추론 윈도우")
            display = observed.merge(templates[["event_id", "template"]], on="event_id", how="left")
            st.dataframe(display.sort_values("timestamp", ascending=False), hide_index=True, use_container_width=True)
            st.download_button("조회 결과 CSV 다운로드", display.to_csv(index=False).encode("utf-8-sig"),
                               "logguard_predictions.csv", "text/csv")
    with detail:
        st.dataframe(sessions.sort_values("last_seen", ascending=False), hide_index=True, use_container_width=True)
        if not sessions.empty:
            choices = list(sessions[["model_name", "block_id"]].itertuples(index=False, name=None))
            selected = st.selectbox("세션 선택", choices, format_func=lambda item: f"{item[0]} · {item[1]}")
            sequence = logs[(logs["model_name"] == selected[0]) & (logs["block_id"] == selected[1])]
            st.caption("선택한 세션의 로드된 전체 윈도우 이력 · 세션 종료 상태는 현재 제공되지 않습니다.")
            st.dataframe(sequence.merge(templates[["event_id", "template"]], on="event_id", how="left"),
                         hide_index=True, use_container_width=True)
    with notifications:
        st.caption("전송 결과를 읽기 전용으로 표시합니다. 알림 전송과 중복 방지는 백엔드 알림 작업자가 담당합니다.")
        if alerts.empty:
            st.info("조회 구간에 알림 기록이 없습니다.")
        else:
            st.dataframe(alerts.sort_values("timestamp", ascending=False), hide_index=True, use_container_width=True)
    with evaluation:
        st.caption("HDFS test · 최종 평가는 세션 단위 · 실시간 조회 필터와 독립적인 저장된 실험 결과")
        try:
            results = evaluation_results(cfg["paths"]["results_dir"])
        except (ValueError, OSError) as exc:
            st.error(f"평가 결과를 읽을 수 없습니다: {exc}")
            return
        if results.empty:
            st.info("저장된 평가 결과가 없습니다.")
        else:
            st.dataframe(results, hide_index=True, use_container_width=True)
            st.bar_chart(results.set_index("model")[["precision", "recall", "f1"]], horizontal=True)
            st.caption("목표 F1 ≥ 0.85 · Autoencoder와 DeepLog는 결과 파일이 추가되면 표시됩니다.")


monitor()
