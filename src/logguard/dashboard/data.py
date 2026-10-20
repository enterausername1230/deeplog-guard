"""Read-only SQLite adapter and explicitly synthetic demonstration data.

대시보드 데이터 계층 (원작성: 이동준 / 2026-10-20 통합 DB 구조 반영: 옥상민)

읽는 테이블 (구조 정의는 src/logguard/api/schema.sql 한 곳, 설명은 docs/db_schema.md)
    logs            : 백엔드가 쓰는 "로그 1줄 × 블록" 행. 대시보드는 판정된 행(status normal/anomaly)만 읽는다
    session_results : 끝난 세션의 종료(EOS) 판정. 중간에 끊긴 이상 세션은 여기에서만 이상으로 나온다
    alerts          : Slack 알림 이력 (알림 작업자가 씀)

세션 최종 판정 = (logs 에 이상 행이 하나라도 있음) OR (session_results.is_anomaly = 1)

대시보드는 DB 를 "읽기 전용"(mode=ro)으로만 연다 → 백엔드 데이터를 실수로 바꿀 수 없다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# logs 테이블에서 읽을 칸 (화면에 필요한 것만)
COLUMNS = ["id", "timestamp", "block_id", "event_id", "model_name", "anomaly_score",
           "threshold", "is_anomaly", "reason", "details", "raw_log", "latency_ms"]
ALERT_COLUMNS = ["id", "timestamp", "block_id", "model_name", "status", "message"]
# 끝난 세션 판정 (session_results) 에서 읽을 칸
CLOSED_COLUMNS = ["block_id", "model_name", "closed_at", "close_reason", "anomaly_score", "is_anomaly",
                  "reason", "details"]


def read_database(path: Path, limit: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Open an existing DB read-only; never create or modify backend tables.

    Returns:
        (logs, alerts, closed) — 각각 최신 limit 행까지. alerts·session_results 테이블이 없으면 빈 표.
    """
    # path.resolve().as_uri() + "?mode=ro" : 파일을 읽기 전용으로 연다. 파일이 없으면 새로 만들지 않고 오류
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3) as conn:
        # 판정된 행만: 모델 미연결(pending_model)·블록 없음(no_block) 행은 점수가 없어 그래프에 쓸 수 없다
        logs = pd.read_sql_query(
            "SELECT " + ", ".join(COLUMNS) + " FROM logs WHERE status IN ('normal', 'anomaly') "
            "ORDER BY timestamp DESC, id DESC LIMIT ?", conn, params=(limit,))
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        alerts = (pd.read_sql_query("SELECT " + ", ".join(ALERT_COLUMNS) +
                  " FROM alerts ORDER BY timestamp DESC, id DESC LIMIT ?", conn, params=(limit,))
                  if "alerts" in tables else pd.DataFrame(columns=ALERT_COLUMNS))
        closed = (pd.read_sql_query("SELECT " + ", ".join(CLOSED_COLUMNS) +
                  " FROM session_results ORDER BY closed_at DESC LIMIT ?", conn, params=(limit,))
                  if "session_results" in tables else pd.DataFrame(columns=CLOSED_COLUMNS))
    return normalize(logs), alerts, normalize_closed(closed)


def normalize(logs: pd.DataFrame) -> pd.DataFrame:
    """타입 정리 + 기본 검증 (시간은 UTC datetime, 점수는 숫자, is_anomaly 는 bool)."""
    logs = logs.copy()
    logs["timestamp"] = pd.to_datetime(logs["timestamp"], utc=True, errors="raise")
    for column in ("anomaly_score", "threshold", "latency_ms"):
        logs[column] = pd.to_numeric(logs[column], errors="raise")
    if not logs["is_anomaly"].isin([0, 1, False, True]).all():
        raise ValueError("is_anomaly must be 0 or 1")
    if logs[["block_id", "model_name", "timestamp"]].isna().any().any():
        raise ValueError("block_id, model_name and timestamp are required")
    logs["is_anomaly"] = logs["is_anomaly"].astype(bool)
    return logs.sort_values(["timestamp", "id"])


def normalize_closed(closed: pd.DataFrame) -> pd.DataFrame:
    """session_results 타입 정리."""
    closed = closed.copy()
    closed["closed_at"] = pd.to_datetime(closed["closed_at"], utc=True)
    closed["is_anomaly"] = closed["is_anomaly"].astype(bool)
    return closed


def explain(details: str | None, reason: str | None = None) -> str:
    """판정 근거 JSON → 사람이 읽는 한 줄 설명 (대시보드 '판정 근거' 칸).

    모델마다 details 모양이 다르다 (docs/interface_backend.md):
        unknown_event : {"unknown_events": [...], "templates": [...]}
        autoencoder   : {"worst": {"actual": {template}, "reconstructed": {template}}, ...}
        deeplog       : {"actual": {template, prob}, "topk": [{template, prob}, ...]}
    """
    if reason == "unknown_event":
        return "학습에 없던 이벤트 등장"
    if not details:
        return ""
    try:
        d = json.loads(details)
    except (TypeError, ValueError):
        return ""
    if "worst" in d:                                   # Autoencoder: 가장 복원이 안 된 위치
        actual = d["worst"]["actual"].get("template") or "?"
        recon = d["worst"]["reconstructed"].get("template") or "?"
        return f"실제: {actual}  →  모델 예상: {recon}"
    if "actual" in d and "topk" in d:                  # DeepLog: 실제 다음 이벤트 vs 예측 1순위
        top = d["topk"][0].get("template") if d["topk"] else "?"
        return f"실제: {d['actual'].get('template')} (확률 {d['actual'].get('prob', 0):.4f})  →  예상 1순위: {top}"
    return ""


def demo_data(templates: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """화면 확인용 합성 데이터 (실제 모델 결과 아님). 반환: (logs, alerts, closed)."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").floor("min")
    rows, alerts, closed = [], [], []
    template_map = templates.set_index("event_id")["template"].to_dict()
    event_ids = templates["event_id"].to_numpy()
    for session in range(90):
        block = f"blk_demo_{session:04d}"
        anomalous = session % 11 == 0
        start = end - pd.Timedelta(minutes=89 - session)
        for index in range(12):
            timestamp = start + pd.Timedelta(seconds=index * 4)
            event = int(rng.choice(event_ids))
            score = float(rng.uniform(.05, .4))
            if anomalous and index == 8:
                score = float(rng.uniform(.7, .95))
            # 데모용 판정 근거: 실제 Autoencoder details 와 같은 모양
            details = json.dumps({"worst": {"actual": {"template": template_map[event]},
                                            "reconstructed": {"template": template_map[int(event_ids[0])]}}},
                                 ensure_ascii=False)
            rows.append((len(rows) + 1, timestamp.isoformat(), block, event, "Demo Autoencoder",
                         score, .5, int(score > .5), "model" if score > .5 else "normal", details,
                         f"{block} {template_map[event]}", float(rng.uniform(8, 45))))
            if score > .5:
                alerts.append((len(alerts) + 1, timestamp.isoformat(), block, "Demo Autoencoder",
                               "failed" if session == 33 else "sent", "샘플 알림: 이상 윈도우 감지"))
        closed.append((block, "Demo Autoencoder", (start + pd.Timedelta(minutes=1)).isoformat(), "manual",
                       .2, False, "normal", None))
    return (normalize(pd.DataFrame(rows, columns=COLUMNS)), pd.DataFrame(alerts, columns=ALERT_COLUMNS),
            normalize_closed(pd.DataFrame(closed, columns=CLOSED_COLUMNS)))


def session_summary(logs: pd.DataFrame, closed: pd.DataFrame | None = None) -> pd.DataFrame:
    """Session OR: any anomalous window marks the observed session anomalous.

    closed(session_results)가 주어지면 세션 종료 판정도 합친다:
        is_anomaly = (윈도우 중 하나라도 이상) OR (종료 판정 이상)
        is_closed  = 종료 판정을 받았는지 (False 면 아직 진행 중 → 잠정 판정)
    """
    summary = logs.groupby(["model_name", "block_id"], as_index=False).agg(
        first_seen=("timestamp", "min"), last_seen=("timestamp", "max"),
        windows=("id", "size"), peak_score=("anomaly_score", "max"),
        is_anomaly=("is_anomaly", "any"))
    if closed is None or closed.empty:
        summary["is_closed"] = False
        summary["close_anomaly"] = False
        return summary
    c = closed[["model_name", "block_id", "is_anomaly", "close_reason"]].rename(
        columns={"is_anomaly": "close_anomaly"})
    summary = summary.merge(c, on=["model_name", "block_id"], how="left")
    summary["is_closed"] = summary["close_anomaly"].notna()
    summary["close_anomaly"] = summary["close_anomaly"].fillna(False).astype(bool)
    summary["is_anomaly"] = summary["is_anomaly"] | summary["close_anomaly"]
    return summary


def evaluation_results(directory: Path) -> pd.DataFrame:
    """'모델 평가' 탭 데이터.

    6단계 최종 평가 결과(results/final_test.csv)가 있으면 그것을 쓴다
        → Rule-Based, 통계 빈도, DeepLog, Autoencoder 와 EOS 비교까지 test 기준으로 표시 (딥러닝은 시드 3개 평균)
    없으면 예전 방식: results/*.json 중 "test" 지표가 있는 파일
    """
    final = directory / "final_test.csv"
    if final.exists():
        t = pd.read_csv(final)
        return pd.DataFrame({"model": t["method"], "precision": t["test_precision"], "recall": t["test_recall"],
                             "f1": t["test_f1"], "f1_mean": t["test_f1_mean"], "f1_std": t["test_f1_std"],
                             "n_seeds": t["n_seeds"]})
    rows = []
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text())
        metrics = result.get("test", {})
        if all(key in metrics for key in ("precision", "recall", "f1")):
            rows.append({"model": result.get("model", path.stem), **metrics})
    return pd.DataFrame(rows)
