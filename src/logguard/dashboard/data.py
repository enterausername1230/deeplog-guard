"""Read-only SQLite adapter and explicitly synthetic demonstration data."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["id", "timestamp", "block_id", "event_id", "model_name", "anomaly_score",
           "threshold", "is_anomaly", "raw_log", "latency_ms"]
ALERT_COLUMNS = ["id", "timestamp", "block_id", "model_name", "status", "message"]


def read_database(path: Path, limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open an existing DB read-only; never create or modify backend tables."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3) as conn:
        logs = pd.read_sql_query(
            "SELECT " + ", ".join(COLUMNS) + " FROM predictions ORDER BY timestamp DESC, id DESC LIMIT ?",
            conn, params=(limit,))
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        alerts = (pd.read_sql_query("SELECT " + ", ".join(ALERT_COLUMNS) +
                  " FROM alerts ORDER BY timestamp DESC, id DESC LIMIT ?", conn, params=(limit,))
                  if "alerts" in tables else pd.DataFrame(columns=ALERT_COLUMNS))
    return normalize(logs), alerts


def normalize(logs: pd.DataFrame) -> pd.DataFrame:
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


def demo_data(templates: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").floor("min")
    rows, alerts = [], []
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
            rows.append((len(rows) + 1, timestamp.isoformat(), block, event, "Demo Autoencoder",
                         score, .6, int(score > .6), f"{block} {template_map[event]}",
                         float(rng.uniform(8, 45))))
            if score > .6:
                alerts.append((len(alerts) + 1, timestamp.isoformat(), block, "Demo Autoencoder",
                               "failed" if session == 33 else "sent", "샘플 알림: 이상 윈도우 감지"))
    return normalize(pd.DataFrame(rows, columns=COLUMNS)), pd.DataFrame(alerts, columns=ALERT_COLUMNS)


def session_summary(logs: pd.DataFrame) -> pd.DataFrame:
    """Session OR: any anomalous window marks the observed session anomalous."""
    return logs.groupby(["model_name", "block_id"], as_index=False).agg(
        first_seen=("timestamp", "min"), last_seen=("timestamp", "max"),
        windows=("id", "size"), peak_score=("anomaly_score", "max"),
        is_anomaly=("is_anomaly", "any"))


def evaluation_results(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text())
        metrics = result.get("test", {})
        if all(key in metrics for key in ("precision", "recall", "f1")):
            rows.append({"model": result.get("model", path.stem), **metrics})
    return pd.DataFrame(rows)
