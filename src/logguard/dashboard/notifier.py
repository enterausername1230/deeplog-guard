"""Slack alert worker for the shared API schema. Run with --preview first."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit

import requests

from logguard.api.settings import load_settings
from logguard.config import ROOT, load_config
from logguard.dashboard.data import explain


def now():
    return datetime.now(timezone.utc).isoformat()


def webhook_url():
    """Read just SLACK_WEBHOOK_URL; never execute the contents of .env."""
    value = os.environ.get("SLACK_WEBHOOK_URL", "")
    env_file = ROOT / ".env"
    if not value and env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, sep, content = line.strip().partition("=")
            if sep and key == "SLACK_WEBHOOK_URL":
                value = content.strip().strip("\"'")
                break
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.netloc not in {"hooks.slack.com", "hooks.slack-gov.com"}
            or not parsed.path.startswith("/services/") or parsed.query or parsed.fragment):
        raise ValueError("프로젝트 .env에 유효한 SLACK_WEBHOOK_URL을 설정하세요.")
    return value


@contextmanager
def connect(path):
    # mode=rw rejects missing DB instead of silently creating an empty one.
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def candidates(conn, limit):
    return conn.execute('''
        WITH events AS (
            SELECT block_id, model_name, id AS log_id, timestamp AS detected_at,
                   anomaly_score, reason, details, '실시간' AS source
            FROM logs WHERE is_anomaly = 1 AND block_id IS NOT NULL AND model_name IS NOT NULL
            UNION ALL
            SELECT block_id, model_name, NULL, closed_at, anomaly_score, reason, details, '세션 종료'
            FROM session_results WHERE is_anomaly = 1
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY block_id, model_name ORDER BY detected_at, log_id) AS rank
            FROM events
        )
        SELECT r.* FROM ranked r
        WHERE rank = 1 AND NOT EXISTS (
            SELECT 1 FROM alerts a WHERE a.block_id=r.block_id AND a.model_name=r.model_name
        ) ORDER BY detected_at LIMIT ?
    ''', (limit,)).fetchall()


def message(row):
    explanation = explain(row["details"], row["reason"])
    return (f"[DeepLog-Guard 이상 감지] {row['source']}\n"
            f"세션: {row['block_id']} | 모델: {row['model_name']}\n"
            f"점수: {row['anomaly_score']:.3f} | 감지: {row['detected_at']}\n"
            f"{explanation}".strip())


def run_once(path, url, settings, send=requests.post):
    sent = failed = 0
    with connect(path) as conn:
        rows = candidates(conn, settings["batch_size"])
        for row in rows:
            body = message(row)
            # Commit the UNIQUE claim before HTTP: concurrent workers cannot both send.
            cursor = conn.execute('''INSERT OR IGNORE INTO alerts
                (timestamp, block_id, model_name, log_id, status, message)
                VALUES (?, ?, ?, ?, 'pending', ?)''',
                (now(), row['block_id'], row['model_name'], row['log_id'], body))
            conn.commit()
            if not cursor.rowcount:
                continue
            alert_id = cursor.lastrowid
            error = None
            try:
                response = send(url, json={"text": body, "mrkdwn": False},
                                timeout=settings["timeout_seconds"], allow_redirects=False)
                if response.status_code != 200 or response.text.strip() != "ok":
                    # Never persist URL-bearing exception text or arbitrary Slack response content.
                    error = f"Slack HTTP {response.status_code} 또는 응답 본문 오류"
            except requests.RequestException:
                error = "Slack 연결 실패 또는 응답 시간 초과 (자동 재전송 안 함)"
            status = "failed" if error else "sent"
            conn.execute("UPDATE alerts SET status=?, completed_at=?, error=? WHERE id=?",
                         (status, now(), error, alert_id))
            if not error:
                conn.execute("UPDATE logs SET alerted=1 WHERE block_id=? AND model_name=?",
                             (row['block_id'], row['model_name']))
            conn.commit()
            failed += bool(error)
            sent += not bool(error)
            time.sleep(settings["send_interval_seconds"])
    return {"sent": sent, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="전송·DB 변경 없이 대상 메시지 표시")
    parser.add_argument("--once", action="store_true", help="한 배치만 처리 후 종료")
    args = parser.parse_args()
    cfg = load_config()["slack"]
    path = load_settings()["database"]
    if args.preview:
        with connect(path) as conn:
            rows = candidates(conn, cfg["batch_size"])
            for row in rows:
                print(message(row), "\n")
            print(f"전송하지 않음 · 대상 {len(rows)}건 (최대 {cfg['batch_size']}건)")
        return
    try:
        url = webhook_url()
    except ValueError as exc:
        parser.error(str(exc))
    print("Slack 알림 작업자 시작 · 기존 미알림 이상 세션도 처리합니다.", flush=True)
    try:
        while True:
            print(run_once(path, url, cfg), flush=True)
            if args.once:
                break
            time.sleep(cfg["poll_seconds"])
    except KeyboardInterrupt:
        print("알림 작업자 종료")


if __name__ == "__main__":
    main()
