"""Slack alert worker for the shared API schema. Run with --preview first.

Slack 알림 작업자 (원작성: 이동준 / 2026-11-06 묶음 발송·메시지·쿼리 개선: 옥상민)

이 프로그램은 무엇인가?
    백엔드와 따로 도는 프로그램이다. DB 를 주기적으로 확인해서 아직 알리지 않은 이상을 찾아 Slack 으로 보낸다.
    백엔드와 분리했기 때문에 Slack 이 느리거나 실패해도 로그 수신 속도에 영향을 주지 않는다.

한 바퀴의 흐름 (run_once)
    1) candidates()      : logs·session_results 에서 아직 알리지 않은 이상 블록을 찾는다
    2) alerts 에 pending : 블록마다 한 줄씩 먼저 기록한다. UNIQUE(block_id, model_name) 덕분에 같은 블록은 두 번 못 들어간다
    3) Slack 전송        : 쌓인 이상을 "메시지 1건" 에 담아 보낸다 (1건이면 상세, 여러 건이면 요약)
    4) 결과 기록         : alerts 를 sent/failed 로 바꾸고 완료 시각을 남긴다 (E2E 지연 계산에 사용)

왜 묶어서 보내나? (2026-11-06 변경)
    Slack Webhook 은 초당 약 1건까지만 받는다. 예전처럼 블록마다 1건씩 보내면 앞 알림이 끝날 때까지 뒤 알림이 줄을 서서
    몰리는 구간에서 2초를 넘겼고(시뮬레이션 p95 4.6초), 세션 종료 판정 약 600건을 보내는 데 약 11분이 걸렸다.
    이제는 전송 차례가 오면 그때까지 쌓인 이상을 전부 한 메시지에 담는다 → 뒤에 줄 서는 알림이 생기지 않는다.

실행 (프로젝트 폴더에서)
    make slack-preview                               보낼 내용만 확인 (전송·DB 변경 없음)
    make slack                                       실제 전송 (.env 의 SLACK_WEBHOOK_URL 필요)
    python -m logguard.dashboard.notifier --retry-failed   실패한 알림을 다시 보내기 (직접 실행할 때만)
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote, urlsplit

import requests

from logguard.api.settings import load_settings
from logguard.config import ROOT, load_config
from logguard.dashboard.data import explain

KST = timezone(timedelta(hours=9))    # 메시지에 표시할 한국 시간
RAW_LOG_MAX = 300                     # 메시지에 넣을 원본 로그 최대 글자 수 (Slack 메시지가 너무 길어지지 않게)

# 판정 유형 → 운영자가 읽는 문구 (모델 용어를 그대로 보여 주지 않는다)
TYPE_UNKNOWN = "처음 보는 형태의 로그 (학습 데이터에 없음)"
TYPE_SEQUENCE = "평소와 다른 이벤트 순서"
TYPE_TRUNCATED = "작업이 중간에 끊긴 블록"


# 아직 알리지 않은 이상 블록 조회 (설명은 candidates() 참고)
CANDIDATES_SQL = '''
        WITH events AS (
            SELECT l.block_id, l.model_name, l.id AS log_id, l.timestamp AS detected_at,
                   l.anomaly_score, l.reason, l.details, l.raw_log, '실시간' AS source
            FROM logs l
            WHERE l.status = 'anomaly' AND l.block_id IS NOT NULL AND l.model_name IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM alerts a
                              WHERE a.block_id = l.block_id AND a.model_name = l.model_name)
            UNION ALL
            SELECT s.block_id, s.model_name, NULL, s.closed_at, s.anomaly_score, s.reason, s.details,
                   (SELECT raw_log FROM logs WHERE id = s.last_log_id), '세션 종료'
            FROM session_results s
            WHERE s.is_anomaly = 1
              AND NOT EXISTS (SELECT 1 FROM alerts a
                              WHERE a.block_id = s.block_id AND a.model_name = s.model_name)
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY block_id, model_name ORDER BY detected_at, log_id) AS rank
            FROM events
        )
        SELECT * FROM ranked WHERE rank = 1 ORDER BY detected_at, log_id LIMIT ?
'''


def now():
    """현재 시각 (UTC, ISO-8601). DB 에 저장하는 시각은 모두 이 형식이다."""
    return datetime.now(timezone.utc).isoformat()


def webhook_url():
    """Read just SLACK_WEBHOOK_URL; never execute the contents of .env.

    환경변수를 먼저 보고, 없으면 프로젝트의 .env 에서 그 한 줄만 읽는다.
    Slack 공식 주소 형식이 아니면 거부해서, 잘못된 주소로 로그 내용이 나가는 일을 막는다.
    """
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
    """DB 연결. mode=rw: 파일이 없으면 빈 DB 를 만들지 않고 오류를 낸다 (백엔드가 먼저 켜져야 한다)."""
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row         # row["칸이름"] 으로 꺼낼 수 있게
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def candidates(conn, limit):
    """아직 알리지 않은 이상 블록을 블록·모델당 1건씩 찾는다 (가장 먼저 감지된 이상 기준).

    성능 (2026-11-06 변경): 이 쿼리는 몇백 ms 간격으로 계속 돈다.
        is_anomaly = 1 조건은 인덱스가 없어 logs 전체를 훑었다 (223만 행에서 1회 약 375ms).
        status = 'anomaly' 로 바꾸면 logs_status_id 인덱스를 쓴다. 백엔드는 이상일 때만 이 값을 저장하므로 뜻은 같다.
        이미 알린 블록은 줄을 세우기(ROW_NUMBER) 전에 빼서 계산량도 줄였다.

    raw_log: 메시지에 넣을 원본 로그. 세션 종료 판정은 그 블록의 마지막 로그(last_log_id)를 보여 준다.
    """
    return conn.execute(CANDIDATES_SQL, (limit,)).fetchall()


def is_critical(row, settings):
    """긴급 판단: 학습에 없던 이벤트이거나 점수가 기준(기본 0.9) 이상이면 긴급."""
    return row["reason"] == "unknown_event" or row["anomaly_score"] >= settings.get("critical_score", 0.9)


def type_label(row):
    """판정 유형 → 운영자용 문구."""
    if row["reason"] == "unknown_event":
        return TYPE_UNKNOWN
    return TYPE_TRUNCATED if row["source"] == "세션 종료" else TYPE_SEQUENCE


def kst(value):
    """UTC ISO 문자열 → 한국 시간 표시. 해석할 수 없는 값이면 원래 문자열을 그대로 보여 준다."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return str(value)
    if parsed.tzinfo is None:              # 시간대 표시가 없으면 UTC 로 본다 (DB 저장 규칙)
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def slack_escape(text):
    """Slack 메시지 문법 무력화.

    로그 내용에는 <:BLK:> 같은 기호가 들어 있고, 외부에서 들어온 값이라 믿을 수 없다.
    Slack 은 <!channel> 같은 표기를 전체 알림으로 해석하므로 &, <, > 를 Slack 규칙대로 바꿔서 글자로만 보이게 한다.
    """
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def block_link(row, settings):
    """대시보드 주소 + ?block=블록ID. 누르면 대시보드가 그 블록으로 걸러진 상태로 열린다. 주소 설정이 없으면 빈 문자열."""
    base = settings.get("dashboard_url")
    return f"{base.rstrip('/')}/?block={quote(row['block_id'], safe='')}" if base else ""


def detail_lines(row, settings, esc=str):
    """블록 1건의 상세 내용 (줄 목록). esc: Slack 으로 보낼 때는 slack_escape, DB 저장용은 그대로."""
    icon, grade = ("🔴", "긴급") if is_critical(row, settings) else ("🟠", "경고")
    lines = [f"{icon} [{grade}] {type_label(row)} · {esc(row['block_id'])}",
             f"점수 {row['anomaly_score']:.2f} (기준 0.50) · 모델 {esc(row['model_name'])} · {kst(row['detected_at'])}"]
    # unknown_event 의 근거 문구는 위 판정 유형과 같은 말이라 생략한다
    explanation = explain(row["details"], row["reason"]) if row["reason"] != "unknown_event" else ""
    if explanation:
        lines.append(f"근거 {esc(explanation)}")
    if row["raw_log"]:
        raw = row["raw_log"]
        lines.append(f"원본 {esc(raw[:RAW_LOG_MAX] + ('…' if len(raw) > RAW_LOG_MAX else ''))}")
    return lines


def message(row, settings=None):
    """alerts.message 에 저장하고 --preview 로 보여 주는 블록 1건 평문 메시지 (대시보드 알림 이력에 표시됨)."""
    settings = settings or {}
    link = block_link(row, settings)
    return "\n".join(detail_lines(row, settings) + ([f"대시보드: {link}"] if link else []))


def slack_text(rows, settings):
    """Slack 으로 보낼 메시지 1건. 1건이면 상세, 여러 건이면 요약 + 상위 목록."""
    limit = settings.get("digest_limit", 10)
    if len(rows) == 1:
        link = block_link(rows[0], settings)
        return "\n".join(detail_lines(rows[0], settings, slack_escape)
                         + ([f"<{link}|▶ 대시보드에서 보기>"] if link else []))
    # 긴급을 먼저, 같은 등급이면 점수가 높은 순으로 보여 준다
    ordered = sorted(rows, key=lambda r: (not is_critical(r, settings), -r["anomaly_score"]))
    critical = sum(is_critical(r, settings) for r in rows)
    closed = sum(r["source"] == "세션 종료" for r in rows)
    lines = [f"🚨 이상 블록 {len(rows)}개 감지 · 긴급 {critical}개 · 세션 종료 판정 {closed}개"]
    for row in ordered[:limit]:
        icon = "🔴" if is_critical(row, settings) else "🟠"
        lines.append(f"{icon} {slack_escape(row['block_id'])} · {row['anomaly_score']:.2f} · {type_label(row)}")
    if len(rows) > limit:
        lines.append(f"… 외 {len(rows) - limit}개")
    if settings.get("dashboard_url"):
        lines.append(f"<{settings['dashboard_url']}|▶ 대시보드에서 전체 보기>")
    return "\n".join(lines)


def run_once(path, url, settings, send=requests.post):
    """한 바퀴: 쌓인 이상을 찾아 메시지 1건으로 보내고 결과를 기록한다.

    Returns:
        {"sent": 성공한 블록 수, "failed": 실패한 블록 수, "messages": 보낸 Slack 메시지 수}
    """
    with connect(path) as conn:
        rows = candidates(conn, settings["batch_size"])
        claimed, alert_ids = [], []
        stamp = now()
        for row in rows:
            # Commit the UNIQUE claim before HTTP: concurrent workers cannot both send.
            cursor = conn.execute('''INSERT OR IGNORE INTO alerts
                (timestamp, block_id, model_name, log_id, status, message)
                VALUES (?, ?, ?, ?, 'pending', ?)''',
                (stamp, row["block_id"], row["model_name"], row["log_id"], message(row, settings)))
            if cursor.rowcount:                # 다른 작업자가 먼저 기록했으면 rowcount 0 → 보내지 않는다
                claimed.append(row)
                alert_ids.append(cursor.lastrowid)
        conn.commit()                          # 전송 전에 확정: 중간에 죽어도 같은 블록을 두 번 보내지 않는다
        if not claimed:
            return {"sent": 0, "failed": 0, "messages": 0}

        error = None
        try:
            response = send(url, json={"text": slack_text(claimed, settings), "mrkdwn": True},
                            timeout=settings["timeout_seconds"], allow_redirects=False)
            if response.status_code != 200 or response.text.strip() != "ok":
                # Never persist URL-bearing exception text or arbitrary Slack response content.
                error = f"Slack HTTP {response.status_code} 또는 응답 본문 오류"
        except requests.RequestException:
            error = "Slack 연결 실패 또는 응답 시간 초과 (자동 재전송 안 함)"

        status, done = ("failed" if error else "sent"), now()
        conn.executemany("UPDATE alerts SET status=?, completed_at=?, error=? WHERE id=?",
                         [(status, done, error, alert_id) for alert_id in alert_ids])
        if not error:
            conn.executemany("UPDATE logs SET alerted=1 WHERE block_id=? AND model_name=?",
                             [(r["block_id"], r["model_name"]) for r in claimed])
        conn.commit()
    # Slack Webhook 제한(초당 약 1건)을 넘지 않도록 다음 메시지까지 쉰다. 이 사이에 쌓인 이상은 다음 메시지 1건에 함께 담긴다
    time.sleep(settings["send_interval_seconds"])
    count = len(claimed)
    return {"sent": 0 if error else count, "failed": count if error else 0, "messages": 1}


def recover_pending(path):
    """시작할 때 1회: 전송 중 프로그램이 죽어 pending 으로 남은 알림을 failed 로 표시한다.

    Slack 에 실제로 갔는지 알 수 없어서 자동으로 다시 보내지는 않는다 (중복 방지).
    대시보드 알림 이력에 보이도록 표시만 하고, 필요하면 --retry-failed 로 직접 다시 보낸다.
    """
    with connect(path) as conn:
        cursor = conn.execute("UPDATE alerts SET status='failed', completed_at=?, error=? WHERE status='pending'",
                              (now(), "작업자가 전송 중 종료되어 결과를 알 수 없음"))
        return cursor.rowcount


def retry_failed(path):
    """--retry-failed: 실패한 알림 기록을 지운다 → 그 블록들이 다음 바퀴에 다시 전송 대상이 된다.

    직접 실행할 때만 동작한다. 재전송 건은 완료 시각이 늦어지므로 E2E 지연 측정에서는 새 DB 로 잰다.
    """
    with connect(path) as conn:
        return conn.execute("DELETE FROM alerts WHERE status='failed'").rowcount


def main():
    """명령줄 진입점. --preview / --once / --retry-failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="전송·DB 변경 없이 대상 메시지 표시")
    parser.add_argument("--once", action="store_true", help="한 배치만 처리 후 종료")
    parser.add_argument("--retry-failed", action="store_true", help="실패한 알림을 다시 보낼 대상으로 되돌림")
    args = parser.parse_args()
    cfg = load_config()["slack"]
    path = load_settings()["database"]
    if args.preview:
        with connect(path) as conn:
            rows = candidates(conn, cfg["batch_size"])
            for row in rows:
                print(message(row, cfg), "\n")
            print(f"전송하지 않음 · 대상 {len(rows)}건 (최대 {cfg['batch_size']}건)")
        return
    try:
        url = webhook_url()
    except ValueError as exc:
        parser.error(str(exc))
    recovered = recover_pending(path)
    if recovered:
        print(f"이전 실행에서 결과를 알 수 없는 알림 {recovered}건을 failed 로 표시했습니다.", flush=True)
    if args.retry_failed:
        print(f"실패한 알림 {retry_failed(path)}건을 다시 보냅니다.", flush=True)
    print("Slack 알림 작업자 시작 · 기존 미알림 이상 세션도 처리합니다.", flush=True)
    try:
        while True:
            result = run_once(path, url, cfg)
            if result["messages"]:             # 보낸 메시지가 있을 때만 결과 출력 (빈 바퀴까지 찍으면 화면이 넘친다)
                print(result, flush=True)
            if args.once:
                break
            if not result["messages"]:         # 보냈다면 run_once 안에서 이미 쉬었으므로 바로 다음 바퀴로 간다
                time.sleep(cfg["poll_seconds"])
    except KeyboardInterrupt:
        print("알림 작업자 종료")


if __name__ == "__main__":
    main()
