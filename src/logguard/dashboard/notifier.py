"""Slack alert worker for the shared API schema. Run with --preview first.

Slack 알림 작업자 (원작성: 이동준 / 2026-11-06 묶음 발송·메시지·쿼리 개선, 2026-11-23 알림 등급·요약 정책: 옥상민)

이 프로그램은 무엇인가?
    백엔드와 따로 도는 프로그램이다. DB 를 주기적으로 확인해서 아직 알리지 않은 이상을 찾아 Slack 으로 보낸다.
    백엔드와 분리했기 때문에 Slack 이 느리거나 실패해도 로그 수신 속도에 영향을 주지 않는다.

알림 정책 (2026-11-23) — "이상은 모두 기록하되, 사람에게는 중요한 것만 바로 알린다"
    예전에는 이상 블록이 나올 때마다 1~2초 간격으로 메시지를 보내 전체 재생에서 수천 건이 쌓였다 (알림 피로).
    🔴 긴급 (즉시, 개별 메시지)
        ① 새 유형 첫 발생 : 이번 실행에서 처음 나온 이상 유형 (configs/event_guide.yaml 의 groups 기준)
        ② 이상 급증       : 최근 60초 이상 블록 수가 그 전 평균의 3배 이상이고 5개 이상 (평소 대비라서 재생 속도와 무관)
    🟠 경고 (요약, 20초마다 1건)
        그 밖의 이상 블록 (이미 알린 유형의 반복 등) → 유형별 개수와 예시만
    ⚪ 정보 — 블록 하나하나의 상세는 Slack 이 아니라 대시보드에서 본다
    메시지 내용: 운영자용 설명·권장 조치(event_guide.yaml), 관련 노드(로그의 IP 로 추정), 블록 흐름 요약, 원본 로그

한 바퀴의 흐름 (Notifier.step, 0.5초마다)
    1) candidates()  : logs·session_results 에서 아직 알리지 않은 이상 블록을 찾는다 (블록·모델당 1건)
    2) 유형 분류      : classify() → 처음 보는 유형이면 긴급, 아니면 경고
    3) alerts 에 pending : 블록마다 먼저 기록한다. UNIQUE(block_id, model_name) 덕분에 같은 블록은 두 번 못 들어간다
    4) 긴급은 바로 전송, 경고는 모아 두었다가 20초마다 요약 전송, 급증이면 급증 경보
    5) 결과 기록      : alerts 를 sent/failed 로 바꾸고 완료 시각을 남긴다
       E2E 지연(목표 ③)은 즉시 보내는 긴급 알림으로 잰다 (message 가 '🔴' 로 시작하는 행, docs/db_schema.md)

실행 (프로젝트 폴더에서)
    make slack-preview                               보낼 내용만 확인 (전송·DB 변경 없음)
    make slack                                       실제 전송 (.env 의 SLACK_WEBHOOK_URL 필요)
    python -m logguard.dashboard.notifier --retry-failed   실패한 알림을 다시 보내기 (직접 실행할 때만)
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote, urlsplit

import requests
import yaml

from logguard.api.settings import load_settings
from logguard.config import ROOT, load_config
from logguard.dashboard.data import explain

KST = timezone(timedelta(hours=9))    # 메시지에 표시할 한국 시간
RAW_LOG_MAX = 300                     # 메시지에 넣을 원본 로그 최대 글자 수 (Slack 메시지가 너무 길어지지 않게)
GUIDE_PATH = ROOT / "configs" / "event_guide.yaml"
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NEW_TYPE = "새 유형 첫 발생"           # 긴급 메시지 첫 줄에 쓰는 말 (재시작 때 이미 알린 유형을 DB 에서 되살리는 표시로도 쓴다)

# 설정 기본값 (configs/config.yaml 의 slack: 에 없으면 이 값)
DEFAULTS = {
    "batch_size": 1000, "timeout_seconds": 5, "send_interval_seconds": 1.1, "poll_seconds": 0.5,
    "digest_seconds": 20, "digest_examples": 3, "dashboard_url": "",
    "spike_window_seconds": 60, "spike_baseline_seconds": 600, "spike_ratio": 3.0,
    "spike_min_blocks": 5, "spike_min_history_seconds": 120, "spike_cooldown_seconds": 300,
}


# 아직 알리지 않은 이상 블록 조회 (설명은 candidates() 참고)
CANDIDATES_SQL = '''
        WITH events AS (
            SELECT l.block_id, l.model_name, l.id AS log_id, l.timestamp AS detected_at,
                   l.anomaly_score, l.reason, l.details, l.raw_log, l.event_id, '실시간' AS source
            FROM logs l
            WHERE l.status = 'anomaly' AND l.block_id IS NOT NULL AND l.model_name IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM alerts a
                              WHERE a.block_id = l.block_id AND a.model_name = l.model_name)
            UNION ALL
            SELECT s.block_id, s.model_name, NULL, s.closed_at, s.anomaly_score, s.reason, s.details,
                   (SELECT raw_log FROM logs WHERE id = s.last_log_id), NULL, '세션 종료'
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
    event_id: 이상이 난 로그의 Drain 템플릿 번호 (유형 분류에 사용, 세션 종료는 NULL)
    """
    return conn.execute(CANDIDATES_SQL, (limit,)).fetchall()


# ── 유형 분류와 메시지 재료 ─────────────────────────────────────────────────────────────────────

def load_guide(path=GUIDE_PATH):
    """configs/event_guide.yaml 읽기. events 의 키를 정수(event_id)로 맞춘다."""
    guide = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    guide["events"] = {int(k): v for k, v in guide.get("events", {}).items()}
    return guide


def classify(row, guide):
    """이상 1건 → 알림 유형(groups 의 키).

    - 세션 종료 판정 이상            → truncated (작업이 중간에 끊긴 블록)
    - 오류 템플릿(group 지정된 로그) → 그 group (예: write_error)
    - 매칭되는 템플릿이 없는 로그    → new_format (처음 보는 형식)
    - 그 밖(정상 형식 로그에서 모델이 이상 판정) → sequence (평소와 다른 작업 순서)
    """
    if row["source"] == "세션 종료":
        return "truncated"
    event = guide["events"].get(row["event_id"]) if row["event_id"] is not None else None
    if event and event.get("group"):
        return event["group"]
    if row["reason"] == "unknown_event" and row["event_id"] is None:
        return "new_format"
    return "sequence"


def block_context(conn, row, guide):
    """블록 흐름 요약과 관련 노드(추정)를 DB 에서 만든다.

    흐름: 그 블록의 이상 시점까지 최근 로그 5개의 단계 이름 (마지막이 이상이 난 로그면 ⚠️ 표시)
    노드: 그 블록 로그에 나온 IP 중 많이 나온 2개. HDFS 로그에는 서버 이름이 없어 IP 로만 추정한다
    """
    if row["log_id"] is not None:
        logs = conn.execute("SELECT event_id, raw_log FROM logs WHERE block_id = ? AND model_name = ? AND id <= ?"
                            " ORDER BY id DESC LIMIT 20", (row["block_id"], row["model_name"], row["log_id"])).fetchall()
    else:
        logs = conn.execute("SELECT event_id, raw_log FROM logs WHERE block_id = ? AND model_name = ?"
                            " ORDER BY id DESC LIMIT 20", (row["block_id"], row["model_name"])).fetchall()
    logs = logs[::-1]                       # 오래된 것 → 최신 순
    steps = [guide["events"].get(e["event_id"], {}).get("step", "처음 보는 로그") if e["event_id"] is not None
             else "처음 보는 로그" for e in logs[-5:]]
    if steps and row["log_id"] is not None:
        steps[-1] = "⚠️ " + steps[-1]
    elif steps:
        steps.append("⛔ 이후 로그 없음")
    ips = Counter(ip for e in logs for ip in set(IP_RE.findall(e["raw_log"] or "")))
    return {"flow": " → ".join(steps), "nodes": [ip for ip, _ in ips.most_common(2)]}


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


def _raw(row):
    raw = row["raw_log"] or ""
    return raw[:RAW_LOG_MAX] + ("…" if len(raw) > RAW_LOG_MAX else "")


def urgent_lines(row, group, ctx, guide, settings):
    """긴급(새 유형 첫 발생) 1건의 평문 메시지 줄. alerts.message 저장·미리보기·Slack 알림 미리보기 문구에 쓴다."""
    g = guide["groups"][group]
    lines = [f"🔴 [긴급] {NEW_TYPE} — {g['name']}",
             f"설명 {g['desc']}",
             f"블록 {row['block_id']} · 관련 노드(추정) {', '.join(ctx['nodes']) or '정보 없음'}"]
    if ctx["flow"]:
        lines.append(f"흐름 {ctx['flow']}")
    if group == "sequence":
        why = explain(row["details"], row["reason"])
        if why:
            lines.append(f"근거 {why}")
    lines += [f"조치 {g['action']}", f"원본 {_raw(row)}",
              f"점수 {row['anomaly_score']:.2f} (기준 0.50) · 모델 {row['model_name']} · {kst(row['detected_at'])}"]
    link = block_link(row, settings)
    return lines + ([f"대시보드: {link}"] if link else [])


def warning_line(row, group, guide):
    """경고(요약으로 보내는) 1건의 평문. alerts.message 에 저장되어 대시보드 알림 이력에 보인다."""
    return (f"🟠 [경고] {guide['groups'][group]['name']} · {row['block_id']} · 점수 {row['anomaly_score']:.2f}"
            f" · {kst(row['detected_at'])} (요약 알림으로 발송)")


def message(row, settings=None, guide=None, conn=None):
    """미리보기(--preview)용: 이 블록이 처음 보는 유형이라면 보낼 긴급 메시지 (유형 상태와 무관하게 상세 형식으로)."""
    guide = guide or load_guide()
    ctx = block_context(conn, row, guide) if conn is not None else {"flow": "", "nodes": []}
    return "\n".join(urgent_lines(row, classify(row, guide), ctx, guide, settings or {}))


# ── Slack 메시지 (Block Kit 카드) ───────────────────────────────────────────────────────────────

def _section(text):
    return {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}


def _link_block(url, label):
    return _section(f"<{url}|{label}>")


def urgent_payload(row, group, ctx, guide, settings):
    """긴급(새 유형 첫 발생) Slack 메시지. text 는 알림 팝업·블록 미지원 환경에서 보이는 대체 문구."""
    g, esc = guide["groups"][group], slack_escape
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"🔴 긴급 · {NEW_TYPE}"}},
        _section(f"*{esc(g['name'])}*\n{esc(g['desc'])}"),
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*블록*\n`{esc(row['block_id'])}`"},
            {"type": "mrkdwn", "text": f"*관련 노드(추정)*\n{esc(', '.join(ctx['nodes']) or '정보 없음')}"},
            {"type": "mrkdwn", "text": f"*점수*\n{row['anomaly_score']:.2f} (기준 0.50)"},
            {"type": "mrkdwn", "text": f"*감지*\n{kst(row['detected_at'])}"},
        ]},
    ]
    if ctx["flow"]:
        blocks.append(_section(f"*흐름*  {esc(ctx['flow'])}"))
    if group == "sequence":
        why = explain(row["details"], row["reason"])
        if why:
            blocks.append(_section(f"*근거*  {esc(why)}"))
    blocks.append(_section(f"*권장 조치*  {esc(g['action'])}"))
    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn", "text": f"원본 `{esc(_raw(row))}` · 모델 {esc(row['model_name'])}"}]})
    link = block_link(row, settings)
    if link:
        blocks.append(_link_block(link, "▶ 대시보드에서 이 블록 보기"))
    text = f"🔴 [긴급] {NEW_TYPE} — {g['name']} · {row['block_id']}"
    return {"text": esc(text), "blocks": blocks}


def digest_payload(alerts, guide, settings, seconds, total):
    """경고 요약 Slack 메시지: 지난 N초 동안 모인 이상 블록을 유형별 개수와 예시로."""
    esc = slack_escape
    counts = Counter(a.group for a in alerts)
    lines = [f"• {esc(guide['groups'][g]['name'])}  *{n}개*" for g, n in counts.most_common()]
    examples = ", ".join(f"`{esc(a.row['block_id'])}`" for a in alerts[:settings["digest_examples"]])
    more = f" 외 {len(alerts) - settings['digest_examples']}개" if len(alerts) > settings["digest_examples"] else ""
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"🟠 경고 요약 · 이상 블록 {len(alerts)}개"}},
        _section("\n".join(lines)),
        {"type": "context", "elements": [{"type": "mrkdwn", "text":
            f"지난 약 {seconds}초 동안 모은 이상 · 이번 실행 누적 {total}개 · 예시 {examples}{more}"}]},
    ]
    if settings.get("dashboard_url"):
        blocks.append(_link_block(settings["dashboard_url"], "▶ 대시보드에서 전체 보기"))
    top = guide["groups"][counts.most_common(1)[0][0]]["name"]
    return {"text": esc(f"🟠 [경고] 이상 블록 {len(alerts)}개 (가장 많은 유형: {top})"), "blocks": blocks}


def spike_payload(recent, base, top_group, guide, settings):
    """이상 급증 긴급 메시지."""
    esc = slack_escape
    w = settings["spike_window_seconds"]
    ratio = f"{recent / base:.1f}배" if base > 0 else "평소 거의 없음"
    top = guide["groups"][top_group]["name"] if top_group else "-"
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "🔴 긴급 · 이상 급증"}},
        _section(f"*최근 {w}초 이상 블록 {recent}개* (그 전 평균 {base:.1f}개 대비 {ratio})\n가장 많은 유형: {esc(top)}"),
        _section("*권장 조치*  클러스터 전체의 네트워크·스토리지 상태를 확인하세요. "
                 "대시보드에서 이상이 특정 노드에 몰리는지 확인하세요."),
    ]
    if settings.get("dashboard_url"):
        blocks.append(_link_block(settings["dashboard_url"], "▶ 대시보드 열기"))
    return {"text": esc(f"🔴 [긴급] 이상 급증 — 최근 {w}초 {recent}개 ({ratio})"), "blocks": blocks}


# ── 알림 작업자 ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    """alerts 에 기록한(pending) 이상 1건과 메시지 재료."""
    alert_id: int
    row: sqlite3.Row
    group: str
    ctx: dict = field(default_factory=dict)
    urgent: bool = False


class Notifier:
    """알림 정책을 적용하는 작업자. 실행 중 상태(이미 알린 유형, 요약 대기 목록, 최근 이상 기록)를 들고 있다.

    send / clock / sleep 을 바꿔 끼울 수 있어서 테스트에서 실제 전송·대기 없이 검사한다.
    """

    def __init__(self, path, url, settings, send=requests.post, clock=time.monotonic, sleep=time.sleep, guide=None):
        self.path, self.url, self.send, self.clock, self.sleep = path, url, send, clock, sleep
        self.cfg = {**DEFAULTS, **(settings or {})}
        self.guide = guide or load_guide()
        self.started = self.last_digest = clock()
        self.pending: list[Alert] = []          # 요약으로 보낼 경고 (alerts 에는 pending 으로 기록됨)
        self.history: deque = deque()           # (감지 시각, 유형) — 급증 판정용
        self.spike_until = 0.0                  # 급증 경보를 다시 보내지 않는 시각 (반복 방지)
        self.last_send = float("-inf")          # 마지막 전송 시각 (Slack 초당 1건 제한을 "보내기 직전" 에만 지킨다)
        self.total = 0                          # 이번 실행에서 기록한 이상 블록 수
        self.seen = self._seen_from_db()        # 이미 긴급으로 알린 유형 (재시작해도 같은 유형을 또 "첫 발생" 으로 보내지 않게)

    def _seen_from_db(self):
        names = {g["name"]: key for key, g in self.guide["groups"].items()}
        prefix = f"🔴 [긴급] {NEW_TYPE} — "
        with connect(self.path) as conn:
            msgs = [m for (m,) in conn.execute("SELECT message FROM alerts WHERE message LIKE ?", (prefix + "%",))]
        return {names[m.split("\n", 1)[0][len(prefix):]] for m in msgs if m.split("\n", 1)[0][len(prefix):] in names}

    def _post(self, payload):
        """Slack 전송. 성공이면 None, 실패면 오류 문구 (주소가 들어간 예외 문구는 저장하지 않는다).
        카드(blocks) 형식을 Slack 이 거부하면(400) 글자만으로 한 번 더 보낸다."""
        # 2026-11-23: 예전에는 보낸 뒤 항상 1.1초를 쉬어서, 요약 직후에 생긴 긴급 알림이 그만큼 늦었다 (E2E 2초 초과).
        #             이제는 다음 전송 직전에 "마지막 전송 후 1.1초" 가 안 됐을 때만 남은 시간만큼 기다린다
        wait = self.last_send + self.cfg["send_interval_seconds"] - self.clock()
        if wait > 0:
            self.sleep(wait)
        try:
            response = self.send(self.url, json=payload, timeout=self.cfg["timeout_seconds"], allow_redirects=False)
            if response.status_code == 400 and "blocks" in payload:
                response = self.send(self.url, json={"text": payload["text"]},
                                     timeout=self.cfg["timeout_seconds"], allow_redirects=False)
            if response.status_code != 200 or response.text.strip() != "ok":
                return f"Slack HTTP {response.status_code} 또는 응답 본문 오류"
        except requests.RequestException:
            return "Slack 연결 실패 또는 응답 시간 초과 (자동 재전송 안 함)"
        finally:
            self.last_send = self.clock()
        return None

    def _finish(self, alerts, error):
        """보낸 결과를 alerts 에 기록하고, 성공이면 logs.alerted = 1."""
        status, done = ("failed" if error else "sent"), now()
        with connect(self.path) as conn:
            conn.executemany("UPDATE alerts SET status=?, completed_at=?, error=? WHERE id=?",
                             [(status, done, error, a.alert_id) for a in alerts])
            if not error:
                conn.executemany("UPDATE logs SET alerted=1 WHERE block_id=? AND model_name=?",
                                 [(a.row["block_id"], a.row["model_name"]) for a in alerts])

    def _spike(self, t):
        """급증 판정: (최근 window 초 이상 블록 수) ≥ max(최소 개수, 배수 × 그 전 baseline 구간의 같은 길이 평균)."""
        c = self.cfg
        w, b = c["spike_window_seconds"], c["spike_baseline_seconds"]
        while self.history and self.history[0][0] < t - w - b:
            self.history.popleft()
        if t - self.started < c["spike_min_history_seconds"] or t < self.spike_until:
            return None
        recent = [g for ts, g in self.history if ts >= t - w]
        older_span = min(b, t - self.started - w)
        if older_span <= 0:
            return None
        base = sum(1 for ts, _ in self.history if t - w - b <= ts < t - w) / older_span * w
        if len(recent) >= c["spike_min_blocks"] and len(recent) >= c["spike_ratio"] * base:
            self.spike_until = t + c["spike_cooldown_seconds"]
            return len(recent), base, Counter(recent).most_common(1)[0][0]
        return None

    def step(self, flush=False):
        """한 바퀴. flush=True 면 요약 대기 중인 경고를 시간과 상관없이 바로 보낸다 (--once, 종료 직전).

        Returns:
            {"sent": 성공한 블록 수, "failed": 실패한 블록 수, "messages": 보낸 Slack 메시지 수}
        """
        t = self.clock()
        result = {"sent": 0, "failed": 0, "messages": 0}
        claimed = []
        with connect(self.path) as conn:
            stamp = now()
            for row in candidates(conn, self.cfg["batch_size"]):
                group = classify(row, self.guide)
                urgent = group not in self.seen
                ctx = block_context(conn, row, self.guide) if urgent else {}
                text = ("\n".join(urgent_lines(row, group, ctx, self.guide, self.cfg)) if urgent
                        else warning_line(row, group, self.guide))
                # Commit the UNIQUE claim before HTTP: concurrent workers cannot both send.
                cursor = conn.execute('''INSERT OR IGNORE INTO alerts
                    (timestamp, block_id, model_name, log_id, status, message)
                    VALUES (?, ?, ?, ?, 'pending', ?)''', (stamp, row["block_id"], row["model_name"], row["log_id"], text))
                if cursor.rowcount:                 # 다른 작업자가 먼저 기록했으면 rowcount 0 → 보내지 않는다
                    if urgent:
                        self.seen.add(group)
                    claimed.append(Alert(cursor.lastrowid, row, group, ctx, urgent))
            conn.commit()                          # 전송 전에 확정: 중간에 죽어도 같은 블록을 두 번 보내지 않는다

        self.total += len(claimed)
        self.history.extend((t, a.group) for a in claimed)

        def record(alerts, error):
            self._finish(alerts, error)
            result["sent" if not error else "failed"] += len(alerts)
            result["messages"] += 1

        for a in (a for a in claimed if a.urgent):            # 🔴 새 유형 첫 발생: 바로 보낸다
            record([a], self._post(urgent_payload(a.row, a.group, a.ctx, self.guide, self.cfg)))
        self.pending += [a for a in claimed if not a.urgent]

        spike = self._spike(t)                                 # 🔴 이상 급증: 블록 기록과 별개인 경보 1건
        if spike:
            error = self._post(spike_payload(*spike, self.guide, self.cfg))
            result["messages"] += 1
            if error:
                print(f"급증 경보 전송 실패: {error}", flush=True)

        due = t - self.last_digest >= self.cfg["digest_seconds"]
        if self.pending and (flush or due):                    # 🟠 경고 요약
            seconds = round(t - self.last_digest)
            record(self.pending, self._post(digest_payload(self.pending, self.guide, self.cfg, seconds, self.total)))
            self.pending, self.last_digest = [], t
        elif not self.pending and due:
            self.last_digest = t                               # 조용한 동안에는 요약 시계를 계속 당겨 둔다
        return result


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
    parser.add_argument("--once", action="store_true", help="쌓인 대상을 한 번 처리(요약 포함)하고 종료")
    parser.add_argument("--retry-failed", action="store_true", help="실패한 알림을 다시 보낼 대상으로 되돌림")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **load_config()["slack"]}
    path = load_settings()["database"]
    if args.preview:
        guide = load_guide()
        with connect(path) as conn:
            rows = candidates(conn, cfg["batch_size"])
            for row in rows:
                print(message(row, cfg, guide, conn), "\n")
            groups = Counter(classify(r, guide) for r in rows)
            print(f"전송하지 않음 · 대상 {len(rows)}건 · 유형 " +
                  ", ".join(f"{guide['groups'][g]['name']} {n}" for g, n in groups.most_common()))
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
    notifier = Notifier(path, url, cfg)
    print("Slack 알림 작업자 시작 · 긴급(새 유형·급증)은 즉시, 경고는 "
          f"{cfg['digest_seconds']}초마다 요약 · 기존 미알림 이상도 처리합니다.", flush=True)
    try:
        while True:
            result = notifier.step(flush=args.once)
            if result["messages"]:             # 보낸 메시지가 있을 때만 결과 출력 (빈 바퀴까지 찍으면 화면이 넘친다)
                print(result, flush=True)
            if args.once:
                break
            time.sleep(cfg["poll_seconds"])
    except KeyboardInterrupt:
        if notifier.pending:                    # 종료 직전 남은 경고 요약을 보내고 끝낸다
            print(notifier.step(flush=True), flush=True)
        print("알림 작업자 종료")


if __name__ == "__main__":
    main()
