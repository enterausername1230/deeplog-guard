"""Slack 알림 작업자 테스트 (원작성: 이동준 / 2026-11-06 묶음 발송, 2026-11-23 알림 등급·요약 정책: 옥상민).

실제 Slack 으로는 보내지 않는다. send 자리에 가짜 함수를, clock 자리에 가짜 시계를 넣어 검사한다.
정책: 🔴 긴급 = 새 유형 첫 발생(즉시) · 이상 급증 / 🟠 경고 = 나머지를 digest_seconds 마다 요약
"""
import json
import sqlite3
from types import SimpleNamespace

import requests

from logguard.config import ROOT
from logguard.dashboard.notifier import (CANDIDATES_SQL, Notifier, block_context, candidates, classify, connect, kst,
                                         load_guide, recover_pending, retry_failed, slack_escape)

GUIDE = load_guide()
CFG = {'batch_size': 1000, 'timeout_seconds': 5, 'send_interval_seconds': 0, 'digest_seconds': 20,
       'digest_examples': 3, 'dashboard_url': 'http://dash:8501',
       'spike_window_seconds': 60, 'spike_baseline_seconds': 600, 'spike_ratio': 3.0, 'spike_min_blocks': 5,
       'spike_min_history_seconds': 120, 'spike_cooldown_seconds': 300}


class Clock:
    """가짜 시계: t 를 직접 옮겨 20초 요약·급증 판정을 기다리지 않고 검사한다."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _db(tmp_path):
    path = tmp_path / 'db.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / 'src/logguard/api/schema.sql').read_text())
    return path


def _log(path, block, event_id, status='anomaly', reason='model', score=.8, raw='raw', details=None):
    """logs 에 1행 추가 (백엔드가 저장하는 모양). status='normal' 이면 흐름용 정상 로그."""
    with sqlite3.connect(path) as conn:
        conn.execute("""INSERT INTO logs (timestamp,log_date,log_time,level,component,block_id,event_id,event_index,
            model_name,is_anomaly,anomaly_score,reason,details,status,raw_log)
            VALUES ('2026-11-23T00:00:00+00:00','081109','203615','INFO','c',?,?,3,'ae',?,?,?,?,?,?)""",
            (block, event_id, int(status == 'anomaly'), score, reason, details, status, raw))


def _sender(sent, status=200, text='ok'):
    def send(url, **kw):
        sent.append(kw['json'])
        return SimpleNamespace(status_code=status, text=text)
    return send


def _row(**kw):
    base = {'source': '실시간', 'event_id': None, 'reason': 'model'}
    return {**base, **kw}


def test_classify_groups():
    """이상 유형 분류: 끊긴 세션 / 오류 템플릿 그룹 / 처음 보는 형식 / 순서 이상"""
    assert classify(_row(source='세션 종료'), GUIDE) == 'truncated'
    assert classify(_row(event_id=16, reason='unknown_event'), GUIDE) == 'write_error'   # IOException 쓰기 실패
    assert classify(_row(event_id=35), GUIDE) == 'delete_error'
    assert classify(_row(event_id=None, reason='unknown_event'), GUIDE) == 'new_format'
    assert classify(_row(event_id=3), GUIDE) == 'sequence'                               # 정상 형식 로그에서 모델 판정


def test_first_of_type_is_urgent_rest_go_to_digest(tmp_path):
    """같은 유형 3개: 첫 번째만 긴급 즉시, 나머지 2개는 20초 뒤 요약 1건. 다시 돌려도 중복 전송 없음."""
    path = _db(tmp_path)
    for b in ('blk_1', 'blk_2', 'blk_3'):
        _log(path, b, 16, reason='unknown_event', score=1.0)
    sent, clock = [], Clock()
    n = Notifier(path, 'fake', CFG, send=_sender(sent), clock=clock, sleep=lambda s: None, guide=GUIDE)
    assert n.step() == {'sent': 1, 'failed': 0, 'messages': 1}
    assert sent[0]['blocks'][0]['text']['text'] == '🔴 긴급 · 새 유형 첫 발생'
    assert '블록 쓰기 중 네트워크·입출력 오류' in json.dumps(sent[0], ensure_ascii=False)
    clock.t = 10
    assert n.step()['messages'] == 0                         # 아직 20초가 안 됨 → 요약 대기
    clock.t = 21
    assert n.step() == {'sent': 2, 'failed': 0, 'messages': 1}
    assert sent[1]['blocks'][0]['text']['text'] == '🟠 경고 요약 · 이상 블록 2개'
    clock.t = 50
    assert n.step()['messages'] == 0 and len(sent) == 2      # 보낼 것 없음
    with connect(path) as conn:
        rows = conn.execute("SELECT status, message FROM alerts ORDER BY id").fetchall()
        assert [r['status'] for r in rows] == ['sent'] * 3
        assert rows[0]['message'].startswith('🔴 [긴급] 새 유형 첫 발생 — 블록 쓰기 중')   # E2E 계산 대상 표시
        assert rows[1]['message'].startswith('🟠 [경고]')
        assert conn.execute("SELECT SUM(alerted) FROM logs").fetchone()[0] == 3


def test_new_type_later_is_urgent_again_and_flush_sends_digest(tmp_path):
    """다른 유형이 나오면 또 긴급. flush=True(--once) 면 요약을 기다리지 않고 바로 보낸다."""
    path = _db(tmp_path)
    _log(path, 'blk_1', 16, reason='unknown_event')
    _log(path, 'blk_2', 16, reason='unknown_event')
    with sqlite3.connect(path) as conn:                       # 세션 종료 판정 이상 (끊긴 세션)
        conn.execute("INSERT INTO session_results VALUES ('blk_9', 'ae', '2026-11-23T00:01:00+00:00', 'manual', 1, .9, .5, 1, 'model', NULL)")
    sent = []
    n = Notifier(path, 'fake', CFG, send=_sender(sent), clock=Clock(), sleep=lambda s: None, guide=GUIDE)
    result = n.step(flush=True)
    assert result == {'sent': 3, 'failed': 0, 'messages': 3}  # 긴급 2 (쓰기 오류, 끊긴 세션) + 요약 1 (blk_2)
    headers = [m['blocks'][0]['text']['text'] for m in sent]
    assert headers.count('🔴 긴급 · 새 유형 첫 발생') == 2 and headers[-1].startswith('🟠 경고 요약')
    assert '작업이 중간에 끊긴 블록' in json.dumps(sent, ensure_ascii=False)


def test_restart_does_not_repeat_first_of_type(tmp_path):
    """작업자를 다시 켜도 이미 긴급으로 알린 유형은 DB 기록으로 기억해 또 '첫 발생' 으로 보내지 않는다."""
    path = _db(tmp_path)
    _log(path, 'blk_1', 16, reason='unknown_event')
    Notifier(path, 'fake', CFG, send=_sender([]), clock=Clock(), sleep=lambda s: None, guide=GUIDE).step()
    _log(path, 'blk_2', 16, reason='unknown_event')
    sent = []
    again = Notifier(path, 'fake', CFG, send=_sender(sent), clock=Clock(), sleep=lambda s: None, guide=GUIDE)
    assert 'write_error' in again.seen
    again.step(flush=True)
    assert len(sent) == 1 and sent[0]['blocks'][0]['text']['text'].startswith('🟠 경고 요약')


def test_spike_alert_after_quiet_period(tmp_path):
    """조용하다가 60초 안에 이상 6개가 몰리면 급증 경보 1건 (쿨다운 동안 반복 없음)."""
    path = _db(tmp_path)
    sent, clock = [], Clock()
    n = Notifier(path, 'fake', CFG, send=_sender(sent), clock=clock, sleep=lambda s: None, guide=GUIDE)
    _log(path, 'blk_0', 3)                                    # t=0: 순서 이상 1개 (첫 발생 긴급)
    n.step()
    clock.t = 300                                             # 5분 조용함
    for i in range(1, 7):
        _log(path, f'blk_{i}', 3)
    n.step()
    texts = [m['text'] for m in sent]
    assert any('이상 급증' in t for t in texts)
    for i in range(7, 14):                                    # 쿨다운 중 또 몰려도 급증 경보는 다시 안 감
        _log(path, f'blk_{i}', 3)
    clock.t = 330
    n.step()
    assert sum('이상 급증' in m['text'] for m in sent) == 1


def test_no_spike_without_history(tmp_path):
    """켠 지 2분이 안 되면(비교할 평소 기록 없음) 이상이 몰려도 급증 경보를 내지 않는다 (30초 시연에서 오경보 방지)."""
    path = _db(tmp_path)
    for i in range(10):
        _log(path, f'blk_{i}', 3)
    sent = []
    Notifier(path, 'fake', CFG, send=_sender(sent), clock=Clock(), sleep=lambda s: None, guide=GUIDE).step(flush=True)
    assert not any('이상 급증' in m['text'] for m in sent)


def test_failure_is_recorded_without_secret_and_not_retried(tmp_path):
    path = _db(tmp_path)
    _log(path, 'blk_1', 35)
    def send(*args, **kwargs):
        raise requests.Timeout('secret-webhook-url')
    n = Notifier(path, 'fake', CFG, send=send, clock=Clock(), sleep=lambda s: None, guide=GUIDE)
    assert n.step()['failed'] == 1
    assert n.step()['failed'] == 0                            # 자동 재전송 없음
    with sqlite3.connect(path) as conn:
        row = conn.execute('SELECT status, completed_at, error FROM alerts').fetchone()
        assert row[0] == 'failed' and row[1] and 'secret' not in row[2]


def test_rejected_blocks_fall_back_to_plain_text(tmp_path):
    """Slack 이 카드 형식을 거부(400)하면 글자만으로 한 번 더 보낸다."""
    path = _db(tmp_path)
    _log(path, 'blk_1', 35)
    calls = []
    def send(url, **kw):
        calls.append(kw['json'])
        return SimpleNamespace(status_code=400 if 'blocks' in kw['json'] else 200,
                               text='invalid_blocks' if 'blocks' in kw['json'] else 'ok')
    n = Notifier(path, 'fake', CFG, send=send, clock=Clock(), sleep=lambda s: None, guide=GUIDE)
    assert n.step() == {'sent': 1, 'failed': 0, 'messages': 1}
    assert 'blocks' in calls[0] and calls[1] == {'text': calls[0]['text']}


def test_block_context_flow_and_nodes(tmp_path):
    """블록 흐름 요약(⚠️ 표시)과 로그 속 IP 로 추정한 관련 노드."""
    path = _db(tmp_path)
    _log(path, 'blk_1', 1, status='normal', raw='Receiving block blk_1 src: /10.0.0.1:5 dest: /10.0.0.2:50010')
    _log(path, 'blk_1', 1, status='normal', raw='Receiving block blk_1 src: /10.0.0.1:5 dest: /10.0.0.3:50010')
    _log(path, 'blk_1', 16, reason='unknown_event', raw='writeBlock blk_1 received exception java.io.IOException')
    with connect(path) as conn:
        row = candidates(conn, 10)[0]
        ctx = block_context(conn, row, GUIDE)
    assert ctx['flow'] == '수신 시작 → 수신 시작 → ⚠️ 쓰기 입출력 오류'
    assert ctx['nodes'][0] == '10.0.0.1' and len(ctx['nodes']) == 2


def test_log_content_cannot_inject_slack_syntax(tmp_path):
    """로그 안의 <!channel> 이 전체 알림으로 해석되지 않아야 한다."""
    assert slack_escape('<!channel> & <:BLK:>') == '&lt;!channel&gt; &amp; &lt;:BLK:&gt;'
    path = _db(tmp_path)
    _log(path, 'blk_1', 35, raw='evil <!channel>')
    sent = []
    Notifier(path, 'fake', CFG, send=_sender(sent), clock=Clock(), sleep=lambda s: None, guide=GUIDE).step()
    payload = json.dumps(sent[0], ensure_ascii=False)
    assert '<!channel>' not in payload and '&lt;!channel&gt;' in payload


def test_query_uses_status_index(tmp_path):
    """실제 대상 조회 쿼리가 logs 전체를 훑지 않고 status 인덱스를 쓰는지 (계속 도는 쿼리라 중요)."""
    path = _db(tmp_path)
    with connect(path) as conn:
        plan = [r[3] for r in conn.execute('EXPLAIN QUERY PLAN ' + CANDIDATES_SQL, (10,))]
    assert any('logs_status_id' in step for step in plan)
    assert not any(step.startswith('SCAN') and ' logs' in f' {step}' and 'USING' not in step for step in plan)


def test_recover_pending_and_retry_failed(tmp_path):
    """전송 중 죽어 pending 으로 남은 알림은 failed 로 표시되고, --retry-failed 뒤에만 다시 대상이 된다."""
    path = _db(tmp_path)
    _log(path, 'blk_1', 3)
    with connect(path) as conn:
        conn.execute("""INSERT INTO alerts (timestamp, block_id, model_name, status, message)
                        VALUES ('t', 'blk_1', 'ae', 'pending', 'm')""")
    assert recover_pending(path) == 1
    with connect(path) as conn:
        assert conn.execute('SELECT status FROM alerts').fetchone()[0] == 'failed'
        assert candidates(conn, 10) == []                    # 자동으로는 다시 보내지 않는다
    assert retry_failed(path) == 1
    with connect(path) as conn:
        assert len(candidates(conn, 10)) == 1


def test_kst_handles_bad_values():
    assert kst('2026-11-06T00:00:00+00:00') == '2026-11-06 09:00:00 KST'
    assert kst('2026-10-22') == '2026-10-22 09:00:00 KST'     # 시간대 없으면 UTC 로 본다
    assert kst('not-a-date') == 'not-a-date'
