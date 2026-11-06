"""Slack 알림 작업자 테스트 (원작성: 이동준 / 2026-11-06 묶음 발송·메시지 테스트 추가: 옥상민).

실제 Slack 으로는 보내지 않는다. send 자리에 가짜 함수를 넣어 보낸 내용을 검사한다.
"""
import sqlite3
from types import SimpleNamespace

import requests

from logguard.config import ROOT
from logguard.dashboard.notifier import (CANDIDATES_SQL, candidates, connect, kst, recover_pending, retry_failed, run_once,
                                         slack_escape, slack_text)


def test_streaming_eos_dedup_and_success(tmp_path):
    path = tmp_path / 'db.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / 'src/logguard/api/schema.sql').read_text())
        for block in ['blk_1', 'blk_2']:
            conn.execute('''INSERT INTO logs (timestamp,log_date,log_time,level,component,
                block_id,event_index,model_name,is_anomaly,anomaly_score,status,raw_log)
                VALUES ('2026-10-22','081109','203615','INFO','test',?,1,'ae',?,.8,?,'sample')''',
                (block, int(block == 'blk_1'), 'anomaly' if block == 'blk_1' else 'normal'))
            conn.execute('''INSERT INTO session_results VALUES
                (?, 'ae', '2026-10-23', 'manual', 1, .9, .5, 1, 'model', NULL)''', (block,))
    sent = []
    def send(url, **kwargs):
        sent.append(kwargs['json'])
        return SimpleNamespace(status_code=200, text='ok')
    cfg = {'batch_size': 20, 'timeout_seconds': 5, 'send_interval_seconds': 0}
    # 이상 블록 2개가 메시지 1건으로 묶여서 나가고, 두 번째 바퀴에는 보낼 것이 없다
    assert run_once(path, 'fake', cfg, send) == {'sent': 2, 'failed': 0, 'messages': 1}
    assert run_once(path, 'fake', cfg, send) == {'sent': 0, 'failed': 0, 'messages': 0}
    assert len(sent) == 1
    assert '이상 블록 2개' in sent[0]['text'] and '세션 종료 판정 1개' in sent[0]['text']
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM alerts WHERE status='sent'").fetchone()[0] == 2
        assert conn.execute("SELECT SUM(alerted) FROM logs").fetchone()[0] == 2


def test_failure_is_recorded_without_secret_and_not_retried(tmp_path):
    path = tmp_path / 'db.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / 'src/logguard/api/schema.sql').read_text())
        conn.execute('''INSERT INTO session_results VALUES
            ('blk_1', 'ae', '2026-10-22', 'manual', 1, .9, .5, 1, 'model', NULL)''')
    def send(*args, **kwargs):
        raise requests.Timeout('secret-webhook-url')
    cfg = {'batch_size': 20, 'timeout_seconds': 5, 'send_interval_seconds': 0}
    assert run_once(path, 'fake', cfg, send)['failed'] == 1
    assert run_once(path, 'fake', cfg, send)['failed'] == 0
    with sqlite3.connect(path) as conn:
        row = conn.execute('SELECT status, completed_at, error FROM alerts').fetchone()
        assert row[0] == 'failed' and row[1] and 'secret' not in row[2]


def _db(tmp_path, rows):
    """테스트용 DB: rows = [(블록, 점수, reason, details, 원본 로그), ...] 를 이상 행으로 저장."""
    path = tmp_path / 'db.sqlite'
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / 'src/logguard/api/schema.sql').read_text())
        for block, score, reason, details, raw in rows:
            conn.execute("""INSERT INTO logs (timestamp,log_date,log_time,level,component,block_id,event_index,
                model_name,is_anomaly,anomaly_score,reason,details,status,raw_log)
                VALUES ('2026-11-06T00:00:00+00:00','081109','203615','WARN','c',?,1,'ae',1,?,?,?,'anomaly',?)""",
                (block, score, reason, details, raw))
    return path


CFG = {'batch_size': 1000, 'timeout_seconds': 5, 'send_interval_seconds': 0, 'critical_score': .9,
       'digest_limit': 3, 'dashboard_url': 'http://dash:8501'}


def test_single_alert_message_has_grade_kst_raw_log_and_link(tmp_path):
    """1건이면 상세 메시지: 긴급 표시, 운영자용 문구, 한국 시간, 원본 로그, 블록 링크."""
    path = _db(tmp_path, [('blk_-7', 1.0, 'unknown_event', None, 'raw line')])
    sent = []
    run_once(path, 'fake', CFG, lambda url, **kw: sent.append(kw['json']) or SimpleNamespace(status_code=200, text='ok'))
    text = sent[0]['text']
    assert '🔴 [긴급] 처음 보는 형태의 로그' in text and '2026-11-06 09:00:00 KST' in text
    assert '원본 raw line' in text and '<http://dash:8501/?block=blk_-7|' in text
    with connect(path) as conn:                  # 대시보드 이력용 메시지는 Slack 문법 없이 저장
        stored = conn.execute('SELECT message FROM alerts').fetchone()[0]
    assert '대시보드: http://dash:8501/?block=blk_-7' in stored


def test_many_alerts_go_in_one_message_critical_first(tmp_path):
    """많이 몰려도 메시지는 1건. 긴급이 먼저 나오고 넘치는 블록은 "외 N개"."""
    path = _db(tmp_path, [(f'blk_{i}', .6, 'model', None, 'r') for i in range(5)] + [('blk_hot', .95, 'model', None, 'r')])
    sent = []
    result = run_once(path, 'fake', CFG, lambda url, **kw: sent.append(kw['json']) or SimpleNamespace(status_code=200, text='ok'))
    assert result == {'sent': 6, 'failed': 0, 'messages': 1} and len(sent) == 1
    lines = sent[0]['text'].splitlines()
    assert '이상 블록 6개' in lines[0] and '긴급 1개' in lines[0]
    assert lines[1].startswith('🔴 blk_hot')                  # 긴급이 맨 위
    assert any('외 3개' in line for line in lines)
    with connect(path) as conn:                               # 블록마다 alerts 1줄씩 (중복 방지·E2E 계산 유지)
        assert conn.execute("SELECT COUNT(*) FROM alerts WHERE status='sent'").fetchone()[0] == 6


def test_log_content_cannot_inject_slack_syntax(tmp_path):
    """로그 안의 <!channel> 이 전체 알림으로 해석되지 않아야 한다."""
    assert slack_escape('<!channel> & <:BLK:>') == '&lt;!channel&gt; &amp; &lt;:BLK:&gt;'
    path = _db(tmp_path, [('blk_1', .6, 'model', None, 'evil <!channel>')])
    with connect(path) as conn:
        text = slack_text(candidates(conn, 10), CFG)
    assert '<!channel>' not in text and '&lt;!channel&gt;' in text


def test_query_uses_status_index(tmp_path):
    """실제 대상 조회 쿼리가 logs 전체를 훑지 않고 status 인덱스를 쓰는지 (계속 도는 쿼리라 중요)."""
    path = _db(tmp_path, [])
    with connect(path) as conn:
        plan = [r[3] for r in conn.execute('EXPLAIN QUERY PLAN ' + CANDIDATES_SQL, (10,))]
    assert any('logs_status_id' in step for step in plan)
    assert not any(step.startswith('SCAN') and ' logs' in f' {step}' and 'USING' not in step for step in plan)


def test_recover_pending_and_retry_failed(tmp_path):
    """전송 중 죽어 pending 으로 남은 알림은 failed 로 표시되고, --retry-failed 뒤에만 다시 대상이 된다."""
    path = _db(tmp_path, [('blk_1', .6, 'model', None, 'r')])
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
