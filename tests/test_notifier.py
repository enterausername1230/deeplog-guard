import sqlite3
from types import SimpleNamespace

import requests

from logguard.config import ROOT
from logguard.dashboard.notifier import run_once


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
    assert run_once(path, 'fake', cfg, send) == {'sent': 2, 'failed': 0}
    assert run_once(path, 'fake', cfg, send) == {'sent': 0, 'failed': 0}
    assert len(sent) == 2
    assert any('세션 종료' in item['text'] for item in sent)
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
