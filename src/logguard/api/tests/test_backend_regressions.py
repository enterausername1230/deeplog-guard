"""백엔드 오류 처리·세션 재판정 회귀 검사. 담당 폴더 안에서 실행한다."""
import sqlite3
import subprocess
import sys
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from logguard.api.main import create_app
from logguard.api.settings import load_settings
from logguard.api.inference import Prediction
from logguard.api.replay import replay

RAW = '081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_123 terminating'

@pytest.fixture
def cfg(tmp_path):
    cfg = load_settings()
    cfg.update(database=tmp_path / 'test.sqlite', model=None, session_idle_lines=None)
    return cfg

class Predictor:
    model_name = 'test'
    def predict(self, events):
        return Prediction(0.1, False)
    def predict_close(self, events):
        return Prediction(0.9 if len(events) == 1 else 0.1, len(events) == 1)


def test_late_log_reopens_session_and_survives_restart(cfg):
    with TestClient(create_app(cfg, Predictor())) as client:
        assert client.post('/logs', json={'raw_log': RAW}).status_code == 201
        assert client.post('/sessions/close').json() == {'closed': 1}
        first = client.get('/sessions').json()[0]
        assert first['is_anomaly'] == 1
        assert client.post('/sessions/close').json() == {'closed': 0}
        assert client.post('/logs', json={'raw_log': RAW}).status_code == 201
    with TestClient(create_app(cfg, Predictor())) as client:
        assert client.post('/sessions/close').json() == {'closed': 1}
        rows = client.get('/sessions').json()
        assert len(rows) == 1 and rows[0]['last_log_id'] > first['last_log_id']
        assert rows[0]['is_anomaly'] == 0
        assert client.post('/sessions/close').json() == {'closed': 0}


def test_close_failure_rolls_back_all_results(cfg):
    class Broken(Predictor):
        calls = 0
        def predict_close(self, events):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('failure')
            return super().predict_close(events)
    with TestClient(create_app(cfg, Broken())) as client:
        for raw in (RAW, RAW.replace('blk_123', 'blk_456')):
            assert client.post('/logs', json={'raw_log': raw}).status_code == 201
        result = client.post('/sessions/close')
        assert result.status_code == 503
        assert client.get('/sessions').json() == []
        assert client.get('/stats').json()['total_records'] == 2
        assert client.post('/sessions/close').json() == {'closed': 2}


@pytest.mark.parametrize('endpoint,method', [('/logs','get'),('/sessions','get'),('/stats','get'),('/sessions/close','post')])
def test_database_failure_returns_503(cfg, monkeypatch, endpoint, method):
    app = create_app(cfg, Predictor())
    with TestClient(app) as client:
        def fail():
            raise sqlite3.OperationalError('private database path')
        monkeypatch.setattr(app.state.db, 'connect', fail)
        response = getattr(client, method)(endpoint)
        assert response.status_code == 503
        assert response.json() == {'detail': 'Database temporarily unavailable'}


@pytest.mark.parametrize('rate,timeout', [(float('nan'),1),(float('inf'),1),(1,float('nan')),(1,float('inf'))])
def test_replay_rejects_nonfinite_rate_or_timeout(rate, timeout):
    with pytest.raises(ValueError):
        replay('unused.log', 'http://localhost/logs', rate, timeout)


def test_collection_import_does_not_require_model_dependencies():
    script = '''
import sys
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'pandas', 'sklearn'}:
            raise ImportError('Model dependency must not load: ' + fullname)
sys.meta_path.insert(0, BlockModels())
from logguard.api.main import create_app
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
