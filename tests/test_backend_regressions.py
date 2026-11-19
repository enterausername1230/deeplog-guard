"""백엔드 오류 처리·세션 재판정 회귀 검사 (원작성: 안명근 2026-11-10, fix/backend-session-recheck).

"회귀 검사" = 한 번 고친 버그가 나중에 다시 생기지 않는지 지키는 테스트.
2026-11-19 merge 때 src/logguard/api/tests/ 에서 이 폴더(tests/)로 옮겼다 → `make test` 에 함께 실행된다.

검사하는 것
  1. 종료 판정 후 같은 블록에 로그가 더 들어오면 다음 종료 검사에서 다시 판정하고 결과를 갱신하는가 (서버 재시작 후에도)
  2. 세션 종료 판정 도중 모델이 실패하면 그 요청에서 저장한 결과를 전부 취소(롤백)하는가
  3. DB 오류(잠김 등)가 나면 내부 경로를 숨긴 503 응답을 돌려주는가
  4. 로그 재생 모듈이 속도·타임아웃의 NaN·무한대 값을 거부하는가
  5. 모델 없이 수집만 하는 모드(model: null)에서 torch·numpy 같은 모델 패키지를 불러오지 않는가
"""
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from logguard.api.main import create_app
from logguard.api.settings import load_settings
from logguard.api.inference import Prediction
from logguard.api.replay import replay

# 테스트용 HDFS 로그 한 줄 (블록 blk_123 의 정상 종료 이벤트)
RAW = '081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_123 terminating'


@pytest.fixture
def cfg(tmp_path):
    """실제 설정(configs/api.yaml)을 읽되, DB 는 임시 파일로 바꾸고 모델·자동 종료는 끈다."""
    cfg = load_settings()
    cfg.update(database=tmp_path / 'test.sqlite', model=None, session_idle_lines=None)
    return cfg


class Predictor:
    """가짜 판정기. 실시간 판정은 항상 정상.
    종료 판정은 이벤트가 1개뿐이면 이상(점수 0.9), 2개 이상이면 정상(0.1) → 재판정으로 결과가 바뀌는지 확인할 수 있다."""
    model_name = 'test'

    def predict(self, events):
        return Prediction(0.1, False)

    def predict_close(self, events):
        return Prediction(0.9 if len(events) == 1 else 0.1, len(events) == 1)


def test_late_log_reopens_session_and_survives_restart(cfg):
    """1. 종료 후 로그가 더 오면 재판정 (서버를 껐다 켜도 DB 기준으로 동작)"""
    with TestClient(create_app(cfg, Predictor())) as client:
        assert client.post('/logs', json={'raw_log': RAW}).status_code == 201
        assert client.post('/sessions/close').json() == {'closed': 1}      # 이벤트 1개 → 이상 판정
        first = client.get('/sessions').json()[0]
        assert first['is_anomaly'] == 1
        assert client.post('/sessions/close').json() == {'closed': 0}      # 새 로그가 없으면 다시 판정하지 않음
        assert client.post('/logs', json={'raw_log': RAW}).status_code == 201   # 같은 블록에 로그 추가
    with TestClient(create_app(cfg, Predictor())) as client:               # 서버 재시작
        assert client.post('/sessions/close').json() == {'closed': 1}      # 추가된 로그 때문에 재판정
        rows = client.get('/sessions').json()
        assert len(rows) == 1 and rows[0]['last_log_id'] > first['last_log_id']   # 행은 1개 그대로, 내용만 갱신
        assert rows[0]['is_anomaly'] == 0                                  # 이벤트 2개 → 정상으로 바뀜
        assert client.post('/sessions/close').json() == {'closed': 0}


def test_close_failure_rolls_back_all_results(cfg):
    """2. 두 번째 블록 판정에서 모델이 실패하면 첫 번째 블록 결과도 저장하지 않는다 (전부 아니면 전무)"""
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
        assert client.get('/sessions').json() == []                        # 종료 결과는 하나도 남지 않음
        assert client.get('/stats').json()['total_records'] == 2           # 수신한 로그 기록은 그대로
        assert client.post('/sessions/close').json() == {'closed': 2}      # 다시 시도하면 두 블록 모두 판정


@pytest.mark.parametrize('endpoint,method', [('/logs', 'get'), ('/sessions', 'get'), ('/stats', 'get'), ('/sessions/close', 'post')])
def test_database_failure_returns_503(cfg, monkeypatch, endpoint, method):
    """3. DB 연결이 실패하면 503 + 일반 문구 (오류 메시지 속 DB 경로 같은 내부 정보는 노출하지 않음)"""
    app = create_app(cfg, Predictor())
    with TestClient(app) as client:
        def fail():
            raise sqlite3.OperationalError('private database path')
        monkeypatch.setattr(app.state.db, 'connect', fail)
        response = getattr(client, method)(endpoint)
        assert response.status_code == 503
        assert response.json() == {'detail': 'Database temporarily unavailable'}


@pytest.mark.parametrize('rate,timeout', [(float('nan'), 1), (float('inf'), 1), (1, float('nan')), (1, float('inf'))])
def test_replay_rejects_nonfinite_rate_or_timeout(rate, timeout):
    """4. NaN·무한대는 '0보다 큰가' 비교를 통과해 버리므로 따로 거부해야 한다"""
    with pytest.raises(ValueError):
        replay('unused.log', 'http://localhost/logs', rate, timeout)


def test_collection_import_does_not_require_model_dependencies():
    """5. 새 파이썬 프로세스에서 모델 패키지를 import 하면 오류가 나게 막아 둔 뒤 서버 모듈을 불러온다.
    수집 모드는 모델 패키지 없이도 서버 모듈을 불러올 수 있어야 한다."""
    script = '''
import sys
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'pandas', 'sklearn'}:
            raise ImportError('Model dependency must not load: ' + fullname)
sys.meta_path.insert(0, BlockModels())
from logguard.api.main import create_app
'''
    # 2026-11-19: 자식 프로세스가 설치된 패키지가 아니라 "이 저장소의 src" 를 불러오도록 경로를 명시
    src = str(Path(__file__).resolve().parents[1] / 'src')
    env = {**os.environ, 'PYTHONPATH': src + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
