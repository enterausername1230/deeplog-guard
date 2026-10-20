"""백엔드 + 모델 통합 테스트 (2026-10-20, 옥상민).

검사하는 것
    1) 모델 연결 어댑터(DetectorPredictor)의 시작 시 안전 검사
       - window_size 가 모델 윈도우보다 작으면 거부 (10/20 시험: 10 이면 정상 세션 대부분 오탐)
       - 어휘가 모델 학습 어휘와 다르면 거부
       - 통계 빈도처럼 윈도우가 없는 탐지기는 거부
    2) 실제 Autoencoder 를 연결한 서버가 판정 근거·모델 이름·임계치·지연시간을 DB 에 저장하는지
    3) 세션 종료 판정: 수동(POST /sessions/close)과 자동(idle) 모두 동작하고, 같은 블록은 한 번만 판정하는지
    4) (데이터가 있을 때만) 실제 test 블록 로그를 서버에 흘려 보낸 결과가 오프라인 평가 판정과 100% 일치하는지
"""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from logguard.api.inference import Prediction
from logguard.api.main import create_app
from logguard.api.predictor import DetectorPredictor
from logguard.api.settings import ROOT, load_settings
from logguard.models.inference import load_detector

RAW = '081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating'


@pytest.fixture
def settings(tmp_path):
    """실제 모델(autoencoder)이 연결되는 설정 + 임시 DB. 자동 세션 종료는 기본으로 끈다."""
    cfg = load_settings()
    cfg['database'] = tmp_path / 'logs.sqlite'
    cfg['session_idle_lines'] = None
    return cfg


def _vocab(cfg):
    return json.loads(cfg['vocab'].read_text())['event_to_index']


# ------------------------------------------------------------------ 1) 어댑터 안전 검사

def test_adapter_rejects_small_window_and_wrong_vocab(settings):
    det = load_detector('autoencoder')
    vocab = _vocab(settings)
    with pytest.raises(ValueError, match='window_size'):
        DetectorPredictor(det, vocab, history_size=10)                 # 모델 윈도우 20 보다 작음
    wrong = dict(vocab)
    wrong[next(iter(wrong))] = 999                                      # 번호 하나를 바꾼 가짜 어휘
    with pytest.raises(ValueError, match='어휘'):
        DetectorPredictor(det, wrong, history_size=20)
    with pytest.raises(ValueError, match='윈도우 기반'):
        DetectorPredictor(load_detector('statistical_frequency'), vocab, history_size=20)
    # 서버도 시작 단계에서 거부해야 한다 (조용히 틀린 판정을 내지 않도록)
    settings['window_size'] = 10
    with pytest.raises(ValueError):
        with TestClient(create_app(settings)):
            pass


def test_adapter_maps_unknown_event_to_immediate_anomaly(settings):
    p = DetectorPredictor.from_settings(settings)
    assert p.model_name == 'autoencoder'
    r = p.predict((1,))                       # 1 = UNK (학습에 없던 이벤트)
    assert r.is_anomaly and r.reason == 'unknown_event' and r.anomaly_score == 1.0 and r.threshold == 0.5


# ------------------------------------------------------------------ 2) 실제 모델 연결 서버

def test_server_with_real_model_stores_reason_and_latency(settings):
    with TestClient(create_app(settings)) as client:
        health = client.get('/health').json()
        assert health['model_ready'] and health['model'] == 'autoencoder'
        rec = client.post('/logs', json={'raw_log': RAW, 'sent_at': '2026-10-20T00:00:00+00:00'}).json()['records'][0]
        assert rec['status'] in ('normal', 'anomaly')
        assert rec['model_name'] == 'autoencoder' and rec['threshold'] == 0.5
        assert rec['reason'] in ('model', 'normal', 'unknown_event')
        assert json.loads(rec['details'])                               # 판정 근거가 JSON 으로 저장됨
        assert rec['latency_ms'] > 0 and rec['sent_at'] == '2026-10-20T00:00:00+00:00'


# ------------------------------------------------------------------ 3) 세션 종료 판정

def test_manual_session_close_once_per_block(settings):
    with TestClient(create_app(settings)) as client:
        client.post('/logs', json={'raw_log': RAW})
        client.post('/logs', json={'raw_log': RAW.replace('blk_38865049064139660', 'blk_7')})
        assert client.post('/sessions/close').json() == {'closed': 2}
        assert client.post('/sessions/close').json() == {'closed': 0}   # 이미 판정한 블록은 다시 안 함
        rows = client.get('/sessions').json()
        assert {r['block_id'] for r in rows} == {'blk_38865049064139660', 'blk_7'}
        assert all(r['close_reason'] == 'manual' and r['model_name'] == 'autoencoder' for r in rows)
        # "PacketResponder … terminating" 한 줄만 있고 끝난 블록 = 정상 흐름이 아니므로 끊긴 세션(이상)으로 판정
        assert all(r['is_anomaly'] == 1 for r in rows)
        assert client.get('/stats').json()['sessions_closed'] == 2


def test_idle_session_close_and_collection_mode(settings):
    class Fake:
        """판정은 항상 정상, 종료 판정 호출 횟수만 센다."""
        model_name = 'fake'
        def __init__(self):
            self.closed = []
        def predict(self, events):
            return Prediction(0.1, False)
        def predict_close(self, events):
            self.closed.append(events)
            return Prediction(0.9, True, reason='model')
    fake = Fake()
    settings.update(session_idle_lines=2, session_sweep_every=1)       # 2행 넘게 소식 없으면 종료
    with TestClient(create_app(settings, fake)) as client:
        client.post('/logs', json={'raw_log': RAW})                     # 블록 A (id 1)
        for _ in range(3):                                              # 블록 B (id 2~4)
            client.post('/logs', json={'raw_log': RAW.replace('blk_38865049064139660', 'blk_B')})
        rows = client.get('/sessions').json()
        assert [r['block_id'] for r in rows] == ['blk_38865049064139660']   # A 만 idle 종료
        assert rows[0]['close_reason'] == 'idle' and len(fake.closed) == 1
    # 모델 없이 수집만 하는 모드에서는 세션 종료 판정을 요청하면 409
    settings['model'] = None
    with TestClient(create_app(settings)) as client:
        assert client.post('/sessions/close').status_code == 409


# ------------------------------------------------------------------ 4) 오프라인 평가와의 일치 (데이터 필요)

TEST_PARQUET = ROOT / 'data/processed/test.parquet'
HDFS_LOG = ROOT / 'data/raw/HDFS_v1/HDFS.log'


@pytest.mark.skipif(not (TEST_PARQUET.exists() and HDFS_LOG.exists()), reason='make data 후 실행 가능')
def test_streaming_matches_offline_evaluation(settings, tmp_path):
    """test 블록 120개의 로그를 실제 순서대로 서버에 보내고 세션 종료까지 판정 → 오프라인 score_session 과 비교."""
    from logguard.api.replay import iter_logs
    test = pd.read_parquet(TEST_PARQUET).sort_values('first_line').head(120)
    allowed = set(test['block_id'])
    head = tmp_path / 'head.log'
    with open(HDFS_LOG, encoding='utf-8') as f:                         # 필요한 줄까지만 잘라 쓴다 (빠르게)
        head.write_text(''.join(line for _, line in zip(range(int(test['last_line'].max()) + 1), f)))
    with TestClient(create_app(settings)) as client:
        for raw in iter_logs(head, allowed):
            assert client.post('/logs', json={'raw_log': raw}).status_code == 201
        client.post('/sessions/close')
        logs = pd.DataFrame(client.get('/logs', params={'limit': 1000}).json())
        after = logs['id'].max()
        while True:                                                     # 1,000행씩 끝까지 읽기
            more = client.get('/logs', params={'after_id': int(after), 'limit': 1000}).json()
            if not more:
                break
            logs = pd.concat([logs, pd.DataFrame(more)])
            after = more[-1]['id']
        closed = pd.DataFrame(client.get('/sessions', params={'limit': 10000}).json()).set_index('block_id')
    online = logs.groupby('block_id')['is_anomaly'].any() | closed['is_anomaly'].astype(bool)
    det = load_detector('autoencoder')
    offline = {b: det.score_session(e.tolist(), closed=True).is_anomaly for b, e in zip(test.block_id, test.events)}
    assert all(bool(online[b]) == offline[b] for b in allowed)
