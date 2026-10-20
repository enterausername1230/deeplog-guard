import pytest
from fastapi.testclient import TestClient
from logguard.api.main import create_app
from logguard.api.settings import load_settings
from logguard.api.inference import Prediction
from logguard.api.replay import iter_logs

RAW = '081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating'


@pytest.fixture
def settings(tmp_path):
    """임시 DB 를 쓰는 설정. 기존 테스트는 "모델 없이 수집만" 동작을 검사하므로 model 을 명시적으로 끈다.

    (2026-10-20: configs/api.yaml 의 기본 model 이 autoencoder 가 되어, 이 줄이 없으면 실제 모델이 연결된다)
    """
    cfg = load_settings()
    cfg['database'] = tmp_path / 'logs.sqlite'
    cfg['model'] = None
    return cfg


def test_collect_and_persist(settings):
    state = settings['drain_state'].read_bytes()
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').json()['model_ready'] is False
        result = client.post('/logs', json={'raw_log': RAW})
        assert result.status_code == 201
        record = result.json()['records'][0]
        assert record['event_id'] is not None
        assert record['status'] == 'pending_model'
        assert record['is_anomaly'] is None and record['anomaly_score'] is None
        assert record['alerted'] is False
        assert client.get('/logs', params={'after_id': record['id']}).json() == []
    with TestClient(create_app(settings)) as client:
        assert len(client.get('/logs').json()) == 1
        assert client.get('/stats').json()['total_records'] == 1
    assert settings['drain_state'].read_bytes() == state


def test_invalid_and_oversized(settings):
    with TestClient(create_app(settings)) as client:
        assert client.post('/logs', json={'raw_log': 'garbage'}).status_code == 422
        assert client.post('/logs', json={'raw_log': RAW + '\n' + RAW}).status_code == 422
        assert client.post('/logs', json={'raw_log': 'x' * 17000}).status_code == 413
        assert client.get('/stats').json()['total_records'] == 0
        assert client.get('/logs?limit=1001').status_code == 422


def test_unknown_no_block_and_multi_block(settings):
    with TestClient(create_app(settings)) as client:
        unknown = '081109 203615 148 INFO custom.Component: brandnew mysterious message'
        record = client.post('/logs', json={'raw_log': unknown}).json()['records'][0]
        assert record['event_id'] is None and record['event_index'] == 1
        assert record['status'] == 'no_block' and record['is_anomaly'] is None
        raw = '081109 204005 35 INFO dfs.FSNamesystem: BLOCK* ask 10.250.14.224:50010 to delete blk_-1 blk_-2 blk_-1'
        records = client.post('/logs', json={'raw_log': raw}).json()['records']
        assert [r['block_id'] for r in records] == ['blk_-1', 'blk_-2']
        assert len(client.get('/logs?block_id=blk_-1').json()) == 1


def test_model_adapter_history_and_restart(settings):
    class Adapter:
        def __init__(self):
            self.calls = []
        def predict(self, events):
            self.calls.append(events)
            return Prediction(0.9, True)
    adapter = Adapter()
    settings['window_size'] = 2
    for _ in range(2):
        with TestClient(create_app(settings, adapter)) as client:
            for _ in range(2):
                result = client.post('/logs', json={'raw_log': RAW})
                assert result.json()['records'][0]['status'] == 'anomaly'
            other = RAW.replace('blk_38865049064139660', 'blk_999')
            client.post('/logs', json={'raw_log': other})
    assert [len(x) for x in adapter.calls] == [1, 2, 1, 3, 3, 2]


def test_inference_failure_rolls_back(settings):
    class Broken:
        def __init__(self):
            self.calls = 0
        def predict(self, events):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('not loaded')
            return Prediction(0.1, False)
    with TestClient(create_app(settings, Broken())) as client:
        assert client.post('/logs', json={'raw_log': RAW + ' blk_999'}).status_code == 503
        assert client.get('/stats').json()['total_records'] == 0


def test_replay_filter(tmp_path):
    file = tmp_path / 'sample.log'
    file.write_text(RAW + '\n' + RAW.replace('terminating', 'terminating blk_99') + '\n')
    assert len(list(iter_logs(file))) == 2
    assert list(iter_logs(file, {'blk_38865049064139660'})) == [RAW]


def test_replay_posts_and_stops_on_failure(tmp_path, monkeypatch):
    import httpx
    from logguard.api import replay as module
    file = tmp_path / 'sample.log'
    file.write_text((RAW + '\n') * 3)
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(201 if len(requests) == 1 else 503)
    factory = httpx.Client
    monkeypatch.setattr(module.httpx, 'Client', lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(httpx.HTTPStatusError):
        module.replay(file, 'http://localhost/logs', 10000, 1)
    assert len(requests) == 2
    assert b'raw_log' in requests[0].content
