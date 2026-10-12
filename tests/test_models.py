"""모델 공통 모듈 테스트 (원본 데이터 없이 실행 가능)."""
import numpy as np
import pandas as pd
import torch

from logguard.data.online import OnlineParser
from logguard.data.windows import EOS, PAD, UNK
from logguard.models.dataset import build_windows, encode_sessions, unk_mask
from logguard.models.inference import StatisticalFrequencyDetector
from logguard.models.scoring import normalize, search_threshold, session_scores
from logguard.models.train_utils import sample_weights, train_weighted

VOCAB = {"event_to_index": {"1": 3, "2": 4}}


def test_encode_sessions_and_unk():
    df = pd.DataFrame({"events": [np.array([1, 2]), np.array([1, 9])]})
    seqs = encode_sessions(df, VOCAB, add_eos=True)
    assert [s.tolist() for s in seqs] == [[3, 4, EOS], [3, UNK, EOS]]
    assert unk_mask(seqs).tolist() == [False, True]


def test_build_windows_dedupes_and_restores():
    seqs = [np.array([3, 4, EOS]), np.array([3, 4, EOS]), np.array([3, EOS])]
    ws = build_windows(seqs, w=2, with_next=True)
    assert ws.n_total == 8 and ws.weight.sum() == 8
    rows = np.concatenate([ws.X, ws.y[:, None]], 1)[ws.inverse]
    assert rows[:3].tolist() == [[PAD, PAD, 3], [PAD, 3, 4], [3, 4, EOS]]
    assert rows[6:].tolist() == [[PAD, PAD, 3], [PAD, 3, EOS]]
    assert len(ws.X) == 4                                  # 고유 윈도우 4개
    ae = build_windows(seqs, w=2, with_next=False)
    assert ae.y is None and ae.n_total == 8 and ae.X.shape[1] == 2


def test_session_scores_max_and_unk():
    seqs = [np.array([3, 4]), np.array([3, 3])]
    ws = build_windows(seqs, w=1, with_next=False)       # 고유 윈도우: [3], [4]
    uniq_scores = np.where(ws.X[:, 0] == 4, 0.9, 0.1)
    s = session_scores(uniq_scores, ws, unk=np.array([False, True]))
    assert s[0] == 0.9 and np.isinf(s[1])


def test_threshold_search_and_normalize():
    scores = np.array([0.1, 0.2, 0.3, 0.9, np.inf])
    labels = np.array([0, 0, 0, 1, 1])
    table, best = search_threshold(scores, labels, [50.0, 100.0])
    assert best["f1"] == 1.0
    n = normalize(np.array([0.0, 0.3, np.inf]), 0.3)
    assert n.tolist() == [0.0, 0.5, 1.0]


def test_train_weighted_runs():
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2)
    X = np.random.rand(20, 2).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64)
    loss = lambda m, xb, yb: torch.nn.functional.cross_entropy(m(xb), yb, reduction="none")
    hist = train_weighted(model, (X, y), sample_weights(np.arange(1, 21)), loss,
                          epochs=3, batch_size=8, lr=0.1, device=torch.device("cpu"), log_every=0)
    assert len(hist) == 3 and hist[-1] < hist[0]


def test_statistical_detector():
    train = pd.DataFrame({"events": [np.array([1, 1, 2]), np.array([1, 2])], "length": [3, 2]})
    det = StatisticalFrequencyDetector.fit(train, 100, {1: "A <*>", 2: "B <*>", 3: "C <*>"})
    assert not det.score_session([1, 1, 2]).is_anomaly
    r = det.score_session([1, 1, 1, 2])
    assert r.is_anomaly and r.reason == "model" and r.details["event_id"] == 1 and r.score > 0.5
    r = det.score_session([1, 3])
    assert r.is_anomaly and r.reason == "unknown_event" and r.score == 1.0
    assert det.score_event([1, None]).reason == "unknown_event"
    restored = StatisticalFrequencyDetector.from_json(det.to_json())
    assert restored.thresholds == det.thresholds
    assert restored.score_session([1, 1, 1, 2]).to_dict() == det.score_session([1, 1, 1, 2]).to_dict()


def test_online_parser_uses_saved_drain_state():
    p = OnlineParser.load()
    a = p.parse_line("081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating")
    b = p.parse_line("081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1 terminating")
    c = p.parse_line("081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1 Interrupted.")
    assert a.event_id == b.event_id is not None and c.event_id not in (None, a.event_id)
    assert a.block_ids == ["blk_38865049064139660"]
    assert p.parse_line("totally new log format").event_id is None
