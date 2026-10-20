"""모델 공통 모듈 테스트 (원본 데이터 없이 실행 가능).

이 파일이 검사하는 것
    - 세션 인코딩과 UNK 표시 (dataset.encode_sessions, unk_mask)
    - 중복 윈도우 압축과 원래 위치 복원 (dataset.build_windows)
    - 세션 점수(최댓값), UNK 세션 무한대, 임계치 탐색, 0~1 정규화 (scoring)
    - 가중 학습 루프가 실제로 손실을 줄이는지 (train_utils)
    - 통계 빈도 탐지기 판정·저장/복원 (inference)
    - 온라인 파서가 저장된 Drain 상태로 학습 때와 같은 Event ID 를 주는지 (online)
    - DeepLog 가 "일찍 끝난 세션" 과 "순서가 바뀐 이벤트" 를 잡는지 (deeplog)
    - Autoencoder 가 정상 패턴은 복원하고 이상 패턴은 잡는지 (autoencoder)

딥러닝 테스트는 실제 데이터 대신 "1 → 2 → 3 → EOS" 만 반복되는 아주 작은 가짜 데이터로
작은 모델을 몇 초 만에 학습시켜, 핵심 동작(정상은 통과, 이상은 탐지)만 확인한다.
"""
import numpy as np
import pandas as pd
import torch

from logguard.data.online import OnlineParser
from logguard.data.windows import EOS, PAD, UNK
from logguard.models.dataset import build_windows, encode_sessions, unk_mask
from logguard.models.inference import StatisticalFrequencyDetector
from logguard.models.scoring import normalize, search_threshold, session_scores
from logguard.models.train_utils import sample_weights, train_weighted

# 테스트용 어휘: Event ID 1 → 번호 3, Event ID 2 → 번호 4 (0~2 는 특수 토큰)
VOCAB = {"event_to_index": {"1": 3, "2": 4}}


def test_encode_sessions_and_unk():
    """Event ID 시퀀스가 모델 번호로 바뀌고, 어휘에 없는 이벤트(9)는 UNK 가 되며, 그 세션이 UNK 세션으로 표시되는지."""
    df = pd.DataFrame({"events": [np.array([1, 2]), np.array([1, 9])]})
    seqs = encode_sessions(df, VOCAB, add_eos=True)
    assert [s.tolist() for s in seqs] == [[3, 4, EOS], [3, UNK, EOS]]
    assert unk_mask(seqs).tolist() == [False, True]


def test_build_windows_dedupes_and_restores():
    """같은 윈도우는 하나로 합쳐지고(weight 로 개수 보존), inverse 로 원래 순서를 정확히 되살릴 수 있는지."""
    # 세션 3개: 앞의 두 세션은 완전히 같다 → 그 윈도우들은 중복
    seqs = [np.array([3, 4, EOS]), np.array([3, 4, EOS]), np.array([3, EOS])]
    ws = build_windows(seqs, w=2, with_next=True)
    # 전체 윈도우 = 세션 길이 합 = 3 + 3 + 2 = 8, 가중치 합도 8 이어야 함 (개수 보존)
    assert ws.n_total == 8 and ws.weight.sum() == 8
    # 고유 윈도우를 inverse 로 펼치면 원래 8행이 순서대로 복원돼야 함 (각 행 = [직전2개, 다음])
    rows = np.concatenate([ws.X, ws.y[:, None]], 1)[ws.inverse]
    assert rows[:3].tolist() == [[PAD, PAD, 3], [PAD, 3, 4], [3, 4, EOS]]     # 세션 0
    assert rows[6:].tolist() == [[PAD, PAD, 3], [PAD, 3, EOS]]                 # 세션 2
    assert len(ws.X) == 4                                  # 고유 윈도우 4개
    # Autoencoder 용(with_next=False)은 y 가 없고 윈도우 폭이 w
    ae = build_windows(seqs, w=2, with_next=False)
    assert ae.y is None and ae.n_total == 8 and ae.X.shape[1] == 2


def test_session_scores_max_and_unk():
    """세션 점수 = 윈도우 점수 최댓값, UNK 세션은 +무한대."""
    seqs = [np.array([3, 4]), np.array([3, 3])]
    ws = build_windows(seqs, w=1, with_next=False)       # 고유 윈도우: [3], [4]
    uniq_scores = np.where(ws.X[:, 0] == 4, 0.9, 0.1)    # 윈도우 [4] 는 0.9점, [3] 은 0.1점
    s = session_scores(uniq_scores, ws, unk=np.array([False, True]))
    assert s[0] == 0.9 and np.isinf(s[1])                # 세션0 = max(0.1, 0.9), 세션1 = UNK 라 무한대


def test_threshold_search_and_normalize():
    """백분위수 임계치 탐색이 정상/이상을 완벽히 가르는 값을 찾고, 정규화가 임계치를 0.5 로 보내는지."""
    scores = np.array([0.1, 0.2, 0.3, 0.9, np.inf])
    labels = np.array([0, 0, 0, 1, 1])
    # p=100 → 정상 점수 최댓값 0.3 이 임계치 → 0.9 와 inf 만 이상 → F1 1.0
    table, best = search_threshold(scores, labels, [50.0, 100.0])
    assert best["f1"] == 1.0
    # s/(s+thr): 0 → 0.0, thr(0.3) → 0.5, 무한대 → 1.0
    n = normalize(np.array([0.0, 0.3, np.inf]), 0.3)
    assert n.tolist() == [0.0, 0.5, 1.0]


def test_train_weighted_runs():
    """가중 학습 루프가 에폭 수만큼 손실 기록을 남기고, 학습할수록 손실이 줄어드는지."""
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2)                          # 아주 간단한 모델 (입력 2개 → 클래스 2개)
    X = np.random.rand(20, 2).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64)                   # 첫 번째 값이 0.5 보다 크면 1 인 쉬운 문제
    loss = lambda m, xb, yb: torch.nn.functional.cross_entropy(m(xb), yb, reduction="none")
    hist = train_weighted(model, (X, y), sample_weights(np.arange(1, 21)), loss,
                          epochs=3, batch_size=8, lr=0.1, device=torch.device("cpu"), log_every=0)
    assert len(hist) == 3 and hist[-1] < hist[0]


def test_statistical_detector():
    """통계 빈도 탐지기: 허용 횟수 이내면 정상, 넘으면 이상, 모르는 이벤트면 즉시 이상, 저장/복원해도 결과가 같은지."""
    # 정상 학습: 이벤트 1 은 최대 2번, 이벤트 2 는 최대 1번 → q=100 이면 허용 횟수 (1: 2회, 2: 1회)
    train = pd.DataFrame({"events": [np.array([1, 1, 2]), np.array([1, 2])], "length": [3, 2]})
    det = StatisticalFrequencyDetector.fit(train, 100, {1: "A <*>", 2: "B <*>", 3: "C <*>"})
    assert not det.score_session([1, 1, 2]).is_anomaly             # 허용 범위 안
    r = det.score_session([1, 1, 1, 2])                             # 이벤트 1 이 3번 > 허용 2번
    assert r.is_anomaly and r.reason == "model" and r.details["event_id"] == 1 and r.score > 0.5
    r = det.score_session([1, 3])                                   # 이벤트 3 은 학습에 없었음
    assert r.is_anomaly and r.reason == "unknown_event" and r.score == 1.0
    assert det.score_event([1, None]).reason == "unknown_event"     # None(파서가 모르는 로그)도 즉시 이상
    # json 저장 → 복원해도 임계치와 판정 결과가 똑같아야 함
    restored = StatisticalFrequencyDetector.from_json(det.to_json())
    assert restored.thresholds == det.thresholds
    assert restored.score_session([1, 1, 1, 2]).to_dict() == det.score_session([1, 1, 1, 2]).to_dict()


def test_online_parser_uses_saved_drain_state():
    """온라인 파서가 git 에 포함된 Drain 상태 파일을 불러와 학습 때와 같은 규칙으로 Event ID 를 매기는지.

    특히 1단계에서 고친 문제(terminating 과 Interrupted 가 합쳐지던 것)가 재발하지 않는지 확인한다.
    """
    p = OnlineParser.load()
    a = p.parse_line("081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating")
    b = p.parse_line("081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1 terminating")
    c = p.parse_line("081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1 Interrupted.")
    # a, b 는 같은 템플릿 / c(Interrupted, 이상 신호)는 다른 템플릿이어야 함
    assert a.event_id == b.event_id is not None and c.event_id not in (None, a.event_id)
    assert a.block_ids == ["blk_38865049064139660"]
    assert p.parse_line("totally new log format").event_id is None   # 처음 보는 형식 → None


def test_deeplog_learns_and_detects_truncation():
    """정상 패턴 1→2→3→EOS 만 학습하면, 1→2 에서 끊긴 세션(EOS 가 너무 이름)을 이상으로 본다."""
    from logguard.models.deeplog import DeepLogDetector, DeepLogModel, window_scores
    torch.manual_seed(0)
    vocab = {"event_to_index": {"1": 3, "2": 4, "3": 5}}
    # 같은 정상 세션 50개로 학습 데이터 구성
    seqs = encode_sessions(pd.DataFrame({"events": [np.array([1, 2, 3])] * 50}), vocab, add_eos=True)
    ws = build_windows(seqs, w=3, with_next=True)
    model = DeepLogModel(num_classes=6, embed_dim=8, hidden_size=16, num_layers=1)   # 테스트용 작은 모델
    loss = lambda m, xb, yb: torch.nn.functional.cross_entropy(m(xb), yb, reduction="none")
    train_weighted(model, (ws.X, ws.y), sample_weights(ws.weight), loss,
                   epochs=150, batch_size=8, lr=0.05, device=torch.device("cpu"), log_every=0)
    nll, rank = window_scores(model, ws.X, ws.y)
    assert (rank == 0).all()                                    # 정상 패턴은 모두 1등 예측

    # top-k(k=1) 방식 Detector: 실제 이벤트가 1등 예측이 아니면 이상
    meta = {"window_size": 3, "add_eos": True, "mode": "topk", "k": 1, "num_classes": 6,
            "event_to_index": vocab["event_to_index"], "templates": {"1": "A", "2": "B", "3": "C"}}
    det = DeepLogDetector(model, meta)
    assert not det.score_session([1, 2, 3]).is_anomaly
    cut = det.score_session([1, 2])                             # 3 이 와야 하는데 EOS
    assert cut.is_anomaly and cut.details["actual"]["template"].startswith("<EOS")
    assert not det.score_event([1, 2]).is_anomaly               # 스트리밍 중엔 아직 끝난 게 아니므로 정상
    assert det.score_event([1, 3]).is_anomaly                   # 2 대신 3 → 순서 이상
    assert det.score_session([1, 7]).reason == "unknown_event"  # 7 은 어휘에 없음


def test_autoencoder_reconstructs_normal_and_flags_unusual():
    """Autoencoder: 정상 패턴은 잘 복원(정상), 일찍 끝나거나 순서가 바뀐 패턴은 복원 실패(이상)."""
    from logguard.models.autoencoder import AutoencoderDetector, WindowAutoencoder, recon_nll, window_errors
    torch.manual_seed(0)
    vocab = {"event_to_index": {"1": 3, "2": 4, "3": 5}}
    seqs = encode_sessions(pd.DataFrame({"events": [np.array([1, 2, 3])] * 50}), vocab, add_eos=True)
    ws = build_windows(seqs, w=3, with_next=False)
    model = WindowAutoencoder(window_size=3, num_classes=6, hidden_dims=[16, 4])
    train_weighted(model, (ws.X,), sample_weights(ws.weight), lambda m, xb: recon_nll(m(xb), xb).sum(1),
                   epochs=300, batch_size=8, lr=0.02, device=torch.device("cpu"), log_every=0, lr_schedule="cosine")
    thr = float(window_errors(model, ws.X).max()) + 0.5          # 정상 윈도우 최대 오차보다 약간 위
    meta = {"window_size": 3, "add_eos": True, "hidden_dims": [16, 4], "num_classes": 6, "threshold": thr,
            "event_to_index": vocab["event_to_index"], "templates": {"1": "A", "2": "B", "3": "C"}}
    det = AutoencoderDetector(model, meta)
    assert not det.score_session([1, 2, 3]).is_anomaly
    assert det.score_session([1, 2]).is_anomaly                  # 일찍 끝남 (EOS 위치 이상)
    bad = det.score_session([2, 1, 3])                           # 순서 뒤바뀜
    assert bad.is_anomaly and bad.score > 0.5 and "worst" in bad.details
    assert det.score_session([1, 7]).reason == "unknown_event"
