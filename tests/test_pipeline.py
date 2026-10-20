"""데이터 파이프라인 단위 테스트 (원본 데이터 없이 실행 가능).

단위 테스트란?
    "이 함수에 이 값을 넣으면 반드시 이 결과가 나와야 한다" 를 코드로 적어 둔 것.
    코드를 고친 뒤 `make test` (= pytest) 를 돌리면, 기존 기능이 망가졌는지 몇 초 만에 자동으로 확인된다.
    assert 조건 : 조건이 거짓이면 그 테스트는 실패(FAILED)로 표시된다.

이 파일이 검사하는 것
    - 로그 헤더 분리 (split_header)
    - Drain 이 같은 모양의 로그를 같은 템플릿으로 묶는지
    - 윈도우 만들기 / Event ID → 모델 번호 변환 (PAD·UNK·EOS)
    - Precision/Recall/F1 계산과 백분위수 임계치
    - train/val/test 분할의 층화(라벨 비율 유지)와 시간순 분할
    - Rule-Based 규칙 3종 (키워드, 길이, 빈도)

실행: make test   또는   .venv/bin/python -m pytest -q
"""
import numpy as np
import pandas as pd

from logguard.baseline.rule_based import RuleBasedDetector, count_matrix, error_event_ids
from logguard.config import ROOT
from logguard.data.parse import HDFSParser, split_header
from logguard.data.split import assign_splits
from logguard.data.windows import EOS, PAD, UNK, encode, next_event_windows, sliding_windows
from logguard.eval.metrics import percentile_threshold, prf

# 실제 HDFS.log 에서 가져온 형식의 예시 로그 6줄 (데이터 파일 없이 테스트할 수 있도록 코드에 직접 넣음)
#   0, 1번: 숫자·블록 ID 만 다른 "PacketResponder … terminating"  → 같은 템플릿이어야 함
#   2, 3번: IP·크기만 다른 "Received block … of size … from …"   → 같은 템플릿이어야 함
#   4, 5번: 삭제할 블록 개수(3개 vs 1개)가 다른 "BLOCK* ask … to delete …" → 같은 템플릿이어야 함
SAMPLE = [
    "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
    "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-6952295868487656571 terminating",
    "081109 203807 222 INFO dfs.DataNode$PacketResponder: Received block blk_-6952295868487656571 of size 67108864 from /10.251.39.144",
    "081109 203807 222 INFO dfs.DataNode$PacketResponder: Received block blk_123 of size 3 from /10.250.11.53",
    "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* ask 10.250.14.224:50010 to delete  blk_-1 blk_-2 blk_3",
    "081109 204106 35 INFO dfs.FSNamesystem: BLOCK* ask 10.251.39.1:50010 to delete  blk_9",
]


def test_split_header():
    """로그 한 줄이 (날짜, 시각, 레벨, 컴포넌트, 내용) 으로 정확히 나뉘고, 형식이 깨진 줄은 None 이 되는지."""
    date, tm, level, comp, content = split_header(SAMPLE[0])
    assert (date, tm, level, comp) == ("081109", "203615", "INFO", "dfs.DataNode$PacketResponder")
    assert content.startswith("PacketResponder 1")
    assert split_header("garbage") is None          # 공백으로 5조각이 안 나오는 줄


def test_drain_groups_same_template():
    """Drain(마스킹 + 클러스터링)이 모양이 같은 로그를 같은 Event ID 로 묶는지 (configs/drain3.ini 규칙 검증)."""
    p = HDFSParser(ROOT / "configs" / "drain3.ini")
    ids = [p.parse(split_header(l)[4]) for l in SAMPLE]   # [4] = 내용(content)
    assert ids[0] == ids[1]          # 숫자/블록ID만 다름
    assert ids[2] == ids[3]          # IP·크기만 다름
    assert ids[4] == ids[5]          # 삭제 블록 개수가 달라도 같은 템플릿 (BLK 마스킹이 여러 블록을 하나로 묶음)
    assert len(set(ids)) == 3        # 6줄 → 템플릿 정확히 3종


def test_windows():
    """윈도우 만들기와 번호 변환이 설명대로 동작하는지 (windows.py 의 예시와 같은 값)."""
    seq = np.array([5, 6, 7])
    # DeepLog 식: 직전 2개(X) → 다음 1개(y). 앞을 PAD 로 채워 첫 이벤트부터 예측
    X, y = next_event_windows(seq, size=2)
    assert X.tolist() == [[PAD, PAD], [PAD, 5], [5, 6]]
    assert y.tolist() == [5, 6, 7]
    # Autoencoder 식: 길이 2 창문을 한 칸씩 이동
    assert sliding_windows(seq, 2).tolist() == [[5, 6], [6, 7]]
    # 시퀀스가 창문보다 짧으면 왼쪽 PAD 로 채운 윈도우 1개
    assert sliding_windows(seq, 5).tolist() == [[PAD, PAD, 5, 6, 7]]
    # 어휘에 없는 99, 그리고 변환표 범위를 넘는 99999 는 모두 UNK
    assert encode(np.array([3, 99, 99999]), {"3": 5}).tolist() == [5, UNK, UNK]
    # add_eos=True 면 끝에 EOS
    assert encode(np.array([3]), {"3": 5}, add_eos=True).tolist() == [5, EOS]


def test_prf_and_threshold():
    """지표 계산이 손으로 계산한 값과 같은지."""
    # 정답 [이상, 이상, 정상, 정상] vs 예측 [이상, 정상, 이상, 정상] → TP·FN·FP·TN 각 1개, F1 = 0.5
    m = prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["f1"] == 0.5
    # 0~100 의 95 백분위수는 95
    assert percentile_threshold(range(101), 95) == 95.0


def _sessions(n_normal=90, n_anom=10):
    """테스트용 가짜 세션 표: 정상 90개 + 이상 10개 (이상 비율 10%)."""
    return pd.DataFrame({
        "block_id": [f"blk_{i}" for i in range(n_normal + n_anom)],
        "first_line": np.arange(n_normal + n_anom),
        "label": np.array([0] * n_normal + [1] * n_anom, dtype=np.int8),
    })


def test_split_no_leak_and_stratified():
    """random 분할은 라벨 비율을 유지하고(층화), chrono 분할은 등장 순서대로 나뉘는지."""
    s = _sessions()
    split = assign_splits(s, {"train": 0.6, "val": 0.2, "test": 0.2}, "random", seed=0)
    assert set(split) == {"train", "val", "test"}
    # 이상 세션 10개가 정확히 6:2:2 로 나뉘어야 층화가 된 것
    assert split.groupby(s["label"]).value_counts().loc[1].to_dict() == {"train": 6, "val": 2, "test": 2}
    chrono = assign_splits(s, {"train": 0.6, "val": 0.2, "test": 0.2}, "chrono", seed=0)
    # 시간순: 앞 60개는 train, 마지막 20개는 test
    assert (chrono.iloc[:60] == "train").all() and (chrono.iloc[80:] == "test").all()


def test_rule_based():
    """Rule-Based 규칙 3종(키워드, 길이, 빈도)이 의도대로 걸리는지."""
    templates = pd.DataFrame({
        "event_id": [1, 2, 3],
        "template": ["Receiving block <*>", "Exception in receiveBlock for block <*>", "Served block <*>"],
        "levels": ["INFO", "INFO", "INFO|WARN"],
    })
    # 2번은 "Exception" 키워드, 3번은 WARN 레벨로 찍힌 적이 있어서 둘 다 오류 이벤트
    assert error_event_ids(templates, ["exception"], ["WARN"]) == {2, 3}

    def sess(seqs):
        """이벤트 시퀀스 목록 → 세션 표 (events, length 열)."""
        return pd.DataFrame({"events": [np.array(s, dtype=np.int16) for s in seqs],
                             "length": [len(s) for s in seqs]})

    train = sess([[1, 1], [1, 1, 1], [1]])
    # 세션별 이벤트 횟수 행렬: 열 = Event ID 0~3
    assert count_matrix(train, 4).tolist() == [[0, 2, 0, 0], [0, 3, 0, 0], [0, 1, 0, 0]]
    det = RuleBasedDetector({2}, 4).fit(train)
    # q=100: 정상 학습 세션의 최댓값을 임계치로 (길이 상한 3, 이벤트1 횟수 상한 3, 이벤트2 상한 0)
    flags = det.rule_flags(sess([[1, 1], [1, 2], [1] * 10]), q=100)
    assert flags["keyword"].tolist() == [False, True, False]       # 이벤트 2(오류)가 있는 세션만
    assert flags["length"].tolist() == [False, False, True]        # 길이 10 > 상한 3
    assert flags["frequency"].tolist() == [False, True, True]      # 이벤트 2 등장(상한 0 초과), 이벤트 1 이 10번
    # 정상 최소 길이(1)보다 짧을 수는 없으니 q=100 에선 짧은 쪽 규칙이 안 걸리고, 학습 길이 2~3 기준 q=50 이면 1 은 짧음
    det2 = RuleBasedDetector({2}, 4).fit(sess([[1, 1], [1, 1, 1], [1, 1], [1, 1, 1]]))
    # q=50: 길이 하한·상한 모두 2.5 → 길이 1(너무 짧음)과 4(너무 김) 둘 다 이상
    assert det2.rule_flags(sess([[1], [1, 1, 1, 1]]), q=50)["length"].tolist() == [True, True]
