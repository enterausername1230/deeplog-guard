"""데이터 파이프라인 단위 테스트 (원본 데이터 없이 실행 가능)."""
import numpy as np
import pandas as pd

from logguard.baseline.rule_based import RuleBasedDetector, count_matrix, error_event_ids
from logguard.config import ROOT
from logguard.data.parse import HDFSParser, split_header
from logguard.data.split import assign_splits
from logguard.data.windows import EOS, PAD, UNK, encode, next_event_windows, sliding_windows
from logguard.eval.metrics import percentile_threshold, prf

SAMPLE = [
    "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
    "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-6952295868487656571 terminating",
    "081109 203807 222 INFO dfs.DataNode$PacketResponder: Received block blk_-6952295868487656571 of size 67108864 from /10.251.39.144",
    "081109 203807 222 INFO dfs.DataNode$PacketResponder: Received block blk_123 of size 3 from /10.250.11.53",
    "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* ask 10.250.14.224:50010 to delete  blk_-1 blk_-2 blk_3",
    "081109 204106 35 INFO dfs.FSNamesystem: BLOCK* ask 10.251.39.1:50010 to delete  blk_9",
]


def test_split_header():
    date, tm, level, comp, content = split_header(SAMPLE[0])
    assert (date, tm, level, comp) == ("081109", "203615", "INFO", "dfs.DataNode$PacketResponder")
    assert content.startswith("PacketResponder 1")
    assert split_header("garbage") is None


def test_drain_groups_same_template():
    p = HDFSParser(ROOT / "configs" / "drain3.ini")
    ids = [p.parse(split_header(l)[4]) for l in SAMPLE]
    assert ids[0] == ids[1]          # 숫자/블록ID만 다름
    assert ids[2] == ids[3]          # IP·크기만 다름
    assert ids[4] == ids[5]          # 삭제 블록 개수가 달라도 같은 템플릿
    assert len(set(ids)) == 3


def test_windows():
    seq = np.array([5, 6, 7])
    X, y = next_event_windows(seq, size=2)
    assert X.tolist() == [[PAD, PAD], [PAD, 5], [5, 6]]
    assert y.tolist() == [5, 6, 7]
    assert sliding_windows(seq, 2).tolist() == [[5, 6], [6, 7]]
    assert sliding_windows(seq, 5).tolist() == [[PAD, PAD, 5, 6, 7]]
    assert encode(np.array([3, 99, 99999]), {"3": 5}).tolist() == [5, UNK, UNK]
    assert encode(np.array([3]), {"3": 5}, add_eos=True).tolist() == [5, EOS]


def test_prf_and_threshold():
    m = prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["f1"] == 0.5
    assert percentile_threshold(range(101), 95) == 95.0


def _sessions(n_normal=90, n_anom=10):
    return pd.DataFrame({
        "block_id": [f"blk_{i}" for i in range(n_normal + n_anom)],
        "first_line": np.arange(n_normal + n_anom),
        "label": np.array([0] * n_normal + [1] * n_anom, dtype=np.int8),
    })


def test_split_no_leak_and_stratified():
    s = _sessions()
    split = assign_splits(s, {"train": 0.6, "val": 0.2, "test": 0.2}, "random", seed=0)
    assert set(split) == {"train", "val", "test"}
    assert split.groupby(s["label"]).value_counts().loc[1].to_dict() == {"train": 6, "val": 2, "test": 2}
    chrono = assign_splits(s, {"train": 0.6, "val": 0.2, "test": 0.2}, "chrono", seed=0)
    assert (chrono.iloc[:60] == "train").all() and (chrono.iloc[80:] == "test").all()


def test_rule_based():
    templates = pd.DataFrame({
        "event_id": [1, 2, 3],
        "template": ["Receiving block <*>", "Exception in receiveBlock for block <*>", "Served block <*>"],
        "levels": ["INFO", "INFO", "INFO|WARN"],
    })
    assert error_event_ids(templates, ["exception"], ["WARN"]) == {2, 3}

    def sess(seqs):
        return pd.DataFrame({"events": [np.array(s, dtype=np.int16) for s in seqs],
                             "length": [len(s) for s in seqs]})

    train = sess([[1, 1], [1, 1, 1], [1]])
    assert count_matrix(train, 4).tolist() == [[0, 2, 0, 0], [0, 3, 0, 0], [0, 1, 0, 0]]
    det = RuleBasedDetector({2}, 4).fit(train)
    flags = det.rule_flags(sess([[1, 1], [1, 2], [1] * 10]), q=100)
    assert flags["keyword"].tolist() == [False, True, False]
    assert flags["length"].tolist() == [False, False, True]
    assert flags["frequency"].tolist() == [False, True, True]
    # 정상 최소 길이(1)보다 짧을 수는 없으니 q=100 에선 짧은 쪽 규칙이 안 걸리고, 학습 길이 2~3 기준 q=50 이면 1 은 짧음
    det2 = RuleBasedDetector({2}, 4).fit(sess([[1, 1], [1, 1, 1], [1, 1], [1, 1, 1]]))
    assert det2.rule_flags(sess([[1], [1, 1, 1, 1]]), q=50)["length"].tolist() == [True, True]
