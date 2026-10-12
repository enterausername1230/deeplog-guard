"""세션(이벤트 시퀀스)을 모델 입력용 슬라이딩 윈도우로 바꾸는 유틸.

- DeepLog : (윈도우 w개 이벤트) -> 다음 이벤트 예측   => next_event_windows()
- Autoencoder : 윈도우 자체를 압축·복원            => sliding_windows()
세션 내부에서만 윈도우를 만들므로 블록 간 섞임(누수)이 없다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 특수 토큰 (프로젝트 전체에서 이 값만 사용)
PAD, UNK, EOS = 0, 1, 2          # EOS: 세션 종료. 정상 시퀀스 앞부분에서 끊긴 이상 세션 탐지용 (EDA 결론 #4)
NUM_SPECIAL = 3


def load_vocab(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def make_lookup(event_to_index: dict[str, int], max_event_id: int = 4096) -> np.ndarray:
    """event_id -> 어휘 인덱스 배열. 어휘에 없는 id 는 UNK."""
    lut = np.full(max_event_id + 1, UNK, dtype=np.int64)
    for e, i in event_to_index.items():
        lut[int(e)] = i
    return lut


def encode(events: np.ndarray, event_to_index: dict[str, int] | np.ndarray, add_eos: bool = False) -> np.ndarray:
    """원본 event_id 시퀀스 -> 어휘 인덱스 (학습 때 못 본 이벤트는 UNK). add_eos 면 끝에 EOS."""
    lut = event_to_index if isinstance(event_to_index, np.ndarray) else make_lookup(event_to_index)
    ev = np.asarray(events, dtype=np.int64)
    out = np.where(ev < len(lut), lut[np.minimum(ev, len(lut) - 1)], UNK)
    return np.append(out, EOS) if add_eos else out


def pad_left(seq: np.ndarray, size: int) -> np.ndarray:
    if len(seq) >= size:
        return seq
    return np.concatenate([np.full(size - len(seq), PAD, dtype=seq.dtype), seq])


def sliding_windows(seq: np.ndarray, size: int, stride: int = 1) -> np.ndarray:
    """길이 size 의 윈도우들 (n_windows, size). 세션이 짧으면 왼쪽 PAD 로 1개 생성."""
    seq = pad_left(np.asarray(seq), size)
    starts = range(0, len(seq) - size + 1, stride)
    return np.stack([seq[s:s + size] for s in starts])


def next_event_windows(seq: np.ndarray, size: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """DeepLog 학습/추론용 (X: (n, size), y: (n,)). 세션 첫 이벤트부터 예측하도록 왼쪽 PAD."""
    seq = np.concatenate([np.full(size, PAD, dtype=np.int64), np.asarray(seq, dtype=np.int64)])
    idx = range(size, len(seq), stride)
    X = np.stack([seq[i - size:i] for i in idx])
    y = np.asarray([seq[i] for i in idx], dtype=np.int64)
    return X, y
