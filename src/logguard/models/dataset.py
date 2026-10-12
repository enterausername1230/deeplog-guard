"""세션 → 모델 입력 윈도우. 중복 윈도우를 고유 윈도우 + 빈도로 압축한다.

정상 학습 윈도우 653만 개가 w=10 에서 고유 (윈도우, 다음 이벤트) 쌍 약 2.9만 개로 줄어든다 (EDA 결론 #5).
- 학습: 고유 쌍 + weight(등장 횟수) → 가중 손실로 전체 데이터 학습과 동일한 목표
- 평가: 고유 윈도우만 한 번씩 채점 → inverse 로 원래 위치에 되돌려 세션 점수 계산

윈도우 정렬: 세션의 각 이벤트(EOS 포함)가 정확히 한 번씩 '마지막(=판정 대상) 위치'에 오도록 왼쪽 PAD.
  with_next=True  (DeepLog)     : X = 직전 w개, y = 현재 이벤트
  with_next=False (Autoencoder) : X = 현재 이벤트로 끝나는 w개
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from logguard.data.windows import PAD, UNK, encode, make_lookup


def encode_sessions(df: pd.DataFrame, vocab: dict, add_eos: bool) -> list[np.ndarray]:
    lut = make_lookup(vocab["event_to_index"])
    return [encode(seq, lut, add_eos=add_eos) for seq in df["events"]]


def unk_mask(seqs: list[np.ndarray]) -> np.ndarray:
    """학습 때 못 본 이벤트가 하나라도 있는 세션 → 모델과 무관하게 즉시 이상 (EDA 결론 #3)."""
    return np.fromiter(((s == UNK).any() for s in seqs), dtype=bool, count=len(seqs))


def _rows(seqs: list[np.ndarray], w: int, with_next: bool) -> tuple[np.ndarray, np.ndarray]:
    width = w + 1 if with_next else w
    pad = np.full(width - 1, PAD, dtype=np.int16)
    rows = [sliding_window_view(np.concatenate([pad, s.astype(np.int16)]), width) for s in seqs]
    session_index = np.repeat(np.arange(len(seqs)), [len(s) for s in seqs])
    return np.concatenate(rows), session_index


@dataclass
class WindowSet:
    X: np.ndarray                    # (n_unique, w)  int64
    y: np.ndarray | None             # (n_unique,)   다음 이벤트 (DeepLog) / None (AE)
    weight: np.ndarray               # (n_unique,)   등장 횟수
    inverse: np.ndarray              # (n_total,)    원래 윈도우 -> 고유 윈도우 인덱스
    session_index: np.ndarray        # (n_total,)    원래 윈도우 -> 세션 인덱스
    n_sessions: int

    @property
    def n_total(self) -> int:
        return len(self.inverse)


def build_windows(seqs: list[np.ndarray], w: int, with_next: bool) -> WindowSet:
    rows, session_index = _rows(seqs, w, with_next)
    # 행 전체를 하나의 바이트열로 보고 1차원 unique (axis=0 보다 수 배 빠름)
    rows = np.ascontiguousarray(rows)
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
    _, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    uniq = rows[first].astype(np.int64)
    X, y = (uniq[:, :-1], uniq[:, -1]) if with_next else (uniq, None)
    return WindowSet(X=X, y=y, weight=counts.astype(np.float64), inverse=inverse.reshape(-1),
                     session_index=session_index, n_sessions=len(seqs))
