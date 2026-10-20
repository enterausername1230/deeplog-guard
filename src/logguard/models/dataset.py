"""세션 → 모델 입력 윈도우. 중복 윈도우를 고유 윈도우 + 빈도로 압축한다.

정상 학습 윈도우 653만 개가 w=10 에서 고유 (윈도우, 다음 이벤트) 쌍 약 2.9만 개로 줄어든다 (EDA 결론 #5).
- 학습: 고유 쌍 + weight(등장 횟수) → 가중 손실로 전체 데이터 학습과 동일한 목표
- 평가: 고유 윈도우만 한 번씩 채점 → inverse 로 원래 위치에 되돌려 세션 점수 계산

윈도우 정렬: 세션의 각 이벤트(EOS 포함)가 정확히 한 번씩 '마지막(=판정 대상) 위치'에 오도록 왼쪽 PAD.
  with_next=True  (DeepLog)     : X = 직전 w개, y = 현재 이벤트
  with_next=False (Autoencoder) : X = 현재 이벤트로 끝나는 w개

────────────────────────────────────────────────────────────────────────────────────────────
"중복 압축" 이 왜 가능하고, 왜 중요한가
────────────────────────────────────────────────────────────────────────────────────────────
HDFS 정상 블록들은 거의 같은 순서로 움직인다. 그래서 수백만 개의 윈도우 중 대부분이 "완전히 똑같은" 윈도우다.
    예: [PAD, PAD, …, 1] 이라는 윈도우(세션의 첫 이벤트)는 학습셋에 약 33만 번 나온다.
똑같은 윈도우로 33만 번 학습하는 것은, 그 윈도우로 1번 학습하면서 손실에 33만 배 가중치를 주는 것과 같다.
→ 고유 윈도우 3만 개 + 가중치로 학습하면 결과는 같고 속도는 수백 배 빠르다 (노트북 CPU 로 수십 초).

평가할 때도 같다: 고유 윈도우만 모델에 한 번씩 넣어 점수를 구하고, inverse(되돌리기 지도)를 이용해
원래 수백만 개 윈도우 자리에 점수를 복사한다.

예시 (w=2, with_next=True, 세션 2개)
    세션0 = [3, 4, EOS] → 윈도우 행: [PAD,PAD→3], [PAD,3→4], [3,4→EOS]
    세션1 = [3, EOS]    → 윈도우 행: [PAD,PAD→3], [PAD,3→EOS]
    전체 5행 중 [PAD,PAD→3] 이 2번 → 고유 4개, 그 행의 weight = 2
    session_index = [0, 0, 0, 1, 1]  (각 원래 행이 어느 세션 것인지)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
# sliding_window_view: 배열을 복사하지 않고 "창문으로 들여다본 뷰" 를 만들어 주는 numpy 기능 (빠르고 메모리 절약)
from numpy.lib.stride_tricks import sliding_window_view

from logguard.data.windows import PAD, UNK, encode, make_lookup


def encode_sessions(df: pd.DataFrame, vocab: dict, add_eos: bool) -> list[np.ndarray]:
    """세션 테이블의 모든 세션을 모델 입력 번호 시퀀스로 바꾼다.

    Args:
        df     : train/val/test parquet 을 읽은 표 ("events" 열 = Drain Event ID 배열)
        vocab  : event_vocab.json 내용
        add_eos: True 면 각 시퀀스 끝에 EOS 를 붙인다 (최종 모델은 True)
    Returns:
        세션별 번호 배열의 리스트. 예: [array([3, 4, 3, 2]), array([3, 2]), ...]
    """
    lut = make_lookup(vocab["event_to_index"])     # 변환표를 한 번만 만들어 모든 세션에 재사용
    return [encode(seq, lut, add_eos=add_eos) for seq in df["events"]]


def unk_mask(seqs: list[np.ndarray]) -> np.ndarray:
    """학습 때 못 본 이벤트가 하나라도 있는 세션 → 모델과 무관하게 즉시 이상 (EDA 결론 #3).

    Returns:
        세션별 True/False 배열. True = UNK 가 하나 이상 있음 (이 세션은 점수를 +무한대로 만들어 무조건 이상)
    """
    # (s == UNK).any() : 세션 배열에 UNK(1) 가 하나라도 있으면 True
    # np.fromiter(…, count=…) : 생성기를 바로 numpy 배열로 (리스트를 거치지 않아 빠름)
    return np.fromiter(((s == UNK).any() for s in seqs), dtype=bool, count=len(seqs))


def _rows(seqs: list[np.ndarray], w: int, with_next: bool) -> tuple[np.ndarray, np.ndarray]:
    """모든 세션의 모든 윈도우를 하나의 큰 2차원 배열로 만든다 (중복 제거 전).

    한 행의 모양
        with_next=True  (DeepLog)     : [직전 w개 …, 현재 이벤트]  → 폭 w+1 (마지막 칸이 정답 y)
        with_next=False (Autoencoder) : [현재 이벤트로 끝나는 w개]   → 폭 w

    Returns:
        rows          : (전체 윈도우 수, 폭) int16 배열
        session_index : (전체 윈도우 수,) 각 행이 몇 번째 세션에서 나왔는지
    """
    width = w + 1 if with_next else w
    # 앞에 PAD 를 (폭-1)개 붙이면, 세션의 "첫 이벤트" 도 마지막 칸에 오는 윈도우가 생긴다.
    # → 세션 길이 = 그 세션에서 나오는 윈도우 수 (모든 이벤트가 정확히 한 번씩 판정 대상)
    pad = np.full(width - 1, PAD, dtype=np.int16)
    # int16(2바이트)으로 저장해 메모리 절약 (번호가 22 미만이라 충분)
    rows = [sliding_window_view(np.concatenate([pad, s.astype(np.int16)]), width) for s in seqs]
    # 세션 i 에서 나온 윈도우 수 = len(seqs[i]) → 세션 번호를 그만큼 반복
    session_index = np.repeat(np.arange(len(seqs)), [len(s) for s in seqs])
    return np.concatenate(rows), session_index


@dataclass
class WindowSet:
    """중복 제거된 윈도우 묶음 + 원래 위치로 되돌리는 정보.

    n_unique = 고유 윈도우 수 (예: 9만), n_total = 원래 전체 윈도우 수 (예: 687만)
    """
    X: np.ndarray                    # (n_unique, w)  int64   모델 입력 (고유 윈도우)
    y: np.ndarray | None             # (n_unique,)   다음 이벤트 (DeepLog) / None (AE)
    weight: np.ndarray               # (n_unique,)   등장 횟수 (학습 시 손실 가중치)
    inverse: np.ndarray              # (n_total,)    원래 윈도우 -> 고유 윈도우 인덱스
    session_index: np.ndarray        # (n_total,)    원래 윈도우 -> 세션 인덱스
    n_sessions: int                  # 세션 개수 (세션 점수 배열의 길이)

    @property
    def n_total(self) -> int:
        """원래(중복 포함) 윈도우 개수. @property 라서 괄호 없이 ws.n_total 로 쓴다."""
        return len(self.inverse)


def build_windows(seqs: list[np.ndarray], w: int, with_next: bool) -> WindowSet:
    """세션 시퀀스들 → 중복 제거된 WindowSet. 학습·평가 모두 이 함수를 쓴다.

    Args:
        seqs     : encode_sessions() 결과
        w        : 윈도우 크기 (최종 설정 20)
        with_next: True = DeepLog 용 (X, y 분리), False = Autoencoder 용 (X 만)
    """
    rows, session_index = _rows(seqs, w, with_next)
    # 행 전체를 하나의 바이트열로 보고 1차원 unique (axis=0 보다 수 배 빠름)
    #   np.unique(rows, axis=0) 는 행을 칸 단위로 비교해서 느리다(21초).
    #   대신 각 행(예: int16 × 21칸 = 42바이트)을 "42바이트짜리 덩어리 하나" 로 보이게 바꾸면
    #   1차원 값 비교가 되어 훨씬 빠르다(5.7초). 같은 행 ⇔ 같은 바이트열 이므로 결과는 동일.
    rows = np.ascontiguousarray(rows)       # 메모리에 연속으로 놓여 있어야 바이트 뷰를 만들 수 있다
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
    # np.unique 의 반환값
    #   first   : 각 고유값이 처음 나온 위치 (→ 그 위치의 행을 고유 윈도우로 사용)
    #   inverse : 원래 각 행이 몇 번째 고유값인지 (→ 나중에 점수를 원래 자리로 되돌릴 때 사용)
    #   counts  : 각 고유값이 몇 번 나왔는지 (→ 학습 가중치)
    _, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    uniq = rows[first].astype(np.int64)     # PyTorch 임베딩 입력은 int64(long) 여야 한다
    # DeepLog 면 마지막 칸을 정답 y 로 떼어 내고, AE 면 행 전체가 입력
    X, y = (uniq[:, :-1], uniq[:, -1]) if with_next else (uniq, None)
    return WindowSet(X=X, y=y, weight=counts.astype(np.float64), inverse=inverse.reshape(-1),
                     session_index=session_index, n_sessions=len(seqs))
