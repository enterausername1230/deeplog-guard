"""세션(이벤트 시퀀스)을 모델 입력용 슬라이딩 윈도우로 바꾸는 유틸.

- DeepLog : (윈도우 w개 이벤트) -> 다음 이벤트 예측   => next_event_windows()
- Autoencoder : 윈도우 자체를 압축·복원            => sliding_windows()
세션 내부에서만 윈도우를 만들므로 블록 간 섞임(누수)이 없다.

이 파일의 핵심 개념 3가지
    1) 어휘(vocab) 인덱스
       Drain 이 붙인 Event ID(1, 2, 3, … 최대 수십)는 그대로 모델에 넣지 않는다.
       학습 데이터에 실제로 나온 이벤트에만 3부터 순서대로 번호를 새로 붙인다 (split.py 의 event_vocab.json).
       0, 1, 2 는 아래의 특수 토큰이 쓰기 때문이다.

    2) 특수 토큰
       PAD (0) : "빈칸". 세션 앞부분처럼 앞에 이벤트가 부족할 때 윈도우를 채우는 용도
       UNK (1) : "모르는 이벤트". 학습 때 한 번도 못 본 이벤트 → 이상의 강력한 신호 (즉시 이상 판정)
       EOS (2) : "세션 끝". 시퀀스 맨 뒤에 붙여서, 모델이 "여기서 끝나도 되는가?" 까지 배우게 한다.
                 EDA 에서 이상 세션의 약 17%는 정상 시퀀스의 앞부분에서 그냥 끊긴 경우였는데,
                 EOS 가 없으면 이런 세션은 원리적으로 못 잡는다. EOS 추가로 test F1 이 약 +9~10%p 올랐다.

    3) 슬라이딩 윈도우
       시퀀스 [a, b, c, d, e] 를 길이 3 창문으로 한 칸씩 밀며 자르면
       [a,b,c], [b,c,d], [c,d,e] 가 된다. 모델은 이 짧은 조각 단위로 "평소 패턴인지" 를 판단한다.

참고: 실제 학습·평가에서는 더 빠른 models/dataset.py 의 build_windows() 를 쓴다.
      이 파일의 sliding_windows()/next_event_windows() 는 같은 개념의 단순한 버전(테스트·설명용)이다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 특수 토큰 (프로젝트 전체에서 이 값만 사용)
#   다른 파일은 숫자 0, 1, 2 를 직접 쓰지 말고 반드시 이 이름(PAD/UNK/EOS)을 import 해서 쓴다.
#   그래야 나중에 번호를 바꿔도 이 한 곳만 고치면 된다.
PAD, UNK, EOS = 0, 1, 2          # EOS: 세션 종료. 정상 시퀀스 앞부분에서 끊긴 이상 세션 탐지용 (EDA 결론 #4)
NUM_SPECIAL = 3                  # 특수 토큰 개수. 실제 이벤트 번호는 이 값(3)부터 시작한다


def load_vocab(path: str | Path) -> dict:
    """event_vocab.json(어휘 파일)을 읽어 dict 로 돌려준다.

    파일 내용 예:
        {"pad_index": 0, "unk_index": 1, "eos_index": 2,
         "event_to_index": {"1": 3, "2": 4, ...},      ← Drain Event ID(문자열) → 모델 입력 번호
         "index_to_template": {"3": "Receiving block <:BLK:> ...", ...},
         "num_classes": 22}                             ← 모델 출력 클래스 수 (특수 3 + 이벤트 19)
    """
    return json.loads(Path(path).read_text())


def make_lookup(event_to_index: dict[str, int], max_event_id: int = 4096) -> np.ndarray:
    """event_id -> 어휘 인덱스 배열. 어휘에 없는 id 는 UNK.

    왜 dict 대신 배열(lookup table)을 쓰나?
        세션 수십만 개, 이벤트 수백만 개를 변환해야 한다. 파이썬 dict 로 하나씩 찾으면 느리지만,
        배열을 만들어 두면 `lut[이벤트배열]` 한 번으로 전부 변환된다 (numpy 의 고속 인덱싱).

    예: event_to_index = {"5": 3, "9": 4}
        → lut[5] = 3, lut[9] = 4, 나머지 칸은 전부 UNK(1)
    """
    # 길이 4097 짜리 배열을 전부 UNK 로 채운다 (= 기본값은 "모르는 이벤트")
    lut = np.full(max_event_id + 1, UNK, dtype=np.int64)
    # 어휘에 있는 이벤트만 해당 칸에 올바른 번호를 써 넣는다. json 키는 문자열이라 int() 로 바꾼다.
    for e, i in event_to_index.items():
        lut[int(e)] = i
    return lut


def encode(events: np.ndarray, event_to_index: dict[str, int] | np.ndarray, add_eos: bool = False) -> np.ndarray:
    """원본 event_id 시퀀스 -> 어휘 인덱스 (학습 때 못 본 이벤트는 UNK). add_eos 면 끝에 EOS.

    Args:
        events        : Drain Event ID 시퀀스. 예: [1, 2, 1, 3]
        event_to_index: 어휘 dict 또는 make_lookup() 으로 미리 만든 배열 (여러 번 호출 시 배열이 빠름)
        add_eos       : True 면 결과 맨 끝에 EOS(2)를 하나 붙인다 (세션이 끝났다는 표시)

    Returns:
        모델 입력 번호 배열. 예: [3, 4, 3, 5] (+ add_eos 면 [3, 4, 3, 5, 2])
    """
    # 이미 배열(lookup table)이면 그대로 쓰고, dict 면 배열로 만든다.
    lut = event_to_index if isinstance(event_to_index, np.ndarray) else make_lookup(event_to_index)
    ev = np.asarray(events, dtype=np.int64)
    # 안전장치: lookup 배열 길이(4097)보다 큰 Event ID 가 들어오면 lut[ev] 가 범위 초과 오류를 낸다.
    #   np.minimum(ev, len(lut) - 1) : 인덱스를 배열 범위 안으로 눌러서 오류를 막고,
    #   np.where(ev < len(lut), …, UNK) : 원래 범위 밖이던 값은 결과를 UNK 로 바꾼다.
    #   (온라인 파서는 모르는 로그를 매우 큰 값 10**9 로 표시하므로 이 처리로 UNK 가 된다)
    out = np.where(ev < len(lut), lut[np.minimum(ev, len(lut) - 1)], UNK)
    # np.append(out, EOS) : 배열 끝에 EOS 하나 추가
    return np.append(out, EOS) if add_eos else out


def pad_left(seq: np.ndarray, size: int) -> np.ndarray:
    """시퀀스가 size 보다 짧으면 왼쪽(앞)을 PAD 로 채워 길이를 size 로 맞춘다.

    예: pad_left([5, 6], 4) → [PAD, PAD, 5, 6] = [0, 0, 5, 6]
    이미 충분히 길면 그대로 돌려준다.
    왼쪽을 채우는 이유: 윈도우의 "마지막 칸" 이 실제 최신 이벤트가 되도록 하기 위해서다.
    """
    if len(seq) >= size:
        return seq
    # np.full(개수, 값) : 값으로 채운 배열 / np.concatenate : 배열 이어 붙이기
    return np.concatenate([np.full(size - len(seq), PAD, dtype=seq.dtype), seq])


def sliding_windows(seq: np.ndarray, size: int, stride: int = 1) -> np.ndarray:
    """길이 size 의 윈도우들 (n_windows, size). 세션이 짧으면 왼쪽 PAD 로 1개 생성.

    Autoencoder 개념 설명용: 윈도우 자체가 입력이자 복원 목표다.

    예: sliding_windows([5, 6, 7], size=2) → [[5, 6], [6, 7]]
        sliding_windows([5, 6, 7], size=5) → [[0, 0, 5, 6, 7]]   (짧아서 PAD 로 채운 1개)

    Args:
        stride: 창문을 한 번에 몇 칸씩 밀지. 1 이면 모든 위치에서 윈도우를 만든다.
    """
    seq = pad_left(np.asarray(seq), size)
    # 윈도우 시작 위치들: 0, stride, 2*stride, … (마지막 윈도우가 시퀀스 끝을 넘지 않는 데까지)
    starts = range(0, len(seq) - size + 1, stride)
    # 각 시작 위치에서 size 개씩 잘라 행(row)으로 쌓는다 → 2차원 배열 (윈도우 수, size)
    return np.stack([seq[s:s + size] for s in starts])


def next_event_windows(seq: np.ndarray, size: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """DeepLog 학습/추론용 (X: (n, size), y: (n,)). 세션 첫 이벤트부터 예측하도록 왼쪽 PAD.

    DeepLog 는 "직전 size 개 이벤트(X)를 보고 다음 이벤트(y)를 맞히는" 문제를 푼다.

    예: next_event_windows([5, 6, 7], size=2)
        → X = [[PAD, PAD], [PAD, 5], [5, 6]]
          y = [   5,          6,        7  ]
        즉 "아무것도 없을 때 첫 이벤트는 5", "5 다음은 6", "5,6 다음은 7" 을 학습한다.
        앞에 PAD 를 size 개 붙였기 때문에 세션의 "첫 이벤트" 부터 예측 대상이 된다.
    """
    # 앞에 PAD 를 size 개 붙인다: [PAD]*size + seq
    seq = np.concatenate([np.full(size, PAD, dtype=np.int64), np.asarray(seq, dtype=np.int64)])
    # 예측 대상 위치: PAD 다음(= 원래 시퀀스의 첫 이벤트)부터 끝까지
    idx = range(size, len(seq), stride)
    X = np.stack([seq[i - size:i] for i in idx])               # 각 위치 직전 size 개 = 입력
    y = np.asarray([seq[i] for i in idx], dtype=np.int64)      # 그 위치의 실제 이벤트 = 정답
    return X, y
