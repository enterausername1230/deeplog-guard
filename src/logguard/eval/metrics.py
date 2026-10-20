"""계획서 Ⅱ-4 평가 지표. 모든 모델과 베이스라인은 세션(Block) 단위로 이 함수로 평가한다.

왜 공통 함수인가?
    모델마다 지표를 따로 계산하면 계산 방식이 미묘하게 달라질 수 있다 (예: 반올림, 0 나누기 처리).
    모든 방법(Rule-Based, 통계 빈도, DeepLog, Autoencoder)이 이 파일의 prf() 하나로 평가되므로
    결과표의 숫자들을 서로 공정하게 비교할 수 있다.

왜 Accuracy(정확도) 를 안 쓰나?
    HDFS 데이터는 이상 세션이 2.9% 뿐이다. "전부 정상" 이라고만 답해도 Accuracy 는 97.1% 가 나온다.
    그래서 이상을 얼마나 잘 잡는지 보여 주는 Precision / Recall / F1 을 쓴다 (계획서 제한조건 ①).
"""
from __future__ import annotations

import numpy as np   # 배열 단위 계산(수십만 개 세션을 한 번에)을 위한 수치 계산 라이브러리


def prf(y_true, y_pred) -> dict:
    """정답과 예측을 비교해서 Precision / Recall / F1 과 혼동행렬 4칸(TP, FP, FN, TN)을 계산한다.

    Args:
        y_true: 정답 라벨 배열. 1(또는 True) = 이상, 0(또는 False) = 정상. 길이 = 세션 수
        y_pred: 모델 예측 배열. 같은 규칙(1 = 이상이라고 판정). y_true 와 길이가 같아야 한다.

    Returns:
        {"precision", "recall", "f1"} : 소수점 4자리로 반올림한 지표
        {"tp", "fp", "fn", "tn"}      : 각 칸에 해당하는 세션 개수 (결과 해석·디버깅용)

    예:
        prf([1, 1, 0, 0], [1, 0, 1, 0])
        → TP=1(첫 번째), FN=1(두 번째 놓침), FP=1(세 번째 헛경보), TN=1(네 번째)
        → precision=0.5, recall=0.5, f1=0.5
    """
    # 입력이 리스트든 pandas Series 든 0/1 이든 True/False 든 상관없이
    # 전부 numpy 의 bool 배열(True/False)로 통일한다. 그래야 아래의 &, ~ 연산을 쓸 수 있다.
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    # 혼동행렬 4칸 계산 (& = 그리고, ~ = 반대)
    #   y_true & y_pred   : 진짜 이상이고, 이상이라고 예측 → TP
    #   ~y_true & y_pred  : 진짜 정상인데, 이상이라고 예측 → FP (헛경보)
    #   y_true & ~y_pred  : 진짜 이상인데, 정상이라고 예측 → FN (놓침)
    #   ~y_true & ~y_pred : 진짜 정상이고, 정상이라고 예측 → TN
    # .sum() 은 True 의 개수를 센다. int() 로 감싸 JSON 저장 시 numpy 타입 문제를 피한다.
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())

    # 분모가 0 이면(예: 아무것도 이상이라고 예측하지 않음) 0 으로 나누기 오류가 나므로 0.0 으로 처리한다.
    # `A if 조건 else B` : 조건이 참이면 A, 거짓이면 B (파이썬의 한 줄 if)
    precision = tp / (tp + fp) if tp + fp else 0.0     # 이상이라고 한 것 중 진짜 이상의 비율
    recall = tp / (tp + fn) if tp + fn else 0.0        # 진짜 이상 중 잡아낸 비율
    # F1 = Precision 과 Recall 의 조화평균. 둘 중 하나라도 낮으면 F1 도 크게 낮아진다.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def percentile_threshold(normal_scores, q: float) -> float:
    """정상 검증셋 이상 스코어 분포의 q 백분위수 -> 이상 판정 임계치 (계획서 독창성 항목).

    아이디어
        "정상 데이터 점수의 상위 몇 % 보다 높으면 이상" 이라고 정한다.
        예) q = 99.9 → 정상 세션 중 99.9% 는 이 점수 이하.
            → 이 값을 임계치로 쓰면 정상 세션이 이상으로 잘못 판정될 비율(오탐률)이 약 0.1% 가 된다.
        임계치를 사람이 감으로 정하지 않고, 데이터 분포에서 계산하므로 데이터가 바뀌어도 자동으로 맞춰진다.

    Args:
        normal_scores: 정상 세션들의 이상 점수 배열 (클수록 이상하다는 뜻)
        q: 백분위수 (0~100). 예: 99.5 → 상위 0.5% 지점

    Returns:
        임계치 값(float). 이 값보다 "큰" 점수를 받은 세션을 이상으로 판정한다.
    """
    # np.percentile(배열, q) : 배열을 크기순으로 줄 세웠을 때 q% 위치의 값
    # dtype=float 로 바꾸는 이유: 정수 배열이 들어와도 보간 계산이 정확히 되도록
    return float(np.percentile(np.asarray(normal_scores, dtype=float), q))
