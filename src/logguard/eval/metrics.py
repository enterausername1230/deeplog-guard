"""계획서 Ⅱ-4 평가 지표. 모든 모델과 베이스라인은 세션(Block) 단위로 이 함수로 평가한다."""
from __future__ import annotations

import numpy as np


def prf(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def percentile_threshold(normal_scores, q: float) -> float:
    """정상 검증셋 이상 스코어 분포의 q 백분위수 -> 이상 판정 임계치 (계획서 독창성 항목)."""
    return float(np.percentile(np.asarray(normal_scores, dtype=float), q))
