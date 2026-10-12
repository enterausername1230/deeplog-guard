"""윈도우 점수 → 세션 점수 → 임계치 선택 → 결과 저장. 모든 모델이 같은 규칙으로 평가된다.

규칙 (docs/plan_week4-6.md 공통 원칙)
  - 세션 점수 = 세션 내 윈도우 점수의 최댓값
  - UNK 세션 = +inf (즉시 이상)
  - 임계치 = val 정상 세션 점수의 p 백분위수, p 는 val F1 최대로 선택 (test 는 마지막에 1회)
  - 대시보드용 정규화 점수 = s / (s + threshold) → 임계치가 항상 0.5, 범위 0~1
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from logguard.eval.metrics import percentile_threshold, prf
from logguard.models.dataset import WindowSet


def session_scores(unique_scores: np.ndarray, ws: WindowSet, unk: np.ndarray | None = None) -> np.ndarray:
    out = np.full(ws.n_sessions, -np.inf)
    np.maximum.at(out, ws.session_index, np.asarray(unique_scores, dtype=float)[ws.inverse])
    if unk is not None:
        out[unk] = np.inf
    return out


def threshold_from_normal(scores: np.ndarray, labels: np.ndarray, q: float) -> float:
    normal = scores[(labels == 0) & np.isfinite(scores)]
    return percentile_threshold(normal, q)


def search_threshold(val_scores: np.ndarray, val_labels: np.ndarray, grid: list[float]) -> tuple[pd.DataFrame, dict]:
    """백분위수 후보별 val 성능. 반환: (전체 표, F1 최대 행)."""
    rows = []
    for q in grid:
        thr = threshold_from_normal(val_scores, val_labels, q)
        rows.append({"percentile": q, "threshold": thr, **prf(val_labels, val_scores > thr)})
    table = pd.DataFrame(rows).sort_values("f1", ascending=False, kind="stable")
    return table, table.iloc[0].to_dict()


def normalize(scores: np.ndarray | float, threshold: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if threshold <= 0:
        return (s > threshold).astype(float)
    with np.errstate(invalid="ignore"):
        out = np.where(np.isinf(s), 1.0, np.clip(s, 0, None) / (np.clip(s, 0, None) + threshold))
    return out


def save_result(cfg: dict, name: str, payload: dict) -> None:
    res_dir = cfg["paths"]["results_dir"]
    (res_dir / f"{name}.json").parent.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{name}.json").write_text(json.dumps({"model": name, **payload}, indent=2, ensure_ascii=False, default=float))


def save_test_scores(cfg: dict, name: str, labels: np.ndarray, scores: np.ndarray, mode: dict) -> None:
    """6단계 그래프(PR 곡선·점수 분포)용 test 세션 점수. 용량이 커서 git 에 넣지 않는다 (artifacts/eval)."""
    path = cfg["paths"]["artifacts_dir"] / "eval" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels, scores=scores,
                        threshold=mode.get("threshold", np.nan), mode=mode["mode"])
