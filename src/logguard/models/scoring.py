"""윈도우 점수 → 세션 점수 → 임계치 선택 → 결과 저장. 모든 모델이 같은 규칙으로 평가된다.

규칙 (docs/plan_week4-6.md 공통 원칙)
  - 세션 점수 = 세션 내 윈도우 점수의 최댓값
  - UNK 세션 = +inf (즉시 이상)
  - 임계치 = val 정상 세션 점수의 p 백분위수, p 는 val F1 최대로 선택 (test 는 마지막에 1회)
  - 대시보드용 정규화 점수 = s / (s + threshold) → 임계치가 항상 0.5, 범위 0~1

전체 흐름 예시
    모델이 윈도우마다 점수를 낸다 (클수록 이상):  세션A 의 윈도우 점수 = [0.1, 0.3, 8.2, 0.2]
    → 세션 점수 = 최댓값 8.2 ("가장 이상한 순간" 이 세션을 대표)
    → val 정상 세션 점수들의 99.9 백분위수가 6.78 이었다면 임계치 = 6.78
    → 8.2 > 6.78 이므로 세션A 는 이상
    → 대시보드 점수 = 8.2 / (8.2 + 6.78) = 0.547  (0.5 를 넘으면 이상)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from logguard.eval.metrics import percentile_threshold, prf
from logguard.models.dataset import WindowSet


def session_scores(unique_scores: np.ndarray, ws: WindowSet, unk: np.ndarray | None = None) -> np.ndarray:
    """고유 윈도우 점수 → 세션 점수(세션 내 최댓값). UNK 세션은 +무한대.

    Args:
        unique_scores: 고유 윈도우별 점수 (길이 = len(ws.X))
        ws           : build_windows() 결과 (원래 위치·세션 정보 포함)
        unk          : 세션별 UNK 포함 여부 (unk_mask 결과). 주면 그 세션들의 점수를 +inf 로 덮어쓴다.
    Returns:
        세션별 점수 배열 (길이 = 세션 수)
    """
    # 모든 세션의 점수를 -무한대로 시작 (어떤 실제 점수보다도 작으므로 최댓값 계산의 출발점으로 적합)
    out = np.full(ws.n_sessions, -np.inf)
    # unique_scores[ws.inverse] : 고유 윈도우 점수를 원래 수백만 개 윈도우 자리로 복사
    # np.maximum.at(out, 세션번호들, 점수들) : 각 세션 칸에 "지금 값과 새 점수 중 큰 값" 을 반복 적용
    #   → 결과적으로 out[i] = 세션 i 에 속한 윈도우 점수들의 최댓값
    np.maximum.at(out, ws.session_index, np.asarray(unique_scores, dtype=float)[ws.inverse])
    if unk is not None:
        out[unk] = np.inf      # 학습 때 못 본 이벤트가 있는 세션은 어떤 임계치보다도 크게 → 무조건 이상
    return out


def threshold_from_normal(scores: np.ndarray, labels: np.ndarray, q: float) -> float:
    """정상 세션(label 0)의 점수만 골라 q 백분위수를 임계치로 계산한다.

    np.isfinite 로 +inf(UNK 세션)를 제외한다. 무한대가 섞이면 백분위수 계산이 망가지기 때문이다.
    (val 정상 세션에는 실제로 UNK 가 없지만 안전장치로 둔다)
    """
    normal = scores[(labels == 0) & np.isfinite(scores)]
    return percentile_threshold(normal, q)


def search_threshold(val_scores: np.ndarray, val_labels: np.ndarray, grid: list[float]) -> tuple[pd.DataFrame, dict]:
    """백분위수 후보별 val 성능. 반환: (전체 표, F1 최대 행).

    예: grid = [99.0, 99.05, …, 99.95, 99.99, 100.0] (config.yaml threshold.percentile_grid)
        각 후보로 임계치를 만들고 val 전체(정상+이상)에 적용해 F1 을 계산한 뒤, F1 이 가장 높은 후보를 고른다.
    """
    rows = []
    for q in grid:
        thr = threshold_from_normal(val_scores, val_labels, q)
        # val_scores > thr : 임계치보다 "큰" 점수 = 이상으로 판정
        rows.append({"percentile": q, "threshold": thr, **prf(val_labels, val_scores > thr)})
    # kind="stable": F1 동점이면 grid 에서 먼저 나온 후보를 우선 → 실행할 때마다 같은 결과
    table = pd.DataFrame(rows).sort_values("f1", ascending=False, kind="stable")
    return table, table.iloc[0].to_dict()


def normalize(scores: np.ndarray | float, threshold: float) -> np.ndarray:
    """모델마다 단위가 다른 점수를 0~1 로 바꾼다. 임계치가 항상 정확히 0.5 에 오도록 만든다.

    공식: s / (s + threshold)
        s = 0          → 0.0   (완전 정상)
        s = threshold  → 0.5   (경계선)
        s → 무한대      → 1.0   (매우 이상)
    덕분에 대시보드는 모델이 무엇이든 "0.5 선" 하나로 이상을 표시할 수 있다.
    """
    s = np.asarray(scores, dtype=float)
    if threshold <= 0:
        # 임계치가 0 이하이면 공식이 성립하지 않으므로(0 으로 나누기 등) 0/1 판정값만 돌려준다
        return (s > threshold).astype(float)
    # errstate(invalid="ignore"): inf/inf 같은 계산에서 나오는 경고 메시지를 숨긴다 (결과는 아래 where 가 처리)
    with np.errstate(invalid="ignore"):
        # np.clip(s, 0, None): 음수 점수는 0 으로 (점수는 원래 0 이상이지만 안전장치)
        # np.where(조건, A, B): 무한대(UNK)면 1.0, 아니면 공식 값
        out = np.where(np.isinf(s), 1.0, np.clip(s, 0, None) / (np.clip(s, 0, None) + threshold))
    return out


def save_result(cfg: dict, name: str, payload: dict) -> None:
    """실험 결과를 results/<name>.json 으로 저장한다. name 에 "final/xxx" 처럼 폴더를 넣을 수 있다."""
    res_dir = cfg["paths"]["results_dir"]
    (res_dir / f"{name}.json").parent.mkdir(parents=True, exist_ok=True)    # 하위 폴더가 없으면 만든다
    # {"model": name, **payload} : 맨 앞에 모델 이름을 넣고 나머지 내용을 펼쳐서 합친다
    (res_dir / f"{name}.json").write_text(json.dumps({"model": name, **payload}, indent=2, ensure_ascii=False, default=float))


def save_test_scores(cfg: dict, name: str, labels: np.ndarray, scores: np.ndarray, mode: dict) -> None:
    """6단계 그래프(PR 곡선·점수 분포)용 test 세션 점수. 용량이 커서 git 에 넣지 않는다 (artifacts/eval).

    npz = numpy 배열 여러 개를 한 파일에 압축 저장하는 형식. np.load(경로)["scores"] 처럼 꺼내 쓴다.
    """
    path = cfg["paths"]["artifacts_dir"] / "eval" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels, scores=scores,
                        threshold=mode.get("threshold", np.nan), mode=mode["mode"])
