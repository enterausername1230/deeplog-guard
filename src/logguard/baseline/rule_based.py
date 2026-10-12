"""베이스라인 2종: Rule-Based (계획서 목표 ② 비교 기준) + 통계 빈도 (참고용).

세션(Block) 단위 규칙:
  R1 keyword   : 오류 이벤트(템플릿에 Error/Failed/Exception 등 키워드, 또는 WARN/ERROR 레벨)가 1회 이상 발생
  R2 length    : 세션 길이 > 정상 학습 세션 길이의 p 백분위수, 또는 < (100-p) 백분위수
                 (이상 세션의 약 37%는 정상 최소 길이 13 보다 짧게 끊긴 세션)
  R3 frequency : 어떤 이벤트든 세션 내 발생 횟수 > 정상 학습 세션에서 그 이벤트 횟수의 p 백분위수
                 (정상 학습 세션에 한 번도 없던 이벤트는 임계치 0 → 등장만 해도 이상)

베이스라인 구성 (2026-10-12 팀 결정: 둘 다 보고)
  rule_based            : R1, R2 조합 — 운영 현장의 키워드·정규식 알람. 목표 ② '+10%p' 비교 대상
  statistical_frequency : R3 — 이벤트별 빈도를 정상 데이터에서 학습하는 통계 기법. HDFS 에서 매우 강해서
                          (test F1≈0.89) 딥러닝 모델이 넘어야 할 실질적 기준으로 함께 보고

각 베이스라인은 규칙 조합 × 백분위수 후보 중 val F1 이 최대인 설정을 골라 test 에서 평가한다.
(베이스라인에 가장 유리한 설정으로 비교해야 공정하다.)

입력 : data/processed/{train,val,test}.parquet, data/interim/hdfs_templates.csv
출력 : results/baseline_rule_based.json, results/baseline_statistical_frequency.json

사용법: python -m logguard.baseline.rule_based
"""
from __future__ import annotations

import argparse
import itertools
import json
import re

import numpy as np
import pandas as pd

from logguard.config import load_config
from logguard.eval.metrics import prf

RULES = ("keyword", "length", "frequency")
# 베이스라인 이름 -> 사용할 규칙 후보
BASELINES = {
    "rule_based": ("keyword", "length"),
    "statistical_frequency": ("frequency",),
}


def error_event_ids(templates: pd.DataFrame, keywords: list[str], error_levels: list[str]) -> set[int]:
    kw = re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\w*", re.IGNORECASE)
    lv = set(error_levels)
    ids = set()
    for row in templates.itertuples():
        levels = set(str(row.levels).split("|"))
        if kw.search(str(row.template)) or levels & lv:
            ids.add(int(row.event_id))
    return ids


def count_matrix(sessions: pd.DataFrame, n_events: int) -> np.ndarray:
    """(세션 수, 이벤트 수) 발생 횟수 행렬."""
    lengths = sessions["length"].to_numpy()
    flat = np.concatenate(sessions["events"].to_numpy()).astype(np.int64)
    rows = np.repeat(np.arange(len(sessions)), lengths)
    return np.bincount(rows * n_events + flat, minlength=len(sessions) * n_events).reshape(len(sessions), n_events)


class RuleBasedDetector:
    def __init__(self, error_ids: set[int], n_events: int):
        self.error_ids = np.asarray(sorted(error_ids), dtype=np.int64)
        self.n_events = n_events
        self.train_counts: np.ndarray | None = None
        self.train_lengths: np.ndarray | None = None

    def fit(self, train_normal: pd.DataFrame) -> "RuleBasedDetector":
        self.train_counts = count_matrix(train_normal, self.n_events)
        self.train_lengths = train_normal["length"].to_numpy()
        return self

    def rule_flags(self, sessions: pd.DataFrame, q: float) -> dict[str, np.ndarray]:
        counts = count_matrix(sessions, self.n_events)
        len_hi = np.percentile(self.train_lengths, q)
        len_lo = np.percentile(self.train_lengths, 100 - q)
        freq_th = np.percentile(self.train_counts, q, axis=0)
        return {
            "keyword": counts[:, self.error_ids].sum(axis=1) > 0 if len(self.error_ids) else np.zeros(len(sessions), bool),
            "length": (sessions["length"].to_numpy() > len_hi) | (sessions["length"].to_numpy() < len_lo),
            "frequency": (counts > freq_th).any(axis=1),
        }

    @staticmethod
    def combine(flags: dict[str, np.ndarray], rules: tuple[str, ...]) -> np.ndarray:
        return np.logical_or.reduce([flags[r] for r in rules])


def run(cfg: dict) -> dict[str, dict]:
    proc, interim, res_dir = cfg["paths"]["processed_dir"], cfg["paths"]["interim_dir"], cfg["paths"]["results_dir"]
    bcfg = cfg["baseline"]
    templates = pd.read_csv(interim / "hdfs_templates.csv")
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    n_events = int(templates["event_id"].max()) + 1
    err_ids = error_event_ids(templates, bcfg["keywords"], bcfg["error_levels"])
    det = RuleBasedDetector(err_ids, n_events).fit(train[train["label"] == 0])

    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    val_flags = {q: det.rule_flags(val, q) for q in bcfg["percentile_grid"]}
    res_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, rule_pool in BASELINES.items():
        combos = [c for k in range(1, len(rule_pool) + 1) for c in itertools.combinations(rule_pool, k)]
        grid = [{"percentile": q, "rules": "+".join(rules), **prf(y_val, det.combine(flags, rules))}
                for q, flags in val_flags.items() for rules in combos]
        grid_df = pd.DataFrame(grid).sort_values("f1", ascending=False, kind="stable")
        best = grid_df.iloc[0]
        best_q, best_rules = float(best["percentile"]), tuple(best["rules"].split("+"))

        test_flags = det.rule_flags(test, best_q)
        test_metrics = prf(y_test, det.combine(test_flags, best_rules))
        result = {
            "model": name,
            "selected_on": "val_f1",
            "best_config": {"percentile": best_q, "rules": list(best_rules)},
            "val": {k: best[k] for k in ("precision", "recall", "f1")},
            "test": test_metrics,
            "test_per_rule": {r: prf(y_test, test_flags[r]) for r in rule_pool},
            "val_grid_top10": grid_df.head(10).to_dict(orient="records"),
        }
        if "keyword" in rule_pool:
            result["error_event_ids"] = sorted(int(e) for e in err_ids)
            result["error_templates"] = templates[templates["event_id"].isin(err_ids)]["template"].tolist()
        (res_dir / f"baseline_{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
        results[name] = result

        print(f"[{name}] best on val: rules={'+'.join(best_rules)}, p={best_q}")
        print(f"[{name}] TEST  P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}  F1={test_metrics['f1']:.4f}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
