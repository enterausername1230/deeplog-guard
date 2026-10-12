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

────────────────────────────────────────────────────────────────────────────────────────────
세 규칙을 예시로 이해하기
────────────────────────────────────────────────────────────────────────────────────────────
  R1 keyword  : "Exception" 이라는 단어가 들어간 로그가 한 번이라도 나오면 이상
                → 단순하지만 정상 블록에서도 흔한 경고("Got exception while serving")에 속아 헛경보가 많다 (F1 0.15)
  R2 length   : 정상 블록은 보통 이벤트 13~30개. 너무 짧게 끊기거나(예: 2개) 비정상적으로 길면 이상
                → 오탐은 거의 없지만 놓치는 게 많다 (F1 0.57, 최종 Rule-Based 로 선택됨)
  R3 frequency: 정상 블록에서 "Receiving block" 은 보통 3번 나오는데 10번 나왔다 → 이상
                → 꽤 강력하다 (F1 0.89). 딥러닝은 이것보다 좋아야 의미가 있다.

"p 백분위수" 가 뭔가?
  정상 학습 세션의 값을 작은 것부터 줄 세웠을 때 p% 위치의 값. p=99.9 면 "정상 세션 99.9% 는 이 값 이하".
  → 이 값을 넘으면 "정상에서는 거의 안 보이는 수준" 이므로 이상으로 본다.
"""
from __future__ import annotations

import argparse
import itertools     # 조합(combinations) 만들기: ("keyword","length") → (keyword), (length), (keyword,length)
import json
import re

import numpy as np
import pandas as pd

from logguard.config import load_config
from logguard.eval.metrics import prf

# 사용 가능한 규칙 3개의 이름
RULES = ("keyword", "length", "frequency")
# 베이스라인 이름 -> 사용할 규칙 후보
#   rule_based 는 R1, R2 중 val 에서 가장 좋은 조합을 고르고 (keyword 단독 / length 단독 / 둘 다)
#   statistical_frequency 는 R3 만 쓴다.
BASELINES = {
    "rule_based": ("keyword", "length"),
    "statistical_frequency": ("frequency",),
}


def error_event_ids(templates: pd.DataFrame, keywords: list[str], error_levels: list[str]) -> set[int]:
    """'오류 이벤트' 로 볼 Event ID 집합을 만든다 (규칙 R1 에 사용).

    오류 이벤트의 조건 (둘 중 하나만 만족해도 됨)
      ① 템플릿 문장에 오류 키워드(error, fail, exception …)가 들어 있음 (대소문자 무시)
      ② 그 이벤트가 WARN/ERROR/FATAL 레벨로 한 번이라도 찍힌 적이 있음
    """
    # 정규식 만들기: \b(error|fail|exception|…)\w*
    #   \b  : 단어의 시작 경계 → "failed", "Exception" 은 맞고, 단어 중간의 우연한 일치는 피한다
    #   \w* : 뒤에 이어지는 글자 허용 → "fail" 로 "failed", "failure" 까지 잡는다
    #   (주의: 그래서 예전에 "invalid" 를 넣었을 때 정상 이벤트 "invalidSet" 이 잘못 걸렸다 → 키워드에서 제외함)
    kw = re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\w*", re.IGNORECASE)
    lv = set(error_levels)
    ids = set()
    # itertuples(): 표를 한 행씩 꺼낸다. row.template, row.levels 처럼 열 이름으로 접근
    for row in templates.itertuples():
        levels = set(str(row.levels).split("|"))          # "INFO|WARN" → {"INFO", "WARN"}
        # kw.search(...) : 키워드가 있으면 결과 객체(참), 없으면 None(거짓)
        # levels & lv    : 두 집합의 교집합. 비어 있지 않으면(=오류 레벨로 찍힌 적 있으면) 참
        if kw.search(str(row.template)) or levels & lv:
            ids.add(int(row.event_id))
    return ids


def count_matrix(sessions: pd.DataFrame, n_events: int) -> np.ndarray:
    """(세션 수, 이벤트 수) 발생 횟수 행렬.

    결과 C[i, e] = i번째 세션에서 Event ID e 가 몇 번 나왔는지.
    예: 세션 0 = [1, 1, 2], n_events = 4 → C[0] = [0, 2, 1, 0]

    빠른 계산 요령 (파이썬 반복문 없이 한 번에)
      1) 모든 세션의 이벤트를 한 줄로 이어 붙인다 (flat)
      2) 각 이벤트가 몇 번째 세션 소속인지 적는다 (rows)
      3) "세션번호 × n_events + 이벤트번호" 라는 고유 칸 번호로 바꿔 np.bincount 로 칸별 개수를 센다
      4) 1차원 결과를 (세션 수, n_events) 2차원 표로 접는다
    """
    lengths = sessions["length"].to_numpy()                                   # 세션별 이벤트 개수
    flat = np.concatenate(sessions["events"].to_numpy()).astype(np.int64)    # 1) 전체 이벤트를 한 줄로
    rows = np.repeat(np.arange(len(sessions)), lengths)                      # 2) 예: [0,0,0,1,1,…]
    # 3) + 4)  minlength: 마지막 칸까지 0 으로 채워 정확히 (세션 수 × n_events) 길이가 되도록
    return np.bincount(rows * n_events + flat, minlength=len(sessions) * n_events).reshape(len(sessions), n_events)


class RuleBasedDetector:
    """규칙 3개(R1, R2, R3)를 계산하는 탐지기. 정상 학습 세션으로 fit() 한 뒤 rule_flags() 로 판정한다."""

    def __init__(self, error_ids: set[int], n_events: int):
        # 오류 이벤트 번호들을 정렬된 numpy 배열로 (행렬에서 해당 열만 골라 더할 때 사용)
        self.error_ids = np.asarray(sorted(error_ids), dtype=np.int64)
        self.n_events = n_events                         # Event ID 최댓값 + 1 (행렬 열 개수)
        self.train_counts: np.ndarray | None = None      # 정상 학습 세션의 발생 횟수 행렬 (fit 후 채워짐)
        self.train_lengths: np.ndarray | None = None     # 정상 학습 세션의 길이 배열 (fit 후 채워짐)

    def fit(self, train_normal: pd.DataFrame) -> "RuleBasedDetector":
        """정상 학습 세션의 통계(이벤트별 횟수, 세션 길이)를 저장한다. 임계치는 판정 때 백분위수로 계산."""
        self.train_counts = count_matrix(train_normal, self.n_events)
        self.train_lengths = train_normal["length"].to_numpy()
        return self      # self 를 돌려주면 `RuleBasedDetector(...).fit(...)` 처럼 한 줄로 쓸 수 있다

    def rule_flags(self, sessions: pd.DataFrame, q: float) -> dict[str, np.ndarray]:
        """세션마다 규칙 3개 각각에 걸렸는지(True/False) 계산한다.

        Args:
            sessions: 판정할 세션들 (val 또는 test)
            q       : 백분위수 (예: 99.9). 클수록 기준이 느슨해져 덜 걸린다.
        Returns:
            {"keyword": bool 배열, "length": bool 배열, "frequency": bool 배열}  (각 길이 = 세션 수)
        """
        counts = count_matrix(sessions, self.n_events)
        len_hi = np.percentile(self.train_lengths, q)          # 정상 길이 상한 (이보다 길면 이상)
        len_lo = np.percentile(self.train_lengths, 100 - q)    # 정상 길이 하한 (이보다 짧으면 이상)
        # axis=0: 이벤트(열)마다 따로 백분위수 → 이벤트별 횟수 임계치 배열
        #   정상 학습에 한 번도 안 나온 이벤트는 임계치가 0 → 한 번만 나와도 0 보다 커서 이상
        freq_th = np.percentile(self.train_counts, q, axis=0)
        return {
            # R1: 오류 이벤트 열들의 횟수 합이 1 이상이면 이상 (오류 이벤트 목록이 비어 있으면 전부 False)
            "keyword": counts[:, self.error_ids].sum(axis=1) > 0 if len(self.error_ids) else np.zeros(len(sessions), bool),
            # R2: 너무 길거나(|) 너무 짧으면 이상
            "length": (sessions["length"].to_numpy() > len_hi) | (sessions["length"].to_numpy() < len_lo),
            # R3: 어떤 이벤트든 하나라도(.any) 임계치를 넘으면 이상
            "frequency": (counts > freq_th).any(axis=1),
        }

    @staticmethod
    def combine(flags: dict[str, np.ndarray], rules: tuple[str, ...]) -> np.ndarray:
        """여러 규칙을 OR 로 합친다: 고른 규칙 중 "하나라도" 걸리면 이상.

        @staticmethod: self(객체 정보)가 필요 없는 함수라는 표시. RuleBasedDetector.combine(...) 으로 호출.
        """
        # np.logical_or.reduce([a, b, c]) = a | b | c
        return np.logical_or.reduce([flags[r] for r in rules])


def run(cfg: dict) -> dict[str, dict]:
    """베이스라인 2종을 만들고 val 로 설정을 골라 test 성능까지 저장한다."""
    proc, interim, res_dir = cfg["paths"]["processed_dir"], cfg["paths"]["interim_dir"], cfg["paths"]["results_dir"]
    bcfg = cfg["baseline"]
    templates = pd.read_csv(interim / "hdfs_templates.csv")
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    n_events = int(templates["event_id"].max()) + 1          # 행렬 열 개수 (Event ID 가 0~최댓값)
    err_ids = error_event_ids(templates, bcfg["keywords"], bcfg["error_levels"])
    # 정상 학습 세션으로만 통계를 낸다 (train 은 이미 정상만 있지만 안전하게 한 번 더 거른다)
    det = RuleBasedDetector(err_ids, n_events).fit(train[train["label"] == 0])

    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    # 백분위수 후보마다 val 의 규칙 판정을 미리 계산해 둔다 (조합마다 다시 계산하지 않도록)
    val_flags = {q: det.rule_flags(val, q) for q in bcfg["percentile_grid"]}
    res_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, rule_pool in BASELINES.items():
        # 규칙 후보로 만들 수 있는 모든 조합 (공집합 제외)
        #   ("keyword","length") → [("keyword",), ("length",), ("keyword","length")]
        combos = [c for k in range(1, len(rule_pool) + 1) for c in itertools.combinations(rule_pool, k)]
        # (백분위수 × 조합) 모든 경우의 val 성능표
        grid = [{"percentile": q, "rules": "+".join(rules), **prf(y_val, det.combine(flags, rules))}
                for q, flags in val_flags.items() for rules in combos]
        # F1 이 가장 높은 설정을 고른다. kind="stable": 동점이면 원래 순서(먼저 나온 것)를 유지 → 결과가 항상 같음
        grid_df = pd.DataFrame(grid).sort_values("f1", ascending=False, kind="stable")
        best = grid_df.iloc[0]
        best_q, best_rules = float(best["percentile"]), tuple(best["rules"].split("+"))

        # val 로 고른 설정을 그대로 test 에 적용 (test 를 보고 설정을 바꾸지 않는다)
        test_flags = det.rule_flags(test, best_q)
        test_metrics = prf(y_test, det.combine(test_flags, best_rules))
        result = {
            "model": name,
            "selected_on": "val_f1",
            "best_config": {"percentile": best_q, "rules": list(best_rules)},
            "val": {k: best[k] for k in ("precision", "recall", "f1")},
            "test": test_metrics,
            # 참고: 규칙 하나씩만 썼을 때의 test 성능 (어느 규칙이 효과가 있는지 해석용)
            "test_per_rule": {r: prf(y_test, test_flags[r]) for r in rule_pool},
            "val_grid_top10": grid_df.head(10).to_dict(orient="records"),
        }
        if "keyword" in rule_pool:
            # 키워드 규칙을 쓰는 경우, 어떤 이벤트가 "오류 이벤트" 로 잡혔는지 기록 (검토용)
            result["error_event_ids"] = sorted(int(e) for e in err_ids)
            result["error_templates"] = templates[templates["event_id"].isin(err_ids)]["template"].tolist()
        # default=float : numpy 숫자 타입처럼 json 이 모르는 값을 float 로 바꿔 저장
        (res_dir / f"baseline_{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
        results[name] = result

        print(f"[{name}] best on val: rules={'+'.join(best_rules)}, p={best_q}")
        print(f"[{name}] TEST  P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}  F1={test_metrics['f1']:.4f}")

    return results


def main() -> None:
    """명령줄 진입점 (`make baseline`)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
