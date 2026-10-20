"""5단계 튜닝: 하이퍼파라미터 그리드 × 시드 → val F1 평균으로 설정 선택 (test 는 보지 않음).

그리드는 configs/config.yaml 의 tuning.<model> 에 정의. 각 조합을 tuning.seeds 의 시드로 학습하고,
평균 val F1 이 가장 높은 조합을 고른다 (같으면 학습 시간이 짧은 쪽).

출력 : results/tuning_<model>.csv          (조합 × 시드 전체)
       results/tuning_<model>_summary.csv  (조합별 평균 ± 표준편차, 정렬)

사용법: python -m logguard.models.tune --model deeplog

용어
    하이퍼파라미터: 학습으로 정해지지 않고 사람이 미리 정하는 설정값 (윈도우 크기, 은닉층 크기 등)
    그리드 탐색   : 후보값들의 "모든 조합" 을 다 시험해 보는 방법
        예) window_size [10, 15, 20] × hidden_size [64, 128] = 6조합, 시드 3개씩 → 18번 학습

5단계 결과
    DeepLog     : w=20, hidden 128 → val F1 0.927 ± 0.001
    Autoencoder : w=20, 잠재 32    → val F1 0.975 ± 0.003  ← 선택되어 config.yaml 에 반영됨
"""
from __future__ import annotations

import argparse
import copy
import importlib
import itertools        # itertools.product: 여러 목록의 모든 조합 (데카르트 곱)
import json
import time

import pandas as pd

from logguard.config import load_config


def run(cfg: dict, model: str) -> pd.DataFrame:
    """그리드의 모든 조합 × 모든 시드로 학습·val 평가를 하고, 조합별 평균 성능 순으로 정렬한 요약표를 돌려준다."""
    mod = importlib.import_module(f"logguard.models.{model}")
    grid = cfg["tuning"][model]                  # 예: {"window_size": [10,15,20], "hidden_size": [64,128]}
    seeds = cfg["tuning"]["seeds"]               # 예: [42, 43, 44]
    keys = list(grid)                            # ["window_size", "hidden_size"]
    # product(*grid.values()) → (10,64), (10,128), (15,64), … 각 조합을 {"window_size":10, "hidden_size":64} dict 로
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    res_dir = cfg["paths"]["results_dir"]
    print(f"[tune:{model}] {len(combos)} 조합 × {len(seeds)} 시드 = {len(combos) * len(seeds)} 회 학습")

    rows = []
    t0 = time.time()
    # enumerate(combos, 1): 번호를 1 부터 매겨 "(3/6)" 처럼 진행 상황을 보여 준다
    for i, combo in enumerate(combos, 1):
        for seed in seeds:
            c = copy.deepcopy(cfg)
            c["seed"] = seed
            c[model].update(combo)               # 이번 조합의 값으로 모델 설정을 덮어쓴다
            r = mod.run(c, tag=f"_tune_{model}", export=False)    # 배포 모델은 건드리지 않음
            # 결과 한 행: 조합 값(리스트는 csv 에 넣기 위해 문자열로) + 시드 + 선택된 백분위수 + 시간 + val 지표
            rows.append({**{k: json.dumps(v) if isinstance(v, list) else v for k, v in combo.items()},
                         "seed": seed, "percentile": r["best_config"].get("percentile"),
                         "train_seconds": r["train_seconds"],
                         **{k: r["val"][k] for k in ("precision", "recall", "f1", "fp", "fn")}})
            # 한 번 학습할 때마다 지금까지의 결과를 저장 → 중간에 멈춰도 결과가 남는다
            pd.DataFrame(rows).to_csv(res_dir / f"tuning_{model}.csv", index=False)   # 중간 저장
            # flush=True: 출력을 버퍼에 모아 두지 않고 바로 화면/로그 파일에 쓴다 (백그라운드 실행 모니터링용)
            print(f"[tune:{model}] ({i}/{len(combos)}) {combo} seed={seed} -> F1 {r['val']['f1']:.4f} "
                  f"(누적 {time.time() - t0:.0f}s)", flush=True)
    (res_dir / f"_tune_{model}.json").unlink(missing_ok=True)     # 임시 결과 파일 정리

    df = pd.DataFrame(rows)
    group = [k for k in keys]
    # 조합별로 시드 결과를 묶어 평균·표준편차 계산
    #   agg(새열이름=(원래열, 계산방법)) 형태. f1_std 는 lambda 로 모집단 표준편차(ddof=0)
    summary = (df.groupby(group, sort=False)
                 .agg(f1_mean=("f1", "mean"), f1_std=("f1", lambda x: x.std(ddof=0)),
                      precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                      train_seconds=("train_seconds", "mean"))
                 .reset_index()
                 # 정렬: 평균 F1 높은 순 → 같으면 학습 시간이 짧은 순
                 .sort_values(["f1_mean", "train_seconds"], ascending=[False, True]))
    summary.round(4).to_csv(res_dir / f"tuning_{model}_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"[tune:{model}] best: {summary.iloc[0][group].to_dict()}")
    return summary


def main() -> None:
    """명령줄 진입점 (`make tune`, 두 모델 합쳐 약 45분)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=["deeplog", "autoencoder"], required=True)
    args = ap.parse_args()
    run(load_config(args.config), args.model)


if __name__ == "__main__":
    main()
