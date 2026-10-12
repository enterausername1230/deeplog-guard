"""5단계 튜닝: 하이퍼파라미터 그리드 × 시드 → val F1 평균으로 설정 선택 (test 는 보지 않음).

그리드는 configs/config.yaml 의 tuning.<model> 에 정의. 각 조합을 tuning.seeds 의 시드로 학습하고,
평균 val F1 이 가장 높은 조합을 고른다 (같으면 학습 시간이 짧은 쪽).

출력 : results/tuning_<model>.csv          (조합 × 시드 전체)
       results/tuning_<model>_summary.csv  (조합별 평균 ± 표준편차, 정렬)

사용법: python -m logguard.models.tune --model deeplog
"""
from __future__ import annotations

import argparse
import copy
import importlib
import itertools
import json
import time

import pandas as pd

from logguard.config import load_config


def run(cfg: dict, model: str) -> pd.DataFrame:
    mod = importlib.import_module(f"logguard.models.{model}")
    grid = cfg["tuning"][model]
    seeds = cfg["tuning"]["seeds"]
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    res_dir = cfg["paths"]["results_dir"]
    print(f"[tune:{model}] {len(combos)} 조합 × {len(seeds)} 시드 = {len(combos) * len(seeds)} 회 학습")

    rows = []
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        for seed in seeds:
            c = copy.deepcopy(cfg)
            c["seed"] = seed
            c[model].update(combo)
            r = mod.run(c, tag=f"_tune_{model}", export=False)
            rows.append({**{k: json.dumps(v) if isinstance(v, list) else v for k, v in combo.items()},
                         "seed": seed, "percentile": r["best_config"].get("percentile"),
                         "train_seconds": r["train_seconds"],
                         **{k: r["val"][k] for k in ("precision", "recall", "f1", "fp", "fn")}})
            pd.DataFrame(rows).to_csv(res_dir / f"tuning_{model}.csv", index=False)   # 중간 저장
            print(f"[tune:{model}] ({i}/{len(combos)}) {combo} seed={seed} -> F1 {r['val']['f1']:.4f} "
                  f"(누적 {time.time() - t0:.0f}s)", flush=True)
    (res_dir / f"_tune_{model}.json").unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    group = [k for k in keys]
    summary = (df.groupby(group, sort=False)
                 .agg(f1_mean=("f1", "mean"), f1_std=("f1", lambda x: x.std(ddof=0)),
                      precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                      train_seconds=("train_seconds", "mean"))
                 .reset_index()
                 .sort_values(["f1_mean", "train_seconds"], ascending=[False, True]))
    summary.round(4).to_csv(res_dir / f"tuning_{model}_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"[tune:{model}] best: {summary.iloc[0][group].to_dict()}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=["deeplog", "autoencoder"], required=True)
    args = ap.parse_args()
    run(load_config(args.config), args.model)


if __name__ == "__main__":
    main()
