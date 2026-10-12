"""시드별 성능 편차 측정. 모델 간 비교가 시드 운에 좌우되지 않는지 확인한다 (val 기준).

출력 : results/seed_study_<model>.csv  (시드별 val P/R/F1, 선택된 백분위수) + 평균·표준편차 출력

사용법: python -m logguard.models.seed_study --model autoencoder --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import copy
import importlib

import pandas as pd

from logguard.config import load_config


def run(cfg: dict, model: str, seeds: list[int]) -> pd.DataFrame:
    mod = importlib.import_module(f"logguard.models.{model}")
    rows = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        r = mod.run(c, tag=f"_seed_{model}", export=False)
        rows.append({"seed": seed, **r["best_config"], **{k: r["val"][k] for k in ("precision", "recall", "f1")}})
    (cfg["paths"]["results_dir"] / f"_seed_{model}.json").unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["results_dir"] / f"seed_study_{model}.csv", index=False)
    print(df.to_string(index=False))
    print(f"[{model}] val F1 mean {df.f1.mean():.4f} ± {df.f1.std(ddof=0):.4f} (n={len(df)})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=["deeplog", "autoencoder"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    run(load_config(args.config), args.model, args.seeds)


if __name__ == "__main__":
    main()
