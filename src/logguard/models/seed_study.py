"""시드별 성능 편차 측정. 모델 간 비교가 시드 운에 좌우되지 않는지 확인한다 (val 기준).

출력 : results/seed_study_<model>.csv  (시드별 val P/R/F1, 선택된 백분위수) + 평균·표준편차 출력

사용법: python -m logguard.models.seed_study --model autoencoder --seeds 42 43 44

왜 필요한가?
    딥러닝 모델은 시작할 때 가중치를 무작위로 정하고, 학습 데이터도 무작위 순서로 섞는다.
    이 "무작위" 의 출발점이 시드(seed)다. 시드만 바꿔도 결과가 조금씩 달라진다.
    4단계에서 시드 하나로 비교했더니 F1 이 ±0.02 정도 흔들렸고, 이는 두 모델 간 차이와 비슷한 크기였다.
    → "AE 가 더 좋다" 가 실력인지 운인지 알려면 여러 시드의 평균과 표준편차를 봐야 한다.
    (5단계 이후로는 tune.py 와 final_eval.py 가 같은 일을 하므로, 이 파일의 결과는 4단계(w=10) 기록이다)
"""
from __future__ import annotations

import argparse
import copy             # deepcopy: 설정 dict 를 통째로 복사 (시드만 바꾼 복사본을 만들 때 원본을 보호)
import importlib        # 문자열 이름("deeplog")으로 모듈을 불러오기

import pandas as pd

from logguard.config import load_config


def run(cfg: dict, model: str, seeds: list[int]) -> pd.DataFrame:
    """같은 설정으로 시드만 바꿔 여러 번 학습하고, 시드별 val 성능을 표로 만든다.

    Args:
        model: "deeplog" 또는 "autoencoder" (logguard.models.<model> 모듈의 run 을 호출)
        seeds: 사용할 시드 목록. 예: [42, 43, 44]
    """
    mod = importlib.import_module(f"logguard.models.{model}")
    rows = []
    for seed in seeds:
        c = copy.deepcopy(cfg)          # 원본 cfg 를 건드리지 않도록 복사
        c["seed"] = seed
        # export=False: 실험용이므로 배포 모델 파일(artifacts/models/*.pt)을 덮어쓰지 않는다
        # tag 앞의 "_" : 임시 결과 파일이라는 표시 (아래에서 지운다)
        r = mod.run(c, tag=f"_seed_{model}", export=False)
        rows.append({"seed": seed, **r["best_config"], **{k: r["val"][k] for k in ("precision", "recall", "f1")}})
    # 각 run 이 남긴 임시 결과 파일 삭제 (missing_ok=True: 없어도 오류 아님)
    (cfg["paths"]["results_dir"] / f"_seed_{model}.json").unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["results_dir"] / f"seed_study_{model}.csv", index=False)
    print(df.to_string(index=False))
    # std(ddof=0): 모집단 표준편차 (시드 3개 자체의 흩어진 정도)
    print(f"[{model}] val F1 mean {df.f1.mean():.4f} ± {df.f1.std(ddof=0):.4f} (n={len(df)})")
    return df


def main() -> None:
    """명령줄 진입점 (`make seed-study`)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", choices=["deeplog", "autoencoder"], required=True)
    # nargs="+": 값을 여러 개 받을 수 있음 → --seeds 42 43 44
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    run(load_config(args.config), args.model, args.seeds)


if __name__ == "__main__":
    main()
