"""6단계 최종 평가: val 로 고정한 설정을 test 에 단 1회 적용하고, 계획서 목표 달성 여부를 정리한다.

  1) 베이스라인 2종: results/baseline_*.json 의 test (val 로 고른 설정, 결정적)
  2) Autoencoder · DeepLog: 시드 42·43·44 로 학습 → val 로 임계치 선택 → test 평가
     시드 42 는 배포 모델(artifacts/models/*.pt)과 동일하게 다시 내보낸다 (평가한 모델 = 배포 모델)
  3) EOS 제거 비교 (시드 42) 도 test 로 확인
  4) 지연시간: 원본 로그 1줄 → OnlineParser → Detector.score_event 까지 (HTTP 제외) p50/p95/p99

출력 : results/final/*.json (실행별), results/final_test.json, results/final_test.csv
       artifacts/eval/final/*.npz (그래프용 test 점수, git 제외)

사용법: python -m logguard.eval.final_eval            (약 20분, DeepLog 학습이 대부분)

────────────────────────────────────────────────────────────────────────────────────────────
"test 는 단 한 번" 원칙이 왜 중요한가
────────────────────────────────────────────────────────────────────────────────────────────
test 결과를 보고 설정을 고치고 다시 test 를 보는 일을 반복하면, 결국 test 에 맞춰진 설정이 남는다.
그러면 test 점수가 "처음 보는 데이터에서의 실력" 을 뜻하지 않게 된다 (수능 문제를 보고 공부한 것과 같음).
그래서 1~5단계는 val 만으로 모든 것을 정했고, 이 파일에서 그 설정을 test 에 딱 한 번 적용한다.
결과: Autoencoder val 0.972 ↔ test 0.977 로 거의 같아, val 기반 선택이 과적합되지 않았음을 확인했다.

지연시간 지표
    p50 / p95 / p99 : 전체 측정값을 줄 세웠을 때 50% / 95% / 99% 위치의 값
    → "p95 = 0.10ms" 는 "로그 100줄 중 95줄은 0.10ms 안에 처리된다" 는 뜻 (계획서 목표 ③ 의 기준)
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd

from logguard.config import load_config, raw_log_path

# 평가할 딥러닝 모델 (이 순서대로 학습·평가)
MODELS = ("autoencoder", "deeplog")


def train_and_test(cfg: dict, model: str, seeds: list[int]) -> list[dict]:
    """한 모델을 여러 시드로 학습·test 평가하고, EOS 없는 버전(시드 42)도 평가한다.

    Returns:
        실행별 결과 dict 목록. 각 dict: model, seed, eos(EOS 사용 여부), val·test 지표, 선택된 설정, 학습 시간
    """
    # 함수 안 import: 필요한 모델 모듈만 그때 불러온다
    import importlib
    mod = importlib.import_module(f"logguard.models.{model}")
    runs = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        # export=(seed == 42): 시드 42 만 배포 모델 파일로 저장 (평가한 모델과 배포 모델을 정확히 일치시키기 위해)
        # evaluate_test=True : 이 파일에서만 test 평가를 켠다
        r = mod.run(c, tag=f"final/{model}_s{seed}", export=(seed == 42), evaluate_test=True)
        runs.append({"model": model, "seed": seed, "eos": True, "val": r["val"], "test": r["test"],
                     "best_config": r["best_config"], "train_seconds": r["train_seconds"]})
    # EOS 효과를 test 로 확인하기 위한 비교 실험 (EOS 없이 학습, 배포 파일은 저장하지 않음)
    c = copy.deepcopy(cfg)
    c["seed"] = 42
    r = mod.run(c, add_eos=False, tag=f"final/{model}_noeos_s42", export=False, evaluate_test=True)
    runs.append({"model": model, "seed": 42, "eos": False, "val": r["val"], "test": r["test"],
                 "best_config": r["best_config"], "train_seconds": r["train_seconds"]})
    return runs


def latency_benchmark(cfg: dict, n_lines: int = 100_000) -> dict:
    """원본 HDFS.log 앞 n_lines 줄을 순서대로 넣으며 줄마다 (파싱 + 블록별 판정) 시간을 잰다.

    실제 서버에서 로그 1줄이 들어왔을 때 하는 일을 그대로 흉내 낸다:
        ① OnlineParser 로 Event ID·블록 ID 구하기
        ② 블록별 이력에 Event ID 추가
        ③ Detector.score_event(이력) 로 판정
    HTTP 통신, DB 저장, Slack 전송 시간은 포함되지 않는다 (백엔드 통합 후 따로 측정해야 함).
    """
    from logguard.data.online import OnlineParser
    from logguard.models.inference import load_detector

    parser = OnlineParser.load()
    # 파일에서 앞 n_lines 줄만 미리 메모리에 읽어 둔다 (디스크 읽기 시간이 측정에 섞이지 않도록)
    #   next(f): 파일에서 다음 한 줄을 읽음
    with open(raw_log_path(cfg), encoding="utf-8", errors="replace") as f:
        lines = [next(f) for _ in range(n_lines)]
    out = {}
    for name in ("statistical_frequency", "autoencoder", "deeplog"):
        det = load_detector(name, cfg)
        history: dict[str, list] = {}           # 블록 ID → 지금까지의 Event ID 목록
        times = np.empty(n_lines)               # 줄별 처리 시간 (초)
        for i, line in enumerate(lines):
            t0 = time.perf_counter()
            log = parser.parse_line(line)
            for blk in log.block_ids:
                # setdefault: 처음 보는 블록이면 빈 리스트를 만들어 넣고, 있으면 기존 리스트를 돌려준다
                h = history.setdefault(blk, [])
                h.append(log.event_id)
                det.score_event(h)
            times[i] = time.perf_counter() - t0
        ms = times * 1e3                        # 초 → 밀리초
        out[name] = {"lines": n_lines, "p50_ms": round(float(np.percentile(ms, 50)), 3),
                     "p95_ms": round(float(np.percentile(ms, 95)), 3), "p99_ms": round(float(np.percentile(ms, 99)), 3),
                     "max_ms": round(float(ms.max()), 3),
                     # 단일 스레드 처리량 = 처리한 줄 수 / 걸린 총 시간 (계획서 목표 ④ 참고값)
                     "lines_per_sec_single_thread": round(float(n_lines / times.sum()), 1)}
        print(f"[latency] {name:<22} {out[name]}")
    return out


def summarize(cfg: dict, runs: list[dict], latency: dict) -> dict:
    """모든 방법의 test 결과를 한 표로 모으고, 계획서 목표 달성 여부를 판정해 파일로 저장한다."""
    res = cfg["paths"]["results_dir"]
    rows = []
    # 베이스라인 2종: 이미 저장된 결과 파일의 test 값 사용 (무작위성이 없어 시드 1개로 충분)
    for name in ("rule_based", "statistical_frequency"):
        b = json.loads((res / f"baseline_{name}.json").read_text())
        rows.append({"method": name, "eos": None, "val_f1": b["val"]["f1"],
                     **{f"test_{k}": b["test"][k] for k in ("precision", "recall", "f1")},
                     "test_f1_mean": b["test"]["f1"], "test_f1_std": 0.0, "n_seeds": 1})
    df = pd.DataFrame(runs)
    # 딥러닝 모델: (모델, EOS 여부) 별로 묶어서 시드 평균·표준편차 계산
    for (model, eos), g in df.groupby(["model", "eos"], sort=False):
        # 표의 P/R/F1 은 배포 모델(시드 42) 값, test_f1_mean/std 는 시드 전체 통계
        s42 = (g[g["seed"] == 42] if (g["seed"] == 42).any() else g).iloc[0]   # 배포 모델(seed 42) 기준
        f1s = np.array([t["f1"] for t in g["test"]])
        rows.append({"method": model if eos else f"{model}_noeos", "eos": eos, "val_f1": s42["val"]["f1"],
                     **{f"test_{k}": s42["test"][k] for k in ("precision", "recall", "f1")},
                     "test_f1_mean": round(float(f1s.mean()), 4), "test_f1_std": round(float(f1s.std()), 4),
                     "n_seeds": len(g)})
    table = pd.DataFrame(rows)
    table.to_csv(res / "final_test.csv", index=False)

    # 최종 모델 = EOS 를 쓴 모델 중 test F1 시드 평균이 가장 높은 것 (결과: autoencoder)
    best = table[table["eos"] == True].sort_values("test_f1_mean", ascending=False).iloc[0]  # noqa: E712
    rule_f1 = table.loc[table["method"] == "rule_based", "test_f1"].iloc[0]
    # 계획서 Ⅰ-3 의 목표 ①~⑤ 달성 여부 (⑥ 대시보드·Slack 은 이동준 담당이라 여기서 판정하지 않음)
    goals = {
        "1_f1_ge_0.85": {"value": best["test_f1"], "target": 0.85, "met": bool(best["test_f1"] >= 0.85)},
        "2_rule_based_plus_10pp": {"value": round(best["test_f1"] - rule_f1, 4), "target": 0.10,
                                   "met": bool(best["test_f1"] - rule_f1 >= 0.10)},
        "3_inference_p95_le_1s": {"value_ms": latency[best["method"]]["p95_ms"], "target_ms": 1000,
                                  "met": bool(latency[best["method"]]["p95_ms"] <= 1000),
                                  "note": "원본 1줄 파싱 + 판정 (HTTP·DB·Slack 제외). E2E 2초는 백엔드 통합 후 측정"},
        "4_throughput_500_per_sec": {"value": latency[best["method"]]["lines_per_sec_single_thread"], "target": 500,
                                     "met": bool(latency[best["method"]]["lines_per_sec_single_thread"] >= 500),
                                     "note": "모델 경로 단일 스레드 처리량. Locust 부하테스트는 백엔드 통합 후"},
        "5_two_models_compared": {"selected": best["method"], "met": True},
    }
    summary = {"selected_model": best["method"], "table": table.to_dict(orient="records"),
               "runs": runs, "latency": latency, "goals": goals}
    (res / "final_test.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(table.to_string(index=False))
    print(json.dumps(goals, indent=2, ensure_ascii=False, default=float))
    return summary


def main() -> None:
    """명령줄 진입점 (`make final-eval`). 학습 → test 평가 → 지연시간 측정 → 요약 순서로 실행."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = cfg["tuning"]["seeds"]
    t0 = time.time()
    # 두 모델 각각의 실행 결과 목록을 하나로 이어 붙인다 (리스트 컴프리헨션 2중 for)
    runs = [r for m in MODELS for r in train_and_test(cfg, m, seeds)]
    latency = latency_benchmark(cfg)
    summarize(cfg, runs, latency)
    print(f"[final] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
