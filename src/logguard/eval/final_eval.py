"""6단계 최종 평가: val 로 고정한 설정을 test 에 단 1회 적용하고, 계획서 목표 달성 여부를 정리한다.

  1) 베이스라인 2종: results/baseline_*.json 의 test (val 로 고른 설정, 결정적)
  2) Autoencoder · DeepLog: 시드 42·43·44 로 학습 → val 로 임계치 선택 → test 평가
     시드 42 는 배포 모델(artifacts/models/*.pt)과 동일하게 다시 내보낸다 (평가한 모델 = 배포 모델)
  3) EOS 제거 비교 (시드 42) 도 test 로 확인
  4) 지연시간: 원본 로그 1줄 → OnlineParser → Detector.score_event 까지 (HTTP 제외) p50/p95/p99

출력 : results/final/*.json (실행별), results/final_test.json, results/final_test.csv
       artifacts/eval/final/*.npz (그래프용 test 점수, git 제외)

사용법: python -m logguard.eval.final_eval            (약 20분, DeepLog 학습이 대부분)
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd

from logguard.config import load_config, raw_log_path

MODELS = ("autoencoder", "deeplog")


def train_and_test(cfg: dict, model: str, seeds: list[int]) -> list[dict]:
    import importlib
    mod = importlib.import_module(f"logguard.models.{model}")
    runs = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        r = mod.run(c, tag=f"final/{model}_s{seed}", export=(seed == 42), evaluate_test=True)
        runs.append({"model": model, "seed": seed, "eos": True, "val": r["val"], "test": r["test"],
                     "best_config": r["best_config"], "train_seconds": r["train_seconds"]})
    c = copy.deepcopy(cfg)
    c["seed"] = 42
    r = mod.run(c, add_eos=False, tag=f"final/{model}_noeos_s42", export=False, evaluate_test=True)
    runs.append({"model": model, "seed": 42, "eos": False, "val": r["val"], "test": r["test"],
                 "best_config": r["best_config"], "train_seconds": r["train_seconds"]})
    return runs


def latency_benchmark(cfg: dict, n_lines: int = 100_000) -> dict:
    """원본 HDFS.log 앞 n_lines 줄을 순서대로 넣으며 줄마다 (파싱 + 블록별 판정) 시간을 잰다."""
    from logguard.data.online import OnlineParser
    from logguard.models.inference import load_detector

    parser = OnlineParser.load()
    with open(raw_log_path(cfg), encoding="utf-8", errors="replace") as f:
        lines = [next(f) for _ in range(n_lines)]
    out = {}
    for name in ("statistical_frequency", "autoencoder", "deeplog"):
        det = load_detector(name, cfg)
        history: dict[str, list] = {}
        times = np.empty(n_lines)
        for i, line in enumerate(lines):
            t0 = time.perf_counter()
            log = parser.parse_line(line)
            for blk in log.block_ids:
                h = history.setdefault(blk, [])
                h.append(log.event_id)
                det.score_event(h)
            times[i] = time.perf_counter() - t0
        ms = times * 1e3
        out[name] = {"lines": n_lines, "p50_ms": round(float(np.percentile(ms, 50)), 3),
                     "p95_ms": round(float(np.percentile(ms, 95)), 3), "p99_ms": round(float(np.percentile(ms, 99)), 3),
                     "max_ms": round(float(ms.max()), 3), "lines_per_sec_single_thread": round(float(n_lines / times.sum()), 1)}
        print(f"[latency] {name:<22} {out[name]}")
    return out


def summarize(cfg: dict, runs: list[dict], latency: dict) -> dict:
    res = cfg["paths"]["results_dir"]
    rows = []
    for name in ("rule_based", "statistical_frequency"):
        b = json.loads((res / f"baseline_{name}.json").read_text())
        rows.append({"method": name, "eos": None, "val_f1": b["val"]["f1"],
                     **{f"test_{k}": b["test"][k] for k in ("precision", "recall", "f1")},
                     "test_f1_mean": b["test"]["f1"], "test_f1_std": 0.0, "n_seeds": 1})
    df = pd.DataFrame(runs)
    for (model, eos), g in df.groupby(["model", "eos"], sort=False):
        s42 = (g[g["seed"] == 42] if (g["seed"] == 42).any() else g).iloc[0]   # 배포 모델(seed 42) 기준
        f1s = np.array([t["f1"] for t in g["test"]])
        rows.append({"method": model if eos else f"{model}_noeos", "eos": eos, "val_f1": s42["val"]["f1"],
                     **{f"test_{k}": s42["test"][k] for k in ("precision", "recall", "f1")},
                     "test_f1_mean": round(float(f1s.mean()), 4), "test_f1_std": round(float(f1s.std()), 4),
                     "n_seeds": len(g)})
    table = pd.DataFrame(rows)
    table.to_csv(res / "final_test.csv", index=False)

    best = table[table["eos"] == True].sort_values("test_f1_mean", ascending=False).iloc[0]  # noqa: E712
    rule_f1 = table.loc[table["method"] == "rule_based", "test_f1"].iloc[0]
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = cfg["tuning"]["seeds"]
    t0 = time.time()
    runs = [r for m in MODELS for r in train_and_test(cfg, m, seeds)]
    latency = latency_benchmark(cfg)
    summarize(cfg, runs, latency)
    print(f"[final] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
