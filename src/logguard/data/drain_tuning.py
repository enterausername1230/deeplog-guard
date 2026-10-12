"""Drain 파싱 품질 검증 및 sim_th·depth 튜닝 (계획서 Ⅱ-1 ①).

정답(ground truth): Loghub 공식 템플릿 29개 (preprocessed/HDFS.log_templates.csv) 를 정규식으로 바꿔 라벨링.
지표:
  grouping_accuracy (GA) : 클러스터와 정답 그룹이 정확히 1:1 로 일치하는 라인 비율 (Zhu et al., ICSE-SEIP 2019)
  merged_lines           : 서로 다른 정답 이벤트가 한 클러스터로 합쳐진 라인 수 → 이상 탐지에 가장 해로움
  split_events           : 하나의 정답 이벤트가 여러 클러스터로 쪼개진 개수 → 어휘가 늘어날 뿐 비교적 덜 해로움

속도: 마스킹 결과가 같은 메시지는 Drain 결과도 같으므로(parse.py 의 캐시와 동일한 원리),
      고유 마스킹 메시지를 처음 등장한 순서대로 한 번만 넣어 설정당 수 초 안에 평가한다.

입력 : data/raw/HDFS_v1/HDFS.log, preprocessed/{HDFS.log_templates.csv, Event_traces.csv},
       data/interim/{hdfs_sessions.parquet, hdfs_templates.csv}
출력 : data/interim/hdfs_unique_masked.parquet (캐시), results/drain_tuning.csv, results/drain_validation.json

사용법: python -m logguard.data.drain_tuning
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from logguard.config import ROOT, load_config, raw_log_path
from logguard.data.parse import build_miner, split_header

SIM_TH_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
DEPTH_GRID = [3, 4, 5, 6]


def unique_masked_messages(cfg: dict) -> pd.DataFrame:
    """고유 마스킹 메시지 (처음 등장 순서) + 라인 수. 한 번 만들면 캐시."""
    cache = cfg["paths"]["interim_dir"] / "hdfs_unique_masked.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    masker = build_miner(ROOT / cfg["parsing"]["drain_config"]).masker
    counts: Counter = Counter()
    t0 = time.time()
    with open(raw_log_path(cfg), encoding="utf-8", errors="replace") as f:
        for line in f:
            h = split_header(line)
            counts[masker.mask(h[4] if h else line.strip())] += 1
    df = pd.DataFrame({"masked": list(counts), "count": np.fromiter(counts.values(), dtype=np.int64)})
    df.to_parquet(cache, index=False)
    print(f"[unique] {len(df):,} unique masked messages from {df['count'].sum():,} lines ({time.time() - t0:.0f}s)")
    return df


def official_regexes(path) -> list[tuple[str, re.Pattern, int]]:
    """'[*]Served block[*]to[*]' -> 정규식. 여러 개가 맞으면 고정 문자열이 가장 긴(가장 구체적인) 템플릿 선택."""
    tpl = pd.read_csv(path)
    out = []
    for eid, t in zip(tpl["EventId"], tpl["EventTemplate"]):
        parts = t.split("[*]")
        out.append((eid, re.compile(".*?".join(map(re.escape, parts)), re.DOTALL), sum(map(len, parts))))
    return out


def label_official(messages: pd.Series, regexes) -> np.ndarray:
    labels = []
    for m in messages:
        hits = [(spec, eid) for eid, rx, spec in regexes if rx.search(m)]
        labels.append(max(hits)[1] if hits else "NONE")
    return np.asarray(labels)


def run_drain(messages: pd.Series, sim_th: float, depth: int, base_ini) -> np.ndarray:
    tm_cfg = TemplateMinerConfig()
    tm_cfg.load(str(base_ini))
    tm_cfg.drain_sim_th, tm_cfg.drain_depth = sim_th, depth
    drain = TemplateMiner(None, tm_cfg).drain
    return np.asarray([drain.add_log_message(m)[0].cluster_id for m in messages])


def grouping_metrics(cluster: np.ndarray, truth: np.ndarray, weight: np.ndarray) -> dict:
    df = pd.DataFrame({"c": cluster, "g": truth, "w": weight})
    cg = df.groupby(["c", "g"])["w"].sum().rename("n").reset_index()
    cg["n_c"] = cg["c"].map(df.groupby("c")["w"].sum())
    cg["n_g"] = cg["g"].map(df.groupby("g")["w"].sum())
    total = df["w"].sum()
    exact = cg[(cg["n"] == cg["n_c"]) & (cg["n"] == cg["n_g"])]
    impure_clusters = cg.groupby("c")["g"].nunique().loc[lambda s: s > 1].index
    return {
        "templates": int(df["c"].nunique()),
        "grouping_accuracy": round(float(exact["n"].sum() / total), 6),
        "exact_events": int(len(exact)),
        "merged_clusters": int(len(impure_clusters)),
        "merged_lines": int(df.loc[df["c"].isin(impure_clusters), "w"].sum()),
        "split_events": int((cg.groupby("g")["c"].nunique() > 1).sum()),
    }


def validate_current_parse(cfg: dict) -> dict:
    """현재 파싱 결과(세션 이벤트 시퀀스)를 Loghub 공식 Event_traces 와 블록 단위로 비교."""
    interim, raw = cfg["paths"]["interim_dir"], cfg["paths"]["raw_dir"]
    ours = pd.read_parquet(interim / "hdfs_sessions.parquet", columns=["block_id", "events"])
    tpl = pd.read_csv(interim / "hdfs_templates.csv")
    regexes = official_regexes(raw / "preprocessed" / "HDFS.log_templates.csv")
    # 우리 템플릿의 마스킹 토큰을 일반 와일드카드로 바꿔 공식 템플릿에 매핑
    plain = tpl["template"].str.replace(r"<:\w+:>|<:\*:>", "*", regex=True)
    to_official = dict(zip(tpl["event_id"], label_official(plain, regexes)))

    traces = pd.read_csv(raw / "preprocessed" / "Event_traces.csv", usecols=["BlockId", "Features"])
    official = dict(zip(traces["BlockId"], traces["Features"].str.strip("[]")))
    match = 0
    mismatch_examples = []
    for blk, seq in zip(ours["block_id"], ours["events"]):
        mapped = ",".join(to_official[int(e)] for e in seq)
        if mapped == official.get(blk):
            match += 1
        elif len(mismatch_examples) < 3:
            mismatch_examples.append({"block_id": blk, "ours": mapped[:200], "official": str(official.get(blk))[:200]})
    return {
        "our_templates": len(tpl),
        "official_templates": len(regexes),
        "our_to_official": {f"E{k}": v for k, v in to_official.items()},
        "unmapped_our_templates": [f"E{k}" for k, v in to_official.items() if v == "NONE"],
        "blocks_compared": len(ours),
        "blocks_exact_match": match,
        "block_match_rate": round(match / len(ours), 6),
        "mismatch_examples": mismatch_examples,
    }


def run(cfg: dict) -> None:
    res_dir = cfg["paths"]["results_dir"]
    raw = cfg["paths"]["raw_dir"]
    base_ini = ROOT / cfg["parsing"]["drain_config"]

    print("[1/3] 현재 파싱 결과 vs Loghub 공식 Event_traces")
    val = validate_current_parse(cfg)
    (res_dir / "drain_validation.json").write_text(json.dumps(val, indent=2, ensure_ascii=False))
    print(f"  블록 시퀀스 완전 일치: {val['blocks_exact_match']:,}/{val['blocks_compared']:,} ({val['block_match_rate']:.4%})")
    print(f"  공식 템플릿에 매핑 안 된 우리 템플릿: {val['unmapped_our_templates']}")

    print("[2/3] 고유 마스킹 메시지 + 공식 라벨")
    uniq = unique_masked_messages(cfg)
    truth = label_official(uniq["masked"], official_regexes(raw / "preprocessed" / "HDFS.log_templates.csv"))
    print(f"  고유 메시지 {len(uniq):,}개, 공식 라벨 없음(NONE) {int(uniq['count'][truth == 'NONE'].sum()):,} 라인")

    print("[3/3] sim_th × depth 그리드")
    rows = []
    for depth in DEPTH_GRID:
        for sim in SIM_TH_GRID:
            clusters = run_drain(uniq["masked"], sim, depth, base_ini)
            m = grouping_metrics(clusters, truth, uniq["count"].to_numpy())
            rows.append({"sim_th": sim, "depth": depth, **m})
            print(f"  depth={depth} sim_th={sim}: {m}")
    grid = pd.DataFrame(rows).sort_values(["merged_lines", "grouping_accuracy", "templates"],
                                          ascending=[True, False, True])
    grid.to_csv(res_dir / "drain_tuning.csv", index=False)
    print(grid.head(8).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
