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

────────────────────────────────────────────────────────────────────────────────────────────
왜 이 검증이 필요했나 (1단계에서 실제로 발견한 문제)
────────────────────────────────────────────────────────────────────────────────────────────
Drain 의 sim_th(유사도 임계치) 는 "두 문장이 얼마나 비슷하면 같은 템플릿으로 볼지" 를 정한다.
기본값 0.5 에서는 아래 두 문장이 6단어 중 5단어가 같아(유사도 0.83) 하나로 합쳐졌다.
    PacketResponder 1 for block blk_… terminating    ← 정상 (170만 줄)
    PacketResponder 1 for block blk_… Interrupted.   ← 이상 신호 (49줄, 이 50개 세션은 전부 이상)
합쳐지면 모델 입장에서는 둘이 같은 이벤트라 이상 신호가 사라진다.
sim_th 를 0.85 로 올려 문제를 없앴고, 공식 정답과 블록 575,061개 전부 100% 일치함을 확인했다.

지표 쉽게 이해하기
    Grouping Accuracy : 우리 템플릿 묶음이 공식 정답 묶음과 "딱 맞아떨어지는" 로그 줄의 비율 (1.0 이 완벽)
    merged (합쳐짐)    : 서로 다른 정답 이벤트가 한 템플릿에 섞임 → 구분 불가 → 이상 탐지에 치명적
    split (쪼개짐)     : 한 정답 이벤트가 여러 템플릿으로 나뉨 → 오히려 더 세밀한 정보라 큰 문제 없음
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter          # 항목별 개수를 세는 dict ("문장 → 등장 횟수")

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from logguard.config import ROOT, load_config, raw_log_path
from logguard.data.parse import build_miner, split_header

# 시험해 볼 설정 후보들 (격자 탐색 = 모든 조합을 다 돌려 봄)
SIM_TH_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]   # 유사도 임계치: 클수록 더 엄격하게 묶음
DEPTH_GRID = [3, 4, 5, 6]                                        # Drain 파싱 트리 깊이


def unique_masked_messages(cfg: dict) -> pd.DataFrame:
    """고유 마스킹 메시지 (처음 등장 순서) + 라인 수. 한 번 만들면 캐시.

    1,117만 줄을 마스킹하면 서로 다른 문장은 54개뿐이다.
    Drain 설정을 36가지나 시험해야 하는데 매번 1,117만 줄을 다시 돌리면 느리므로,
    "54개 문장 + 각 문장이 몇 줄이었는지" 만 한 번 계산해 파일로 저장해 둔다 (약 60초, 이후엔 즉시).

    Returns:
        DataFrame 열: masked(마스킹된 문장), count(그 문장이 나온 줄 수)
        행 순서 = 원본 로그에서 처음 등장한 순서 (Counter 는 넣은 순서를 기억한다)
        → Drain 은 입력 순서에 따라 결과가 달라질 수 있으므로 원본과 같은 순서를 지키는 게 중요하다.
    """
    cache = cfg["paths"]["interim_dir"] / "hdfs_unique_masked.parquet"
    if cache.exists():                      # 이미 만들어 둔 캐시가 있으면 바로 읽어서 끝
        return pd.read_parquet(cache)
    # Drain 전체는 필요 없고 "마스킹 규칙" 만 필요하므로 masker 만 꺼낸다
    masker = build_miner(ROOT / cfg["parsing"]["drain_config"]).masker
    counts: Counter = Counter()
    t0 = time.time()
    with open(raw_log_path(cfg), encoding="utf-8", errors="replace") as f:
        for line in f:
            h = split_header(line)
            # h[4] = 내용(content). 형식이 깨진 줄이면 줄 전체를 내용으로 본다 (parse.py 와 동일 규칙)
            counts[masker.mask(h[4] if h else line.strip())] += 1
    df = pd.DataFrame({"masked": list(counts), "count": np.fromiter(counts.values(), dtype=np.int64)})
    df.to_parquet(cache, index=False)
    print(f"[unique] {len(df):,} unique masked messages from {df['count'].sum():,} lines ({time.time() - t0:.0f}s)")
    return df


def official_regexes(path) -> list[tuple[str, re.Pattern, int]]:
    """'[*]Served block[*]to[*]' -> 정규식. 여러 개가 맞으면 고정 문자열이 가장 긴(가장 구체적인) 템플릿 선택.

    Loghub 공식 템플릿은 바뀌는 부분을 [*] 로 표시한다. 이를 정규식으로 바꿔 어떤 문장이 어느 공식 이벤트인지 판별한다.
        "[*]Served block[*]to[*]"
        → [*] 기준으로 자르기: ["", "Served block", "to", ""]
        → 고정 부분은 re.escape 로 특수문자를 무력화하고, 사이를 ".*?"(아무 글자 0개 이상)로 연결
        → 정규식 "Served\\ block.*?to"
    Returns:
        [(공식 이벤트 ID "E3", 컴파일된 정규식, 고정 문자열 총 길이), ...]
        고정 문자열 길이는 "구체성" 점수: 여러 템플릿이 동시에 맞으면 더 긴(구체적인) 쪽을 정답으로 고른다.
    """
    tpl = pd.read_csv(path)
    out = []
    for eid, t in zip(tpl["EventId"], tpl["EventTemplate"]):
        parts = t.split("[*]")
        # re.DOTALL: "." 이 줄바꿈 문자에도 맞도록 (혹시 모를 여러 줄 메시지 대비)
        out.append((eid, re.compile(".*?".join(map(re.escape, parts)), re.DOTALL), sum(map(len, parts))))
    return out


def label_official(messages: pd.Series, regexes) -> np.ndarray:
    """각 문장에 공식 이벤트 ID("E1"~"E29") 를 붙인다. 아무 템플릿에도 안 맞으면 "NONE"."""
    labels = []
    for m in messages:
        # 이 문장에 맞는 모든 공식 템플릿을 (구체성 점수, 이벤트 ID) 로 모은다
        hits = [(spec, eid) for eid, rx, spec in regexes if rx.search(m)]
        # max(hits): 튜플은 첫 값(구체성 점수)부터 비교 → 가장 구체적인 템플릿. [1] 은 그 이벤트 ID
        labels.append(max(hits)[1] if hits else "NONE")
    return np.asarray(labels)


def run_drain(messages: pd.Series, sim_th: float, depth: int, base_ini) -> np.ndarray:
    """주어진 sim_th·depth 로 Drain 을 새로 만들어 문장들을 넣고, 각 문장의 클러스터(템플릿) 번호를 돌려준다.

    base_ini 의 마스킹 규칙 등은 그대로 쓰고, 튜닝 대상인 sim_th 와 depth 만 바꿔 끼운다.
    """
    tm_cfg = TemplateMinerConfig()
    tm_cfg.load(str(base_ini))
    tm_cfg.drain_sim_th, tm_cfg.drain_depth = sim_th, depth    # 두 값만 덮어쓰기
    drain = TemplateMiner(None, tm_cfg).drain
    # add_log_message(m) 은 (클러스터, 변경종류) 를 돌려주므로 [0] 으로 클러스터만 꺼내 번호를 얻는다
    # (messages 는 이미 마스킹된 문장이므로 masker 를 거치지 않고 drain 에 바로 넣는다)
    return np.asarray([drain.add_log_message(m)[0].cluster_id for m in messages])


def grouping_metrics(cluster: np.ndarray, truth: np.ndarray, weight: np.ndarray) -> dict:
    """우리 클러스터와 공식 정답을 비교해 파싱 품질 지표를 계산한다.

    Args:
        cluster: 문장별 우리 템플릿 번호
        truth  : 문장별 공식 이벤트 ID
        weight : 문장별 줄 수 (54개 문장이 각각 몇 줄을 대표하는지) → 줄 단위 지표로 환산하는 가중치
    """
    df = pd.DataFrame({"c": cluster, "g": truth, "w": weight})
    # (우리 클러스터 c, 정답 그룹 g) 쌍마다 줄 수 n 을 합친 "교차표"
    cg = df.groupby(["c", "g"])["w"].sum().rename("n").reset_index()
    cg["n_c"] = cg["c"].map(df.groupby("c")["w"].sum())     # 그 클러스터의 전체 줄 수
    cg["n_g"] = cg["g"].map(df.groupby("g")["w"].sum())     # 그 정답 그룹의 전체 줄 수
    total = df["w"].sum()
    # 완전 일치: 클러스터 전부가 한 정답 그룹이고(n == n_c), 그 정답 그룹 전부가 이 클러스터에 있음(n == n_g)
    exact = cg[(cg["n"] == cg["n_c"]) & (cg["n"] == cg["n_g"])]
    # 정답 그룹이 2개 이상 섞인 클러스터 = "합쳐진(merged)" 클러스터
    impure_clusters = cg.groupby("c")["g"].nunique().loc[lambda s: s > 1].index
    return {
        "templates": int(df["c"].nunique()),                                   # 우리 템플릿 개수
        "grouping_accuracy": round(float(exact["n"].sum() / total), 6),       # 완전 일치한 줄 비율
        "exact_events": int(len(exact)),                                      # 완전 일치한 이벤트 수
        "merged_clusters": int(len(impure_clusters)),                         # 섞인 클러스터 수
        "merged_lines": int(df.loc[df["c"].isin(impure_clusters), "w"].sum()),  # 섞인 클러스터에 속한 줄 수
        "split_events": int((cg.groupby("g")["c"].nunique() > 1).sum()),     # 여러 클러스터로 쪼개진 정답 이벤트 수
    }


def validate_current_parse(cfg: dict) -> dict:
    """현재 파싱 결과(세션 이벤트 시퀀스)를 Loghub 공식 Event_traces 와 블록 단위로 비교.

    방법
        1) 우리 템플릿 52개 각각이 공식 이벤트 몇 번인지 대응표를 만든다 (E1 → "E5" 같은 식)
        2) 블록마다 우리 시퀀스를 공식 번호로 바꿔 문자열로 만들고,
        3) 공식 Event_traces.csv 의 같은 블록 시퀀스와 글자 하나까지 똑같은지 비교한다.
    결과: 현재 설정(sim_th 0.85)에서 575,061 블록 전부 일치 (block_match_rate = 1.0)
    """
    interim, raw = cfg["paths"]["interim_dir"], cfg["paths"]["raw_dir"]
    ours = pd.read_parquet(interim / "hdfs_sessions.parquet", columns=["block_id", "events"])
    tpl = pd.read_csv(interim / "hdfs_templates.csv")
    regexes = official_regexes(raw / "preprocessed" / "HDFS.log_templates.csv")
    # 우리 템플릿의 마스킹 토큰을 일반 와일드카드로 바꿔 공식 템플릿에 매핑
    #   예: "Receiving block <:BLK:> src: <:IP:>" → "Receiving block * src: *"
    plain = tpl["template"].str.replace(r"<:\w+:>|<:\*:>", "*", regex=True)
    to_official = dict(zip(tpl["event_id"], label_official(plain, regexes)))   # {우리 번호: "E공식번호"}

    # 공식 정답: 블록 ID → "E5,E22,E5,..." 형태 문자열 (원본은 "[E5,E22,...]" 이라 대괄호를 벗긴다)
    traces = pd.read_csv(raw / "preprocessed" / "Event_traces.csv", usecols=["BlockId", "Features"])
    official = dict(zip(traces["BlockId"], traces["Features"].str.strip("[]")))
    match = 0
    mismatch_examples = []
    for blk, seq in zip(ours["block_id"], ours["events"]):
        mapped = ",".join(to_official[int(e)] for e in seq)      # 우리 시퀀스를 공식 번호 문자열로
        if mapped == official.get(blk):
            match += 1
        elif len(mismatch_examples) < 3:
            # 불일치 사례를 3개까지만 기록 (디버깅용, 너무 길면 200자로 자름)
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
    """[1] 현재 파싱 검증 → [2] 고유 문장 + 공식 라벨 준비 → [3] sim_th × depth 36가지 시험."""
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
            rows.append({"sim_th": sim, "depth": depth, **m})   # **m : dict m 의 내용을 펼쳐서 합침
            print(f"  depth={depth} sim_th={sim}: {m}")
    # 정렬 기준: ① 합쳐진 줄이 적을수록(가장 중요) ② GA 가 높을수록 ③ 템플릿 수가 적을수록(단순할수록)
    grid = pd.DataFrame(rows).sort_values(["merged_lines", "grouping_accuracy", "templates"],
                                          ascending=[True, False, True])
    grid.to_csv(res_dir / "drain_tuning.csv", index=False)
    print(grid.head(8).to_string(index=False))


def main() -> None:
    """명령줄 진입점 (`make drain-eval`)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
