"""HDFS 원시 로그를 Drain3로 파싱하여 Event ID로 구조화하고 Block ID 세션을 만든다.

입력 : data/raw/HDFS_v1/HDFS.log, preprocessed/anomaly_label.csv
출력 :
  data/interim/hdfs_lines.parquet      라인별 (date, time, level, component, event_id)
  data/interim/hdfs_templates.csv      event_id, event_key(E#), template, occurrences, levels
  data/interim/hdfs_sessions.parquet   block_id, label, first_line, last_line, length, events(list)
  artifacts/drain3/drain3_state.bin    온라인 추론(FastAPI)에서 재사용할 Drain3 상태

사용법: python -m logguard.data.parse [--limit N]
"""
from __future__ import annotations

import argparse
import json
from array import array
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from logguard.config import ROOT, label_path, load_config, raw_log_path

BLK_RE = re.compile(r"blk_-?\d+")


def split_header(line: str) -> tuple[str, str, str, str, str] | None:
    """'081109 203615 148 INFO dfs.DataNode$PacketResponder: content' -> (date, time, level, component, content)"""
    parts = line.rstrip("\n").split(" ", 4)
    if len(parts) < 5:
        return None
    date, tm, _pid, level, rest = parts
    component, sep, content = rest.partition(": ")
    if not sep:
        return None
    return date, tm, level, component, content


def build_miner(drain_ini: Path) -> TemplateMiner:
    tm_cfg = TemplateMinerConfig()
    tm_cfg.load(str(drain_ini))
    return TemplateMiner(persistence_handler=None, config=tm_cfg)


class HDFSParser:
    """Drain3 래퍼. 마스킹 결과가 같은 메시지는 캐시로 처리해 1,100만 줄을 빠르게 파싱한다."""

    def __init__(self, drain_ini: Path):
        self.miner = build_miner(drain_ini)
        self._cache: dict[str, int] = {}

    def parse(self, content: str) -> int:
        masked = self.miner.masker.mask(content)
        cid = self._cache.get(masked)
        if cid is None:
            cluster, _ = self.miner.drain.add_log_message(masked)
            cid = cluster.cluster_id
            self._cache[masked] = cid
        return cid

    def templates(self) -> dict[int, str]:
        return {c.cluster_id: c.get_template() for c in self.miner.drain.clusters}

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.miner.persistence_handler = FilePersistence(str(path))
        self.miner.save_state("final")
        self.miner.persistence_handler = None


def run(cfg: dict, limit: int | None = None) -> None:
    paths = cfg["paths"]
    interim, artifacts = paths["interim_dir"], paths["artifacts_dir"]
    interim.mkdir(parents=True, exist_ok=True)

    parser = HDFSParser(ROOT / cfg["parsing"]["drain_config"])
    every = cfg["parsing"]["progress_every"]

    # 1,100만 줄이므로 파이썬 리스트 대신 compact array + 범주 코드로 보관
    dates, times, eids = array("i"), array("i"), array("h")
    level_codes, comp_codes = array("b"), array("h")
    level_vocab: dict[str, int] = {}
    comp_vocab: dict[str, int] = {}
    blk_events: dict[str, list[int]] = defaultdict(list)
    blk_first: dict[str, int] = {}
    blk_last: dict[str, int] = {}
    levels_by_event: dict[int, set] = defaultdict(set)
    skipped = 0

    t0 = time.time()
    with open(raw_log_path(cfg), encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            h = split_header(line)
            if h is None:
                skipped += 1
                h = ("0", "0", "UNKNOWN", "UNKNOWN", line.strip())
            date, tm, level, comp, content = h
            eid = parser.parse(content)

            dates.append(int(date) if date.isdigit() else 0)
            times.append(int(tm) if tm.isdigit() else 0)
            level_codes.append(level_vocab.setdefault(level, len(level_vocab)))
            comp_codes.append(comp_vocab.setdefault(comp, len(comp_vocab)))
            eids.append(eid)
            levels_by_event[eid].add(level)

            # 한 줄에 여러 블록이 있으면 각 블록 세션에 모두 추가 (Loghub 전처리와 동일)
            for blk in dict.fromkeys(BLK_RE.findall(content)):
                blk_events[blk].append(eid)
                blk_first.setdefault(blk, i)
                blk_last[blk] = i

            if every and (i + 1) % every == 0:
                el = time.time() - t0
                print(f"  {i + 1:>10,} lines | {len(parser.templates()):>3} templates | "
                      f"{len(blk_events):>7,} blocks | {(i + 1) / el:,.0f} lines/s")

    n_lines = len(eids)
    print(f"[parse] {n_lines:,} lines in {time.time() - t0:.1f}s (header skipped: {skipped})")

    # 1) 라인 테이블
    lines_df = pd.DataFrame({
        "date": np.frombuffer(dates, dtype=np.int32),
        "time": np.frombuffer(times, dtype=np.int32),
        "level": pd.Categorical.from_codes(np.frombuffer(level_codes, dtype=np.int8), list(level_vocab)),
        "component": pd.Categorical.from_codes(np.frombuffer(comp_codes, dtype=np.int16), list(comp_vocab)),
        "event_id": np.frombuffer(eids, dtype=np.int16),
    })
    lines_df.to_parquet(interim / "hdfs_lines.parquet", index=False)

    # 2) 템플릿 테이블
    counts = lines_df["event_id"].value_counts()
    tpl = pd.DataFrame(
        [(cid, f"E{cid}", t, int(counts.get(cid, 0)), "|".join(sorted(levels_by_event[cid])))
         for cid, t in sorted(parser.templates().items())],
        columns=["event_id", "event_key", "template", "occurrences", "levels"],
    )
    tpl.to_csv(interim / "hdfs_templates.csv", index=False)

    # 3) 세션 테이블 + 라벨
    labels = pd.read_csv(label_path(cfg))
    label_map = dict(zip(labels["BlockId"], (labels["Label"] == "Anomaly").astype(np.int8)))
    blocks = list(blk_events)
    sess = pd.DataFrame({
        "block_id": blocks,
        "first_line": np.asarray([blk_first[b] for b in blocks], dtype=np.int64),
        "last_line": np.asarray([blk_last[b] for b in blocks], dtype=np.int64),
        "length": np.asarray([len(blk_events[b]) for b in blocks], dtype=np.int32),
        "events": [np.asarray(blk_events[b], dtype=np.int16) for b in blocks],
    })
    sess["label"] = sess["block_id"].map(label_map)
    unlabeled = int(sess["label"].isna().sum())
    sess = sess.dropna(subset=["label"]).astype({"label": np.int8})
    sess = sess.sort_values("first_line").reset_index(drop=True)
    sess.to_parquet(interim / "hdfs_sessions.parquet", index=False)

    # 4) 온라인 추론용 Drain3 상태
    parser.save_state(artifacts / "drain3" / "drain3_state.bin")

    summary = {
        "lines": n_lines,
        "header_skipped": skipped,
        "templates": len(tpl),
        "sessions": len(sess),
        "sessions_unlabeled_dropped": unlabeled,
        "anomaly_sessions": int(sess["label"].sum()),
        "anomaly_ratio": round(float(sess["label"].mean()), 5),
        "session_length": sess["length"].describe().round(2).to_dict(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (interim / "parse_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N줄만 파싱 (빠른 테스트용)")
    args = ap.parse_args()
    run(load_config(args.config), args.limit)


if __name__ == "__main__":
    main()
