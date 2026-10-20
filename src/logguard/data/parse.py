"""HDFS 원시 로그를 Drain3로 파싱하여 Event ID로 구조화하고 Block ID 세션을 만든다.

입력 : data/raw/HDFS_v1/HDFS.log, preprocessed/anomaly_label.csv
출력 :
  data/interim/hdfs_lines.parquet      라인별 (date, time, level, component, event_id)
  data/interim/hdfs_templates.csv      event_id, event_key(E#), template, occurrences, levels
  data/interim/hdfs_sessions.parquet   block_id, label, first_line, last_line, length, events(list)
  artifacts/drain3/drain3_state.bin    온라인 추론(FastAPI)에서 재사용할 Drain3 상태

사용법: python -m logguard.data.parse [--limit N]

────────────────────────────────────────────────────────────────────────────────────────────
이 파일이 하는 일을 예시로 이해하기
────────────────────────────────────────────────────────────────────────────────────────────
원본 로그 한 줄 (공백으로 구분된 6개 부분):
    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
    ──┬─── ──┬─── ─┬─ ─┬── ───────────┬──────────── ──────────────────────────┬───────────────────────────────────────────────
     날짜    시각  PID 레벨      컴포넌트(발생 위치)                           내용(content)

1단계 — 마스킹 (configs/drain3.ini 의 규칙):
    바뀌는 값(IP, 경로, 블록 ID, 숫자)을 기호로 바꾼다.
    → "Receiving block <:BLK:> src: <:IP:> dest: <:IP:>"

2단계 — Drain 클러스터링:
    마스킹된 문장들 중 모양이 같은 것끼리 하나의 "템플릿" 으로 묶고 번호(Event ID)를 붙인다.
    → 이 문장은 Event ID 1 번 템플릿

3단계 — 블록(세션)별로 모으기:
    HDFS 에서는 "블록(blk_…)" 하나의 생애(생성 → 복제 → 삭제)가 하나의 작업 단위다.
    같은 블록 ID 가 나온 줄들의 Event ID 를 순서대로 모으면 그 블록의 "이벤트 시퀀스" 가 된다.
    → blk_-1608 : [1, 2, 1, 1, 3, 4, …]
    정답 라벨(정상/이상)도 블록 단위로 주어지므로, 모델은 "블록 하나 = 세션 하나" 를 판정한다.
"""
from __future__ import annotations

import argparse                       # 명령줄 옵션 (--limit)
import json                           # 요약 결과를 json 으로 저장
from array import array               # 숫자를 아주 촘촘하게(메모리 적게) 저장하는 표준 배열
import re                             # 정규표현식 (블록 ID 찾기)
import time                           # 실행 시간 측정
from collections import defaultdict   # 없는 키에 자동으로 기본값(빈 리스트 등)을 만들어 주는 dict
from pathlib import Path

import numpy as np
import pandas as pd                   # 표 형태 데이터 처리 + parquet 저장
from drain3 import TemplateMiner      # Drain 알고리즘 구현체 (계획서 [3] He et al., ICWS 2017)
from drain3.file_persistence import FilePersistence             # Drain 상태를 파일로 저장/불러오기
from drain3.template_miner_config import TemplateMinerConfig    # drain3.ini 설정 읽기

from logguard.config import ROOT, label_path, load_config, raw_log_path

# 블록 ID 패턴: "blk_" 뒤에 (음수일 수 있는) 숫자. 예: blk_-1608999687919862906, blk_38865049064139660
#   -?   : 마이너스 기호가 있을 수도 없을 수도
#   \d+  : 숫자 1개 이상
# re.compile 로 미리 컴파일해 두면 1,117만 번 반복 사용할 때 더 빠르다.
BLK_RE = re.compile(r"blk_-?\d+")


def split_header(line: str) -> tuple[str, str, str, str, str] | None:
    """'081109 203615 148 INFO dfs.DataNode$PacketResponder: content' -> (date, time, level, component, content)

    HDFS 로그 한 줄을 앞부분(헤더)과 내용으로 나눈다. 형식이 맞지 않는 줄이면 None 을 돌려준다.
    (PID 는 분석에 쓰지 않으므로 버린다)
    """
    # rstrip("\n") : 줄 끝 개행 문자 제거
    # split(" ", 4) : 공백 기준으로 "최대 4번만" 자른다 → 앞의 4개(날짜, 시각, PID, 레벨) + 나머지 전부
    #   내용(content)에도 공백이 많기 때문에 4번에서 멈춰야 내용이 잘게 쪼개지지 않는다.
    parts = line.rstrip("\n").split(" ", 4)
    if len(parts) < 5:          # 조각이 5개가 안 되면 정상적인 로그 형식이 아님
        return None
    date, tm, _pid, level, rest = parts   # _pid: 안 쓰는 값이라는 뜻으로 밑줄을 붙임
    # rest = "dfs.DataNode$PacketResponder: PacketResponder 1 for block …"
    # partition(": ") : 처음 나오는 ": " 를 기준으로 (앞, 구분자, 뒤) 3개로 나눈다
    component, sep, content = rest.partition(": ")
    if not sep:                 # ": " 가 없으면 컴포넌트/내용을 구분할 수 없으므로 형식 오류
        return None
    return date, tm, level, component, content


def build_miner(drain_ini: Path) -> TemplateMiner:
    """drain3.ini 설정(마스킹 규칙, sim_th, depth 등)으로 Drain 템플릿 추출기를 만든다.

    persistence_handler=None : 파싱 도중에는 상태를 파일에 계속 저장하지 않는다 (느려지므로).
                               다 끝난 뒤 HDFSParser.save_state() 로 한 번만 저장한다.
    """
    tm_cfg = TemplateMinerConfig()
    tm_cfg.load(str(drain_ini))
    return TemplateMiner(persistence_handler=None, config=tm_cfg)


class HDFSParser:
    """Drain3 래퍼. 마스킹 결과가 같은 메시지는 캐시로 처리해 1,100만 줄을 빠르게 파싱한다.

    속도 비결 (캐시)
        HDFS 로그 1,117만 줄은 마스킹하고 나면 서로 다른 문장이 겨우 54종뿐이다.
        (블록 ID, IP, 숫자만 다르고 문장 모양은 같은 줄이 대부분이기 때문)
        그래서 "마스킹된 문장 → Event ID" 를 dict 에 기억해 두고, 같은 문장이 다시 나오면
        Drain 을 다시 돌리지 않고 바로 꺼내 쓴다. 덕분에 전체 파싱이 약 78초에 끝난다.
    """

    def __init__(self, drain_ini: Path):
        self.miner = build_miner(drain_ini)
        # 캐시: {마스킹된 문장: Event ID}. 앞에 _ 를 붙인 것은 "클래스 내부용" 이라는 관례 표시.
        self._cache: dict[str, int] = {}

    def parse(self, content: str) -> int:
        """로그 내용(content) 한 줄 → Event ID(템플릿 번호)."""
        # 1) 마스킹: 바뀌는 값(IP/경로/블록ID/숫자)을 <:IP:> 같은 기호로 바꾼다
        masked = self.miner.masker.mask(content)
        # 2) 캐시에 있으면 그 번호를 바로 쓴다 (없으면 None)
        cid = self._cache.get(masked)
        if cid is None:
            # 3) 처음 보는 문장이면 Drain 에 넣어 템플릿을 찾거나 새로 만든다.
            #    add_log_message 는 (클러스터, 변경종류) 를 돌려주는데 변경종류는 안 쓰므로 _ 로 버린다.
            cluster, _ = self.miner.drain.add_log_message(masked)
            cid = cluster.cluster_id          # 클러스터 번호 = Event ID
            self._cache[masked] = cid         # 다음번을 위해 캐시에 기억
        return cid

    def templates(self) -> dict[int, str]:
        """지금까지 만들어진 모든 템플릿 {Event ID: 템플릿 문장} 을 돌려준다."""
        return {c.cluster_id: c.get_template() for c in self.miner.drain.clusters}

    def save_state(self, path: Path) -> None:
        """Drain 의 학습 상태(템플릿 트리 전체)를 파일로 저장한다.

        왜 저장하나?
            실시간 서버(FastAPI)는 새 로그를 받을 때 "학습 때와 똑같은 템플릿 번호" 를 써야 한다.
            번호가 달라지면 모델이 엉뚱한 이벤트로 착각하기 때문이다.
            그래서 이 상태 파일을 data/online.py 의 OnlineParser 가 그대로 불러와 쓴다.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        # drain3 의 저장 기능은 "persistence_handler" 가 설정돼 있어야 동작하므로
        # 잠깐 파일 저장기를 붙였다가 → 저장하고 → 다시 떼어 낸다.
        self.miner.persistence_handler = FilePersistence(str(path))
        self.miner.save_state("final")      # "final" 은 로그에 남는 저장 사유 문자열
        self.miner.persistence_handler = None


def run(cfg: dict, limit: int | None = None) -> None:
    """전체 파싱 파이프라인: 로그 읽기 → Event ID 부여 → 블록별 묶기 → 파일 4종 저장.

    Args:
        cfg  : load_config() 결과
        limit: 앞에서 N 줄만 처리 (빠른 시험용). None 이면 전체 1,117만 줄.
    """
    paths = cfg["paths"]
    interim, artifacts = paths["interim_dir"], paths["artifacts_dir"]
    interim.mkdir(parents=True, exist_ok=True)

    parser = HDFSParser(ROOT / cfg["parsing"]["drain_config"])
    every = cfg["parsing"]["progress_every"]     # 몇 줄마다 진행 상황을 출력할지 (예: 50만)

    # 1,100만 줄이므로 파이썬 리스트 대신 compact array + 범주 코드로 보관
    #   파이썬 리스트에 정수 1,100만 개를 넣으면 개당 수십 바이트 → 수백 MB.
    #   array("i") 는 개당 4바이트, array("h") 는 2바이트, array("b") 는 1바이트라 메모리가 훨씬 적다.
    #   타입 코드: "i" = 4바이트 정수, "h" = 2바이트 정수(최대 32767), "b" = 1바이트 정수(최대 127)
    dates, times, eids = array("i"), array("i"), array("h")      # 날짜(081109), 시각(203518), Event ID
    level_codes, comp_codes = array("b"), array("h")             # 레벨·컴포넌트를 "번호" 로 저장
    # 범주 코드표: "INFO" 같은 문자열을 줄마다 저장하지 않고, 처음 나온 순서대로 0, 1, 2… 번호를 붙여 저장
    level_vocab: dict[str, int] = {}     # 예: {"INFO": 0, "WARN": 1}
    comp_vocab: dict[str, int] = {}      # 예: {"dfs.DataNode$DataXceiver": 0, ...}
    # 블록별 정보 (defaultdict(list): 처음 보는 블록이면 자동으로 빈 리스트를 만들어 준다)
    blk_events: dict[str, list[int]] = defaultdict(list)    # 블록 → 그 블록의 Event ID 시퀀스
    blk_first: dict[str, int] = {}                           # 블록이 처음 나온 줄 번호 (시간순 분할에 사용)
    blk_last: dict[str, int] = {}                            # 블록이 마지막으로 나온 줄 번호
    levels_by_event: dict[int, set] = defaultdict(set)       # Event ID → 그 이벤트가 찍힌 레벨들 {"INFO","WARN"}
    skipped = 0                                              # 형식이 깨진 줄 개수 (HDFS 에서는 0)

    t0 = time.time()
    # errors="replace" : 혹시 깨진 글자가 있어도 멈추지 않고 � 로 바꿔서 계속 읽는다
    with open(raw_log_path(cfg), encoding="utf-8", errors="replace") as f:
        # enumerate : (줄 번호 i, 줄 내용 line) 을 같이 꺼낸다. i 는 0 부터 시작
        # 파일을 한 줄씩 읽으므로 1.5GB 를 한꺼번에 메모리에 올리지 않는다.
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            h = split_header(line)
            if h is None:
                # 형식이 깨진 줄: 버리지 않고 "알 수 없음" 으로 표시해서 줄 번호가 밀리지 않게 한다
                skipped += 1
                h = ("0", "0", "UNKNOWN", "UNKNOWN", line.strip())
            date, tm, level, comp, content = h
            eid = parser.parse(content)          # 내용 → Event ID

            # 줄별 정보 저장 (isdigit: 숫자로만 된 문자열인지 확인 → 아니면 0)
            dates.append(int(date) if date.isdigit() else 0)
            times.append(int(tm) if tm.isdigit() else 0)
            # setdefault(키, 기본값): 키가 있으면 기존 번호, 없으면 새 번호(현재 개수)를 등록하고 돌려줌
            level_codes.append(level_vocab.setdefault(level, len(level_vocab)))
            comp_codes.append(comp_vocab.setdefault(comp, len(comp_vocab)))
            eids.append(eid)
            levels_by_event[eid].add(level)      # 베이스라인의 "WARN 레벨 이벤트" 판단에 사용

            # 한 줄에 여러 블록이 있으면 각 블록 세션에 모두 추가 (Loghub 전처리와 동일)
            #   예: "BLOCK* ask … to delete blk_1 blk_2 blk_3" → 세 블록 모두에 이 이벤트를 넣는다
            #   dict.fromkeys(리스트) : 순서를 유지하면서 중복 제거 (같은 블록이 한 줄에 두 번 나와도 1번만)
            for blk in dict.fromkeys(BLK_RE.findall(content)):
                blk_events[blk].append(eid)
                blk_first.setdefault(blk, i)     # 처음 나온 줄만 기록 (이미 있으면 그대로)
                blk_last[blk] = i                # 나올 때마다 덮어써서 마지막 줄을 기록

            # 진행 상황 출력: every 줄마다 (처리 줄 수 | 템플릿 수 | 블록 수 | 초당 처리 속도)
            if every and (i + 1) % every == 0:
                el = time.time() - t0
                print(f"  {i + 1:>10,} lines | {len(parser.templates()):>3} templates | "
                      f"{len(blk_events):>7,} blocks | {(i + 1) / el:,.0f} lines/s")

    n_lines = len(eids)
    print(f"[parse] {n_lines:,} lines in {time.time() - t0:.1f}s (header skipped: {skipped})")

    # ── 출력 1) 라인 테이블: 원본 한 줄 = 한 행. 행 번호 = HDFS.log 의 줄 번호(0부터) ─────────────
    # np.frombuffer : array 의 메모리를 복사 없이 numpy 배열로 바라보기 (빠르고 메모리 절약)
    # pd.Categorical.from_codes : "번호 + 코드표" 를 pandas 범주형 열로 복원 (저장 크기도 작음)
    lines_df = pd.DataFrame({
        "date": np.frombuffer(dates, dtype=np.int32),
        "time": np.frombuffer(times, dtype=np.int32),
        "level": pd.Categorical.from_codes(np.frombuffer(level_codes, dtype=np.int8), list(level_vocab)),
        "component": pd.Categorical.from_codes(np.frombuffer(comp_codes, dtype=np.int16), list(comp_vocab)),
        "event_id": np.frombuffer(eids, dtype=np.int16),
    })
    # parquet: 열 단위로 압축 저장하는 형식. CSV 보다 훨씬 작고 빠르다 (1,117만 행 → 약 6MB)
    lines_df.to_parquet(interim / "hdfs_lines.parquet", index=False)

    # ── 출력 2) 템플릿 테이블: Event ID 별 템플릿 문장·등장 횟수·찍힌 레벨 ─────────────────────
    counts = lines_df["event_id"].value_counts()     # Event ID 별 등장 횟수
    tpl = pd.DataFrame(
        # 각 템플릿마다 한 행: (번호, "E번호", 템플릿 문장, 등장 횟수, "INFO|WARN" 형태의 레벨 목록)
        [(cid, f"E{cid}", t, int(counts.get(cid, 0)), "|".join(sorted(levels_by_event[cid])))
         for cid, t in sorted(parser.templates().items())],
        columns=["event_id", "event_key", "template", "occurrences", "levels"],
    )
    tpl.to_csv(interim / "hdfs_templates.csv", index=False)

    # ── 출력 3) 세션 테이블: 블록 하나 = 한 행. 모델 학습·평가의 기본 단위 ──────────────────────
    labels = pd.read_csv(label_path(cfg))            # 열: BlockId, Label("Normal"/"Anomaly")
    # {블록 ID: 1(이상) 또는 0(정상)} dict 만들기
    label_map = dict(zip(labels["BlockId"], (labels["Label"] == "Anomaly").astype(np.int8)))
    blocks = list(blk_events)                        # 로그에 등장한 모든 블록 ID (등장 순서)
    sess = pd.DataFrame({
        "block_id": blocks,
        "first_line": np.asarray([blk_first[b] for b in blocks], dtype=np.int64),
        "last_line": np.asarray([blk_last[b] for b in blocks], dtype=np.int64),
        "length": np.asarray([len(blk_events[b]) for b in blocks], dtype=np.int32),   # 이벤트 개수
        "events": [np.asarray(blk_events[b], dtype=np.int16) for b in blocks],       # 이벤트 시퀀스
    })
    sess["label"] = sess["block_id"].map(label_map)  # 블록 ID 로 정답 라벨 붙이기 (없으면 NaN)
    unlabeled = int(sess["label"].isna().sum())      # 라벨이 없는 블록 수 (HDFS 는 0개)
    sess = sess.dropna(subset=["label"]).astype({"label": np.int8})   # 라벨 없는 블록은 제외
    sess = sess.sort_values("first_line").reset_index(drop=True)      # 처음 등장한 순서로 정렬
    sess.to_parquet(interim / "hdfs_sessions.parquet", index=False)

    # ── 출력 4) 온라인 추론용 Drain3 상태 (FastAPI 가 같은 템플릿 번호를 쓰도록) ─────────────────
    parser.save_state(artifacts / "drain3" / "drain3_state.bin")

    # ── 요약 정보: 화면 출력 + parse_summary.json 저장 (git 에 올라가서 팀원이 바로 확인 가능) ──
    summary = {
        "lines": n_lines,
        "header_skipped": skipped,
        "templates": len(tpl),
        "sessions": len(sess),
        "sessions_unlabeled_dropped": unlabeled,
        "anomaly_sessions": int(sess["label"].sum()),
        "anomaly_ratio": round(float(sess["label"].mean()), 5),
        "session_length": sess["length"].describe().round(2).to_dict(),   # 평균·최소·최대·사분위수
        "elapsed_sec": round(time.time() - t0, 1),
    }
    # ensure_ascii=False : 한글을 \uXXXX 로 바꾸지 않고 그대로 저장
    (interim / "parse_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    """명령줄 진입점. `python -m logguard.data.parse --limit 100000` 처럼 실행."""
    # RawDescriptionHelpFormatter : --help 에서 위 설명문의 줄바꿈을 그대로 보여 준다
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N줄만 파싱 (빠른 테스트용)")
    args = ap.parse_args()
    run(load_config(args.config), args.limit)


if __name__ == "__main__":
    main()
