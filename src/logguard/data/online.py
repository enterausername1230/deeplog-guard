"""실시간(온라인) 로그 파서 — FastAPI 에서 원본 로그 한 줄을 Event ID 로 바꿀 때 사용.

학습 때 저장한 Drain3 상태(artifacts/drain3/drain3_state.bin)를 불러와 match() 만 한다.
새 템플릿을 학습(add_log_message)하지 않으므로 학습·추론의 Event ID 가 항상 같다.
매칭되는 템플릿이 없는 줄은 event_id=None → 모델 쪽에서 UNK(즉시 이상)로 처리.

사용 예:
    parser = OnlineParser.load()
    log = parser.parse_line("081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 ...")
    log.event_id, log.block_ids
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from logguard.config import ROOT, load_config
from logguard.data.parse import split_header

BLK_RE = re.compile(r"blk_-?\d+")


@dataclass
class ParsedLog:
    raw: str
    date: str
    time: str
    level: str
    component: str
    content: str
    event_id: int | None
    block_ids: list[str] = field(default_factory=list)


class OnlineParser:
    def __init__(self, miner: TemplateMiner):
        self.miner = miner
        self._cache: dict[str, int | None] = {}

    @classmethod
    def load(cls, state_path: str | Path | None = None, drain_ini: str | Path | None = None) -> "OnlineParser":
        cfg = load_config()
        state_path = Path(state_path or cfg["paths"]["artifacts_dir"] / "drain3" / "drain3_state.bin")
        if not state_path.exists():
            raise FileNotFoundError(f"Drain3 상태 없음: {state_path} (python -m logguard.data.parse 먼저 실행)")
        tm_cfg = TemplateMinerConfig()
        tm_cfg.load(str(drain_ini or ROOT / cfg["parsing"]["drain_config"]))
        miner = TemplateMiner(FilePersistence(str(state_path)), tm_cfg)
        miner.persistence_handler = None  # 추론 중 상태 파일을 덮어쓰지 않도록
        return cls(miner)

    def event_id(self, content: str) -> int | None:
        masked = self.miner.masker.mask(content)
        if masked not in self._cache:
            cluster = self.miner.drain.match(masked, full_search_strategy="never")
            self._cache[masked] = cluster.cluster_id if cluster else None
        return self._cache[masked]

    def parse_line(self, line: str) -> ParsedLog:
        line = line.rstrip("\n")
        h = split_header(line) or ("", "", "UNKNOWN", "UNKNOWN", line)
        date, tm, level, comp, content = h
        return ParsedLog(raw=line, date=date, time=tm, level=level, component=comp, content=content,
                         event_id=self.event_id(content), block_ids=list(dict.fromkeys(BLK_RE.findall(content))))

    def template(self, event_id: int) -> str | None:
        for c in self.miner.drain.clusters:
            if c.cluster_id == event_id:
                return c.get_template()
        return None
