"""실시간(온라인) 로그 파서 — FastAPI 에서 원본 로그 한 줄을 Event ID 로 바꿀 때 사용.

학습 때 저장한 Drain3 상태(artifacts/drain3/drain3_state.bin)를 불러와 match() 만 한다.
새 템플릿을 학습(add_log_message)하지 않으므로 학습·추론의 Event ID 가 항상 같다.
매칭되는 템플릿이 없는 줄은 event_id=None → 모델 쪽에서 UNK(즉시 이상)로 처리.

사용 예:
    parser = OnlineParser.load()
    log = parser.parse_line("081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 ...")
    log.event_id, log.block_ids

parse.py 와 무엇이 다른가?
    parse.py  (오프라인): 1,117만 줄 전체를 읽으며 템플릿을 "새로 만들고(학습)" 결과를 파일로 저장한다.
    online.py (실시간)  : 이미 만들어진 템플릿 목록에서 "찾기만(매칭)" 한다. 새 템플릿은 절대 만들지 않는다.
    → 만약 실시간에도 새 템플릿을 만들면, 학습 때 없던 번호가 생기거나 기존 템플릿이 바뀌어
      모델이 받은 번호의 의미가 학습 때와 달라지는 사고가 난다.

검증 결과 (2단계): HDFS.log 139만 줄을 오프라인 파싱 결과와 대조 → 불일치 0건, 평균 8.4µs/줄
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field   # dataclass: 데이터를 담는 클래스를 짧게 정의하는 도구
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from logguard.config import ROOT, load_config
from logguard.data.parse import split_header    # 헤더 분리 규칙은 오프라인과 반드시 같아야 하므로 재사용

# 블록 ID 패턴 (parse.py 와 동일): blk_ 뒤에 숫자, 앞에 마이너스가 올 수 있음
BLK_RE = re.compile(r"blk_-?\d+")


@dataclass
class ParsedLog:
    """로그 한 줄을 파싱한 결과를 담는 상자.

    @dataclass 를 붙이면 아래 필드들을 받는 생성자(__init__)와 보기 좋은 출력(__repr__)이 자동으로 만들어진다.
    백엔드는 log.event_id, log.block_ids 처럼 이름으로 꺼내 쓰면 된다.
    """
    raw: str                    # 원본 줄 그대로 (DB 저장·대시보드 "원본 로그" 표시용)
    date: str                   # 날짜 "081109" (YYMMDD)
    time: str                   # 시각 "203518" (HHMMSS)
    level: str                  # 로그 레벨 "INFO" / "WARN" …
    component: str              # 로그를 남긴 컴포넌트 "dfs.DataNode$DataXceiver"
    content: str                # 본문 (블록 ID, IP 등이 들어 있는 부분)
    event_id: int | None        # 템플릿 번호. 학습 때 본 적 없는 형태의 로그면 None (→ 즉시 이상)
    # 이 줄에 등장한 블록 ID 목록. 한 줄에 여러 블록이 있을 수 있다.
    # field(default_factory=list): 객체마다 "새" 빈 리스트를 만든다 (기본값으로 [] 를 직접 쓰면 객체끼리 공유되는 버그)
    block_ids: list[str] = field(default_factory=list)


class OnlineParser:
    """저장된 Drain 상태로 로그를 Event ID 로 바꾸는 실시간 파서. 서버 시작 시 load() 로 한 번만 만든다."""

    def __init__(self, miner: TemplateMiner):
        self.miner = miner
        # 캐시 {마스킹된 문장: Event ID 또는 None}. HDFS 는 마스킹 후 문장 종류가 54개뿐이라
        # 처음 몇 줄 이후로는 거의 전부 캐시에서 바로 답이 나온다 (그래서 줄당 수 µs).
        self._cache: dict[str, int | None] = {}

    @classmethod
    def load(cls, state_path: str | Path | None = None, drain_ini: str | Path | None = None) -> "OnlineParser":
        """저장된 Drain 상태 파일을 불러와 파서를 만든다.

        @classmethod : 객체 없이 `OnlineParser.load()` 형태로 부르는 "만드는 함수" (팩토리)
        Args:
            state_path: Drain 상태 파일. 생략하면 artifacts/drain3/drain3_state.bin
            drain_ini : 마스킹 규칙 설정. 생략하면 configs/drain3.ini (학습 때와 같은 파일이어야 함)
        """
        cfg = load_config()
        state_path = Path(state_path or cfg["paths"]["artifacts_dir"] / "drain3" / "drain3_state.bin")
        if not state_path.exists():
            # 친절한 오류: 무엇을 먼저 실행해야 하는지 메시지에 적어 준다
            raise FileNotFoundError(f"Drain3 상태 없음: {state_path} (python -m logguard.data.parse 먼저 실행)")
        tm_cfg = TemplateMinerConfig()
        tm_cfg.load(str(drain_ini or ROOT / cfg["parsing"]["drain_config"]))
        # FilePersistence 를 넘기면 TemplateMiner 가 생성되면서 상태 파일을 자동으로 읽어 들인다
        miner = TemplateMiner(FilePersistence(str(state_path)), tm_cfg)
        miner.persistence_handler = None  # 추론 중 상태 파일을 덮어쓰지 않도록
        return cls(miner)

    def event_id(self, content: str) -> int | None:
        """로그 본문 → Event ID. 아는 템플릿이 없으면 None."""
        masked = self.miner.masker.mask(content)       # 학습 때와 똑같은 마스킹 규칙 적용
        if masked not in self._cache:
            # match(): 기존 템플릿 중 맞는 것을 "찾기만" 한다 (없으면 None, 새로 만들지 않음)
            # full_search_strategy="never": 트리 탐색으로 못 찾으면 전체 탐색을 하지 않고 바로 None
            cluster = self.miner.drain.match(masked, full_search_strategy="never")
            self._cache[masked] = cluster.cluster_id if cluster else None
        return self._cache[masked]

    def parse_line(self, line: str) -> ParsedLog:
        """원본 로그 한 줄 → ParsedLog. 백엔드가 로그를 받을 때마다 호출하는 메인 함수."""
        line = line.rstrip("\n")
        # 형식이 깨진 줄이면 split_header 가 None 을 주므로, 줄 전체를 내용으로 보는 기본값을 쓴다
        h = split_header(line) or ("", "", "UNKNOWN", "UNKNOWN", line)
        date, tm, level, comp, content = h
        return ParsedLog(raw=line, date=date, time=tm, level=level, component=comp, content=content,
                         event_id=self.event_id(content),
                         # 블록 ID 를 순서 유지하며 중복 제거 (parse.py 와 같은 규칙)
                         block_ids=list(dict.fromkeys(BLK_RE.findall(content))))

    def template(self, event_id: int) -> str | None:
        """Event ID → 템플릿 문장 (대시보드에 "어떤 로그였는지" 보여 줄 때 사용). 없으면 None."""
        for c in self.miner.drain.clusters:
            if c.cluster_id == event_id:
                return c.get_template()
        return None
