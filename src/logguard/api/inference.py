"""Read-only Drain matching and the interface for the model owner's adapter.

이 파일은 무엇인가? (원작성: 안명근 / 2026-10-20 통합: 옥상민)
    1) LogMatcher : 로그 한 줄 → (헤더 정보, Event ID, 모델 입력 번호, 블록 ID 목록)
                    학습 때 저장한 Drain 상태로 "매칭만" 한다 (새 템플릿을 만들지 않음 → 학습·실시간 번호 일치)
                    검증(10/18): 오프라인 파싱과 HDFS.log 20만 줄 대조 → 불일치 0건
    2) Prediction : 판정기(모델)가 백엔드에 돌려주는 결과 형식
    3) Predictor  : "판정기라면 이 함수들을 가져야 한다" 는 약속 (실제 구현: api/predictor.py 의 DetectorPredictor)
"""
from dataclasses import dataclass
import json
import math
import re
from typing import Protocol

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

# 블록 ID 패턴: blk_ 뒤에 (음수일 수 있는) 숫자
BLOCK = re.compile(r'blk_-?\d+')
# HDFS 로그 한 줄 형식: "날짜6자리 시각6자리 PID 레벨 컴포넌트: 내용"
#   그룹: (1) 날짜 (2) 시각 (3) 레벨 (4) 컴포넌트 (5) 내용   — PID 는 저장하지 않으므로 그룹으로 잡지 않음
HEADER = re.compile(r'^(\d{6})\s+(\d{6})\s+\d+\s+(\w+)\s+([^:]+):\s*(.+)$')


@dataclass(frozen=True)
class Prediction:
    """판정기 결과 1건. frozen=True: 만든 뒤 값을 바꿀 수 없음 (실수로 덮어쓰기 방지).

    필수: anomaly_score(유한한 실수), is_anomaly
    선택(2026-10-20 추가, 기본값 None → 예전 코드·테스트는 그대로 동작):
        reason     : unknown_event / model / normal         → DB logs.reason
        details    : 판정 근거 dict (대시보드에 표시)          → DB logs.details (JSON 문자열로 저장)
        model_name : 판정한 모델 이름 (autoencoder 등)       → DB logs.model_name
        threshold  : 기준선 (정규화 점수라 항상 0.5)          → DB logs.threshold
    """
    anomaly_score: float
    is_anomaly: bool
    reason: str | None = None
    details: dict | None = None
    model_name: str | None = None
    threshold: float | None = None

    def __post_init__(self):
        # NaN·무한대 점수는 DB·그래프를 망가뜨리므로 만들 때 바로 거부한다
        if not math.isfinite(self.anomaly_score):
            raise ValueError('Prediction score must be finite')


class Predictor(Protocol):
    """판정기가 지켜야 할 약속 (Protocol: 이 이름의 함수를 가진 객체면 무엇이든 판정기로 쓸 수 있다).

    필수   predict(events)       : 실시간 판정. 로그가 들어올 때마다 호출
    선택   predict_close(events) : 세션 종료 판정(EOS). 없으면 세션 종료 판정을 건너뛴다 (hasattr 로 확인)
           model_name            : 판정기 이름 (session_results.model_name 에 저장)

    events 형식:
        Encoded indices in chronological order, including the current event.
        At most window_size + 1 entries; PAD and model-specific shaping belong to the adapter.
        = 그 블록의 모델 입력 번호(event_index)들. 오래된 것 → 최신 순, 마지막이 방금 들어온 이벤트.
          최대 window_size + 1 개. PAD 채우기 등 모델별 처리는 판정기(어댑터) 쪽 책임.
    """
    def predict(self, events: tuple[int, ...]) -> Prediction: ...


class LogMatcher:
    """저장된 Drain 상태로 로그를 Event ID 로 바꾸는 파서 (서버 시작 시 1개 생성)."""

    def __init__(self, settings):
        # 필요한 파일이 없으면 서버 시작 단계에서 바로 알려 준다 (요청 처리 중에 터지는 것보다 원인 찾기 쉬움)
        for key in ('drain_config', 'drain_state', 'vocab'):
            if not settings[key].is_file():
                raise FileNotFoundError(f'Missing {key}: {settings[key]}')
        config = TemplateMinerConfig()
        config.load(str(settings['drain_config']))
        # FilePersistence: 생성할 때 상태 파일을 읽어 들인다. match() 만 쓰므로 파일을 덮어쓰지 않는다
        self.miner = TemplateMiner(FilePersistence(str(settings['drain_state'])), config)
        # 어휘: {"Event ID": 모델 입력 번호}. 어휘에 없는 이벤트는 1(UNK)
        self.vocab = json.loads(settings['vocab'].read_text())['event_to_index']

    def parse(self, raw):
        """로그 한 줄 → (헤더·Event ID·입력번호 dict, 블록 ID 목록). 형식이 틀리면 ValueError (→ HTTP 422)."""
        match = HEADER.fullmatch(raw.strip())
        if not match:
            raise ValueError('Expected one HDFS log line: YYMMDD HHMMSS PID LEVEL component: message')
        date, time, level, component, content = match.groups()
        cluster = self.miner.match(content)              # 아는 템플릿을 찾기만 함 (없으면 None)
        event_id = cluster.cluster_id if cluster else None
        return {
            'log_date': date, 'log_time': time, 'level': level, 'component': component,
            'event_id': event_id, 'event_index': self.vocab.get(str(event_id), 1),
        }, list(dict.fromkeys(BLOCK.findall(content)))    # 블록 ID: 순서 유지 + 중복 제거
