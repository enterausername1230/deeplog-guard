"""모델 연결 어댑터: 학습된 Detector(옥상민) 를 백엔드의 Predictor 약속(안명근)에 맞춰 끼워 주는 부품.

왜 어댑터가 필요한가? (두 쪽의 "말" 이 다르다)
    백엔드  : 블록별 최근 이벤트를 "모델 입력 번호(event_index)" 튜플로 준다.    예) (3, 4, 4, 5)
    Detector: "원본 Drain Event ID" 목록을 받는다 (모르는 이벤트는 None).        예) [1, 2, 2, 3]
    → 어댑터가 번호를 되돌려 Detector 에 넘기고, 결과(DetectionResult)를 백엔드 형식(Prediction)으로 바꾼다.

사용 (백엔드 main.py 가 서버 시작 시 자동으로 호출):
    predictor = DetectorPredictor.from_settings(settings)   # configs/api.yaml 의 model, window_size, vocab 사용
    prediction = predictor.predict((3, 4, 4, 5))             # 로그 1줄마다 (실시간 경보)
    prediction = predictor.predict_close((... 최근 이벤트))  # 세션 종료 시 (EOS 판정)

시작 시 안전 검사 (문제가 있으면 서버가 아예 켜지지 않게 해서, 조용히 틀린 판정을 내는 사고를 막는다)
    1) window_size(백엔드가 꺼내 주는 이력 개수) ≥ 모델 윈도우(20)
       10/20 통합 시험: 10 이면 정상 세션 292/300 오탐 (PAD 로 채워진 모양을 처음 보는 패턴으로 판단)
                        20 이면 오프라인 평가와 판정 100% 일치
    2) 백엔드 어휘(event_vocab.json) == 모델 학습 때 어휘 (다르면 번호의 의미가 달라져 엉뚱한 판정)
    3) 윈도우 기반 모델(autoencoder, deeplog)만 지원
       (통계 빈도 탐지기는 세션 "전체" 횟수가 필요해서 최근 일부만 보는 백엔드 구조와 맞지 않음)
"""
from __future__ import annotations

import json
from typing import Sequence

from logguard.api.inference import Prediction
from logguard.data.windows import UNK
from logguard.models.inference import THRESHOLD, DetectionResult, Detector, load_detector


class DetectorPredictor:
    """Detector → Predictor 어댑터. 백엔드 create_app(predictor=...) 에 그대로 넣을 수 있다."""

    def __init__(self, detector: Detector, event_to_index: dict[str, int], history_size: int):
        """
        Args:
            detector      : load_detector("autoencoder") 등으로 불러온 탐지기 (윈도우 크기 속성 w 필요)
            event_to_index: 백엔드가 쓰는 어휘 {"Event ID": 입력 번호} (data/processed/event_vocab.json)
            history_size  : 백엔드가 블록마다 꺼내 주는 이전 이벤트 수 (configs/api.yaml 의 window_size)
        Raises:
            ValueError: 위 "시작 시 안전 검사" 를 통과하지 못하면
        """
        # (3) 윈도우 기반 모델만 지원: Autoencoder·DeepLog Detector 는 윈도우 크기를 self.w 로 갖고 있다
        if not hasattr(detector, 'w'):
            raise ValueError(f'{detector.name}: 윈도우 기반 모델(autoencoder, deeplog)만 백엔드에 연결할 수 있습니다')
        # (1) 이력 개수 검사. Autoencoder 는 직전 w-1 개, DeepLog 는 직전 w 개가 필요 → 넉넉하게 w 이상 요구
        if history_size < detector.w:
            raise ValueError(f'configs/api.yaml window_size={history_size} 가 모델 윈도우 {detector.w} 보다 작습니다. '
                             f'{detector.w} 이상으로 설정하세요 (작으면 정상 세션 대부분이 이상으로 판정됨)')
        # (2) 어휘 일치 검사: 백엔드 어휘와 모델 학습 어휘가 글자 하나까지 같아야 한다
        if dict(event_to_index) != dict(detector.meta['event_to_index']):
            raise ValueError('백엔드 어휘(event_vocab.json)가 모델 학습 어휘와 다릅니다. '
                             'make data 로 데이터를 다시 만들었다면 모델도 다시 학습·내보내기 해야 합니다')
        self.detector = detector
        self.model_name = detector.name                   # session_results.model_name 등에 저장되는 이름
        # 입력 번호 → Event ID 되돌리기 표. UNK(1)·PAD(0)·EOS(2) 는 표에 없으므로 .get() 결과가 None 이 된다
        self.index_to_event = {int(i): int(e) for e, i in event_to_index.items()}

    @classmethod
    def from_settings(cls, settings: dict) -> 'DetectorPredictor | None':
        """configs/api.yaml 설정으로 어댑터를 만든다. model 이 비어 있으면(None) 모델 없이 수집만 → None.

        설정 키: model(autoencoder/deeplog/None), window_size, vocab(event_vocab.json 경로)
        """
        name = settings.get('model')
        if not name:
            return None
        vocab = json.loads(settings['vocab'].read_text(encoding='utf-8'))['event_to_index']
        return cls(load_detector(name), vocab, settings['window_size'])

    def _to_event_ids(self, events: Sequence[int]) -> list[int | None]:
        """입력 번호 튜플 → Event ID 목록. UNK(학습에 없던 이벤트)는 None → Detector 가 즉시 이상으로 판정."""
        return [None if i == UNK else self.index_to_event.get(int(i)) for i in events]

    def _to_prediction(self, r: DetectionResult) -> Prediction:
        """DetectionResult(모델 쪽 결과) → Prediction(백엔드 쪽 결과). 판정 근거까지 그대로 넘긴다."""
        return Prediction(anomaly_score=r.score, is_anomaly=r.is_anomaly, reason=r.reason,
                          details=r.details, model_name=r.model, threshold=THRESHOLD)

    def predict(self, events: tuple[int, ...]) -> Prediction:
        """실시간 판정: 방금 들어온 이벤트(튜플의 마지막)를 직전 이력과 함께 판정한다."""
        return self._to_prediction(self.detector.score_event(self._to_event_ids(events)))

    def predict_close(self, events: tuple[int, ...]) -> Prediction:
        """세션 종료 판정: 이 블록이 "여기서 끝나도 정상인가" (EOS 판정). 이상 세션의 약 18% 는 이것으로만 잡힌다."""
        return self._to_prediction(self.detector.score_close(self._to_event_ids(events)))
