# 추론 인터페이스 안내 (안명근 · 이동준 참고)

- 작성: 옥상민 / 2026-10-12
- **최종 채택 모델: `autoencoder` (2026-10-12, test F1 0.977).** 코드에서는 `from logguard.models.inference import RECOMMENDED_DETECTOR` 후 `load_detector(RECOMMENDED_DETECTOR)` 를 쓰면 이후 모델이 바뀌어도 수정이 필요 없다.
- 세 Detector(`statistical_frequency`, `deeplog`, `autoencoder`)는 같은 인터페이스라 비교·시연 때 이름만 바꾸면 된다.
- **2026-10-20 백엔드 연결 완료:** `src/logguard/api/predictor.py` 의 `DetectorPredictor` 가 백엔드 `Predictor` 약속을 구현하고,
  서버가 `configs/api.yaml` 의 `model` 로 자동 연결한다. DB 구조는 `docs/db_schema.md` (백엔드·대시보드 공용 `api/schema.sql`).

## 준비

```bash
make setup && make data                               # 데이터 (최초 1회)
python -m logguard.models.inference --export statistical_frequency   # 또는 make export-detector
```

`artifacts/drain3/drain3_state.bin`, `artifacts/models/statistical_frequency.json` 은 git 에 포함되어 있어서 **데이터 없이도** 아래 코드가 동작한다.

## 1. 로그 한 줄 → Event ID (`logguard.data.online.OnlineParser`)

```python
from logguard.data.online import OnlineParser

parser = OnlineParser.load()          # 서버 시작 시 1회
log = parser.parse_line(raw_line)     # 로그 1줄마다
log.event_id     # int | None   (None = 처음 보는 형태 → 이상으로 처리됨)
log.block_ids    # ["blk_-160..."]  한 줄에 여러 블록이 있을 수 있음
log.level, log.component, log.content, log.date, log.time
```

- 학습 때 저장한 Drain 상태로 **매칭만** 한다 (새 템플릿 학습 X) → 학습/추론 Event ID 동일 보장
- 검증: HDFS.log 139만 줄 대조 결과 오프라인 파싱과 불일치 0건, 평균 8.4µs/줄

## 2. 블록별 이벤트 → 이상 판정 (`logguard.models.inference`)

```python
from logguard.models.inference import load_detector

detector = load_detector("statistical_frequency")    # 서버 시작 시 1회
block_events: dict[str, list[int | None]] = {}

for blk in log.block_ids:                             # 로그 1줄마다
    block_events.setdefault(blk, []).append(log.event_id)
    result = detector.score_event(block_events[blk])
    if result.is_anomaly:
        ...  # SQLite 저장 + Slack 알림
```

| 메서드 | 용도 |
|---|---|
| `score_event(history)` | 스트리밍 판정. 블록의 지금까지 이벤트 목록을 넣으면 마지막 이벤트 시점 기준으로 판정 |
| `score_session(events, closed=True)` | 세션 전체 이벤트로 판정 (오프라인 평가용). `closed=True` 면 '너무 일찍 끝난 세션'까지 검사 |
| `score_close(history)` | **세션 종료 판정 (2026-10-20 추가).** 최근 w(20)개만 받아 EOS 위치 하나만 판정. 백엔드처럼 이력 일부만 가진 곳에서 사용. 실시간 판정 OR `score_close` == 전체 `score_session(closed=True)` (test 11.5만 세션 100% 일치) |

`DetectionResult` (`result.to_dict()` 로 JSON 변환 가능)

| 필드 | 타입 | 설명 |
|---|---|---|
| `score` | float 0~1 | 이상 점수. **0.5 가 임계치** (모든 모델 공통) → 대시보드 기준선 0.5 |
| `is_anomaly` | bool | `score > 0.5` 와 같은 의미 (정확히 0.5 는 정상) |
| `reason` | str | `unknown_event`(학습에 없던 이벤트) / `model`(모델 판정) / `normal` |
| `model` | str | `statistical_frequency`, `deeplog`, `autoencoder` |
| `details` | dict | 판정 근거. 통계 빈도: `event_id, count, threshold, template` / DeepLog: `position, actual{event_id, template, prob, rank}, topk[...]` / Autoencoder: `position, reconstruction_error, threshold, window, worst{offset, actual, reconstructed}` |

- 속도 (CPU, 원본 1줄 파싱 + 판정, 10만 줄 측정): Autoencoder p95 **0.10ms** · 약 10,700줄/초, DeepLog p95 0.57ms · 약 1,900줄/초, 통계 빈도 p95 0.08ms. 목표(p95 1초, 500줄/초) 대비 모델 구간은 여유가 크므로 E2E 지연은 HTTP·DB·Slack 쪽에서 관리하면 된다
- **Autoencoder 사용 가능 (2026-10-12):** `load_detector("autoencoder")`. 최종 모델은 6단계에서 확정
- **DeepLog 사용 가능 (2026-10-12):** `load_detector("deeplog")`. `details` 예시:
  `{"position": 5, "actual": {"event_id": 13, "template": "Verification succeeded for <:BLK:>", "prob": 0.0003, "rank": 7}, "topk": [{"event_id": 3, "template": "...", "prob": 0.61}, ...]}`
  세션이 너무 일찍 끝난 경우 `actual.template` 이 `<EOS: 세션 종료>` 로 표시된다 (`score_session(..., closed=True)` 일 때만).
- 세션 종료 시점 (2026-10-20 결정): HDFS 정상 블록은 이벤트 사이가 수십만~130만 줄 벌어진다. 그래서 "초" 가 아니라 "행 수" 로
  idle 을 판단한다 (`session_idle_lines` 150만, 재생 속도와 무관). 시연·평가는 재생 종료 후 `POST /sessions/close` 로 한꺼번에 판정.
  실시간 경보만: test F1 0.880 / 세션 종료 판정 포함: 0.977

## SQLite 저장 (계획서 Ⅱ-1 ④) — 2026-10-20 통합 규격으로 확정

`docs/db_schema.md` 참고 (`logs`, `session_results`, `alerts`).

## 문의

인터페이스 변경이 필요하면 `src/logguard/models/inference.py` 를 직접 고치지 말고 이슈/PR 로 요청 (모델 쪽과 동시에 바뀌어야 함).
