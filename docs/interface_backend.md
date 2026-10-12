# 추론 인터페이스 안내 (안명근 · 이동준 참고)

- 작성: 옥상민 / 2026-10-12
- 모델(DeepLog/Autoencoder)이 완성되기 전에도 **같은 인터페이스**로 통계 빈도 베이스라인이 동작하므로, FastAPI·대시보드 연동을 지금 바로 시작할 수 있다. 모델이 나오면 `load_detector("deeplog")` 처럼 이름만 바꾸면 된다.

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
| `score_session(events, closed=True)` | 블록이 끝났을 때 전체 판정. DeepLog 는 `closed=True` 일 때 '너무 일찍 끝난 세션'까지 검사 예정 |

`DetectionResult` (`result.to_dict()` 로 JSON 변환 가능)

| 필드 | 타입 | 설명 |
|---|---|---|
| `score` | float 0~1 | 이상 점수. **0.5 가 임계치** (모든 모델 공통) → 대시보드 기준선 0.5 |
| `is_anomaly` | bool | `score > 0.5` 와 같은 의미 (정확히 0.5 는 정상) |
| `reason` | str | `unknown_event`(학습에 없던 이벤트) / `model`(모델 판정) / `normal` |
| `model` | str | `statistical_frequency`, `deeplog`, `autoencoder` |
| `details` | dict | 판정 근거. 통계 빈도: `event_id, count, threshold, template` / DeepLog: `position, actual{event_id, template, prob, rank}, topk[...]` / Autoencoder: `position, reconstruction_error, threshold, window, worst{offset, actual, reconstructed}` |

- 속도 (CPU): 통계 빈도 0.06ms/세션, DeepLog 0.74ms/세션 · 0.23ms/이벤트, Autoencoder 0.11ms/세션 · 0.09ms/이벤트 (목표 p95 1초 대비 여유 큼)
- **Autoencoder 사용 가능 (2026-10-12):** `load_detector("autoencoder")`. 최종 모델은 6단계에서 확정
- **DeepLog 사용 가능 (2026-10-12):** `load_detector("deeplog")`. `details` 예시:
  `{"position": 5, "actual": {"event_id": 13, "template": "Verification succeeded for <:BLK:>", "prob": 0.0003, "rank": 7}, "topk": [{"event_id": 3, "template": "...", "prob": 0.61}, ...]}`
  세션이 너무 일찍 끝난 경우 `actual.template` 이 `<EOS: 세션 종료>` 로 표시된다 (`score_session(..., closed=True)` 일 때만).
- 세션 종료 시점: HDFS 로그에는 '블록 끝' 표시가 없으므로 일정 시간 새 로그가 없으면 종료로 보는 타임아웃 방식을 백엔드에서 정한다 (제안: 마지막 로그 후 N초).

## SQLite 저장 제안 (계획서 Ⅱ-1 ④)

`timestamp, block_id, event_id, template, anomaly_score(=result.score), reason, model, raw_log, alerted`

## 문의

인터페이스 변경이 필요하면 `src/logguard/models/inference.py` 를 직접 고치지 말고 이슈/PR 로 요청 (모델 쪽과 동시에 바뀌어야 함).
