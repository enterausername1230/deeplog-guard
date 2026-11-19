# FastAPI 백엔드 인수인계 (안명근 담당)

현재 범위: HDFS 로그 수신 → 기존 Drain3 템플릿 매칭 → **Autoencoder 실시간 판정** → SQLite 저장 → 조회,
그리고 **세션 종료(EOS) 판정**. Autoencoder/DeepLog 학습, Streamlit 화면, Slack 발송은 이 구현에 포함하지 않는다.

> 2026-10-20 통합 (옥상민): 모델 연결 어댑터(`predictor.py`), DB 구조 통일(`schema.sql`), 세션 종료 판정
> (`POST /sessions/close`, 자동 idle), `sent_at`·`latency_ms` 기록 추가. DB 설명: `docs/db_schema.md`

## 실행

프로젝트 공통 환경(Python 3.11, `make setup`)에서 실행한다. 모델(torch)까지 필요하므로 공통 `requirements.txt` 를 쓴다.

```bash
make setup                      # 최초 1회 (uv 필요)
make api                        # = uvicorn logguard.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

`src/logguard/api/requirements.txt` 는 모델 없이 수집만 할 때(`model: null`)의 최소 환경이다.

- API 설명 및 직접 요청: http://127.0.0.1:8000/docs
- 상태: `GET /health` (`model: autoencoder`, `model_ready: true` 면 판정 중. `false` 면 수집만)
- 설정: `configs/api.yaml`; 다른 파일은 `LOGGUARD_API_CONFIG` 환경변수로 지정.
- 데이터베이스: 기본 `artifacts/api/logguard.sqlite` (실행 시 생성, Git 제외).
- 필요 파일: `configs/drain3.ini`, `artifacts/drain3/drain3_state.bin`, `data/processed/event_vocab.json`,
  `artifacts/models/autoencoder.pt` (모두 git 포함).
- 모델: `configs/api.yaml` 의 `model` (`autoencoder` 기본 / `deeplog` / `null`=수집만).
  `window_size` 는 모델 윈도우(20) 이상이어야 하며, 작거나 어휘가 모델과 다르면 **서버가 시작을 거부한다**
  (window_size 10 이면 정상 세션 대부분이 이상으로 판정되는 것을 2026-10-20 통합 시험에서 확인).
- 원본 HDFS.log나 학습 parquet 없이도 수신 서버를 시작할 수 있다.
- 실행 환경은 위 명령으로 별도 설치한다. 검증용 임시 가상환경은 배포 환경이 아니다.

## API 계약

```bash
curl -X POST http://127.0.0.1:8000/logs \
  -H 'Content-Type: application/json' \
  -d '{"raw_log":"081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating"}'
```

`POST /logs`는 HDFS 로그 한 줄을 받아 201과 `{"records": [...]}`를 반환한다.
한 줄에 여러 Block ID가 있으면 블록마다 한 행을 저장한다. 같은 줄에서 중복된 Block ID는 한 번만 저장한다.
`timestamp`는 서버 수신 시각(UTC), `log_date`/`log_time`은 원본 HDFS 날짜/시간 문자열이다.

| status | 의미 |
|---|---|
| pending_model | 모델 미연결(`model: null`). 점수와 정상/이상 판정은 null |
| no_block | Block ID가 없어 시퀀스 추론 대상이 아님. 판정은 null |
| normal | 연결한 모델이 해당 입력을 정상으로 판정 |
| anomaly | 연결한 모델이 해당 입력을 이상으로 판정 |

- 알 수 없는 템플릿: `event_id=null`, `event_index=1(UNK)`.
- 템플릿은 있지만 학습 어휘에 없으면 해당 event_id를 유지하며 event_index는 UNK.
- 판정된 행에는 `model_name`, `threshold`(0.5), `reason`, `details`(판정 근거 JSON), `latency_ms` 가 함께 저장된다.
- 요청 본문에 `sent_at`(UTC ISO)을 넣으면 그대로 저장된다 (재생 모듈이 자동으로 넣음, E2E 지연 계산용).
- `alerted=false`는 발송하지 않았다는 뜻이다. 백엔드는 Slack을 발송하지 않는다 (Slack 알림 작업자: 이동준, `docs/db_schema.md`).
- 형식 오류: 422. 설정된 길이 제한 초과: 413 (스키마 상한 65,536자 초과는 422).
- 모델 실패: 503. 해당 요청의 여러 블록 저장을 전부 취소한다.
- 같은 요청을 다시 보내면 별도로 저장된다. 전송 실패 시 무조건 재시도하지 않는다.

조회:

- `GET /logs?after_id=0&limit=100`: ID 오름차순, 최대 1,000행. 마지막 id를 after_id로 넘겨 다음 페이지 조회.
- `block_id=blk_...`, `status=anomaly`로 필터링 가능.
- `GET /stats`: 상태별 저장 행 수. 여러 블록을 가진 한 줄이 여러 행이므로 원본 로그 줄 수와 다를 수 있다.

## 모델 연결 (2026-10-20 완료)

`predictor.py` 의 `DetectorPredictor` 가 `inference.py` 의 `Predictor` 프로토콜을 구현한다.
서버 시작 시 `configs/api.yaml` 의 `model` 로 자동 생성되며, 테스트에서는 `create_app(predictor=...)` 로 주입할 수 있다.

- `predict(events)`: **현재 이벤트를 포함한**, 오래된 순서의 어휘 인덱스 튜플 (최대 `window_size + 1`)
  → 원본 Event ID 로 되돌려 `Detector.score_event()` → `Prediction(점수, 이상 여부, reason, details, model_name, threshold)`
- `predict_close(events)`: 세션 종료 판정. 최근 `window_size` 개로 "여기서 끝나도 정상인가(EOS)" 만 본다
  (최근 20개만으로 한 판정 == 세션 전체로 한 오프라인 판정, test 11.5만 세션 100% 일치 확인)

각 응답은 해당 시점의 윈도우 판정이다. 세션 판정 = 블록별 OR + `session_results` (`docs/db_schema.md`).
`tests/test_integration.py` 가 실제 test 블록 로그를 흘려 오프라인 평가 판정과 100% 일치하는지 검사한다.

## 세션 종료 판정

- `POST /sessions/close` : 열린 세션 전부 종료 판정 (`?idle_lines=N` 이면 N 행 넘게 소식 없는 블록만)
- `GET /sessions?anomalies_only=true` : 종료 판정 결과
- 자동: `session_sweep_every` 행마다 `session_idle_lines` 행 넘게 소식 없는 블록을 종료 판정 (`api.yaml`)

## 로그 재생

```bash
PYTHONPATH=src .venv/bin/python -m logguard.api.replay data/raw/HDFS_v1/HDFS.log --rate 10 --limit 100
```

- `--url`, `--rate`(초당 로그 줄), `--timeout`, `--limit` 지정 가능.
- `--close-sessions`: 재생이 끝나면 `POST /sessions/close` 로 남은 세션을 종료 판정 (최종 F1 0.977 재현에 필요).
- 요청마다 `sent_at` 을 함께 보낸다.
- 요청을 순서대로 전송한다. 지정 속도는 상한이며 서버 응답이 느리면 실제 속도도 낮아진다.
- 평가용: `--test-sessions data/processed/test.parquet` 추가. pandas/pyarrow 설치 필요.
- 여러 블록 중 하나라도 test 소속이 아닌 줄은 제외한다. 따라서 전체 원본 평가 세션과 완전히 동일한 시퀀스라는 보장은 없다.
- HTTP 오류는 즉시 중단하며 자동 재시도하지 않는다.

## 테스트 및 성능 측정

```bash
.venv/bin/python -m pytest tests/test_api.py -q
.venv/bin/locust -f src/logguard/api/locustfile.py --host http://127.0.0.1:8000 \
  --headless -u 50 -r 10 -t 70s --csv /tmp/logguard-load
```

Locust는 사용자당 초당 10건을 목표로 한다. 50명은 최대 목표 500 logs/s이며 달성 보장이 아니다.
상승 구간 이후 60초 처리량, 오류율, p95를 실제 보고서로 확인해야 한다.
현재 샘플은 동일 블록 반복이므로 연결된 실제 모델/다양한 로그로 최종 측정을 다시 해야 한다.
모델 미연결 측정은 추론 지연이나 Slack 포함 End-to-End 목표를 검증하지 못한다.

## 현재 운영 범위

- 로컬 개발용. 인증과 외부 공개 배포는 아직 구현하지 않았다.
- 블록 순서 보장을 위해 **workers=1**, DB를 쓰는 서버 인스턴스도 하나로 실행한다.
- Drain 상태에는 `match()`만 호출하며 새 템플릿을 학습하거나 상태 파일을 덮어쓰지 않는다.
- 요청 내 모든 블록 저장은 하나의 트랜잭션이다. 판정기는 외부 부작용 없는 추론이어야 한다.
- 별도 백엔드 설정 로더가 실제 프로젝트 루트를 기준으로 경로를 읽는다. 공통 `config.py`의 기존 경로 처리 코드는 변경하지 않았다.
- SQLite 데이터 보관/삭제 정책은 추후 합의해야 한다.
- 처리량 (2026-10-20, Mac M2, Autoencoder 연결): 실제 uvicorn 서버에 재생 모듈 1개 연결로 순차 전송 시
  **약 660줄/초**, 서버 내부 지연(수신 → 판정) p95 0.53ms. 목표 500 logs/s 를 넘지만 여유가 크지 않다.
  요청마다 SQLite 연결을 새로 열고 저장 행을 다시 읽어 오는 부분이 주된 비용이므로, Locust(동시 50명) 실측에서
  부족하면 연결 재사용·배치 저장으로 개선한다 (담당 안명근).


## 2026-11-10 백엔드 보완

변경 범위는 `src/logguard/api/` 내부다. 모델 구현, 공용 설정, DB 테이블 구조,
대시보드 및 Slack 알림 코드는 변경하지 않았다.

- **수집 모드의 모델 의존성 분리:** `model: null`이면 모델 어댑터를 가져오지 않는다.
  모델 없는 최소 환경에서도 API를 시작할 수 있다. 실제 모델 모드에는 기존처럼 모델 패키지가 필요하다.
- **추가 로그의 종료 재판정:** 종료된 블록에 새 로그가 오면 다음 수동/idle 종료 검사에서
  최신 이력으로 재판정하고 기존 `(block_id, model_name)` 행을 갱신한다.
  새 로그가 없으면 반복 종료 요청은 `closed: 0`을 반환한다. DB 구조와 행 수 규칙은 유지한다.
  갱신 전까지 조회에는 직전 종료 결과가 보이며, 이미 전송된 Slack 알림을 취소하거나 재전송하지 않는다.
- **오류 응답:** 조회·세션 종료의 SQLite 운영 오류는 내부 경로를 노출하지 않고 503으로 반환한다.
  종료 모델 호출이 실패하면 503을 반환하고 해당 종료 요청에서 저장한 결과 전체를 취소한다.
  로그 수신 자체의 기록은 보존한다. 자동 idle 종료 오류는 기존처럼 기록하고 로그 수신은 성공 처리한다.
- **재생 입력 검사:** 속도·타임아웃의 NaN 및 무한대 값을 거부한다.
- **회귀 테스트:** `tests/test_backend_regressions.py`에 수집 모드 의존성, 재시작 후 재판정,
  종료 실패의 일괄 취소, 조회/종료 DB 장애, 잘못된 재생 인자 검사를 추가했다.

백엔드 폴더에서 검증하는 명령 (프로젝트 기본 환경 대신 이 작업의 독립 검증 환경 사용):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../.. .venv-check/bin/python -m pytest \
  ../../../tests/test_api.py ../../../tests/test_integration.py tests \
  -q -p no:cacheprovider --basetemp=.check-output/pytest
```

`.venv-check/`는 로컬 검증용 패키지 환경, `.check-output/`은 임시 DB·설정·테스트 결과다.
폴더 내부 `.gitignore`로 Git에서 제외하며, 다른 팀원이 받는 소스에 포함되지 않는다.
공용 가상환경은 이번 작업에서 수정하지 않았다.

### 검증 결과 (2026-11-10)

- 기존 백엔드 테스트 + 실제 모델 연동 테스트 + 신규 회귀 테스트: **23 passed, 1 skipped**.
  건너뛴 항목은 원본 HDFS/test parquet가 필요한 120개 블록의 오프라인 대비 일치 검사다.
- 모델 패키지가 없는 기존 최소 가상환경에서도 기존 API 및 신규 회귀 검사 **18 passed**.
- 실제 Uvicorn 단일 워커 + Autoencoder: `/health`, 10줄 로그 재생, SQLite 조회,
  10개 세션 종료를 확인했다. 기존 서비스 DB 대신 `.check-output/live.sqlite`를 사용했다.
- Locust: 50명, 초당 10명 증가, 실행 제한 20초. 최종 콘솔 집계 **7,600 요청, 실패 0**,
  평균 **442.43 요청/초**, 평균 응답 **35ms**, p95 **55ms**, p99 **80ms**.
  CSV는 종료 직전 주기 집계(7,502건)이며 최종 콘솔 집계와 시점이 다르다.
- 최종 DB는 재생 10건 + 부하 7,600건 = **7,610행**, 종료 결과 **11개**로 확인했다.
- 동일 샘플·동일 블록을 반복한 단기 로컬 부하 시험이다. 실제 데이터 분포의 처리량,
  지속적인 500건/초, 전체 데이터 탐지 정확도 또는 Slack까지의 지연을 입증하지 않는다.
- 테스트 서버는 종료했다. 공용 환경·실서비스 DB·모델 파일은 변경하지 않았다.
- 테스트 도구의 httpx 사용 중단 예정 경고 1건이 있었으나 테스트 실패는 없었다.
