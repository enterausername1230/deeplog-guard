# FastAPI 백엔드 인수인계 (안명근 담당)

현재 범위: HDFS 로그 수신 → 기존 Drain3 템플릿 매칭 → SQLite 저장 → 조회.
모델 담당자가 Predictor를 연결하면 블록별 최근 이벤트를 전달해 판정 결과도 저장한다.
Autoencoder/DeepLog 학습, Streamlit 화면, Slack 발송은 이 구현에 포함하지 않는다.

## 실행

프로젝트 루트에서 Python 3.12 기준:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r src/logguard/api/requirements.txt
.venv/bin/python -m uvicorn logguard.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

- API 설명 및 직접 요청: http://127.0.0.1:8000/docs
- 상태: `GET /health` (`model_ready: false`이면 수집만 가능)
- 설정: `configs/api.yaml`; 다른 파일은 `LOGGUARD_API_CONFIG` 환경변수로 지정.
- 데이터베이스: 기본 `artifacts/api/logguard.sqlite` (실행 시 생성, Git 제외).
- 필요 파일: `configs/drain3.ini`, `artifacts/drain3/drain3_state.bin`, `data/processed/event_vocab.json`.
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
| pending_model | 모델 미연결. 점수와 정상/이상 판정은 null |
| no_block | Block ID가 없어 시퀀스 추론 대상이 아님. 판정은 null |
| normal | 연결한 모델이 해당 입력을 정상으로 판정 |
| anomaly | 연결한 모델이 해당 입력을 이상으로 판정 |

- 알 수 없는 템플릿: `event_id=null`, `event_index=1(UNK)`.
- 템플릿은 있지만 학습 어휘에 없으면 해당 event_id를 유지하며 event_index는 UNK.
- `alerted=false`는 발송하지 않았다는 뜻이다. 백엔드는 Slack을 발송하지 않는다.
- 형식 오류: 422. 설정된 길이 제한 초과: 413 (스키마 상한 65,536자 초과는 422).
- 모델 실패: 503. 해당 요청의 여러 블록 저장을 전부 취소한다.
- 같은 요청을 다시 보내면 별도로 저장된다. 전송 실패 시 무조건 재시도하지 않는다.

조회:

- `GET /logs?after_id=0&limit=100`: ID 오름차순, 최대 1,000행. 마지막 id를 after_id로 넘겨 다음 페이지 조회.
- `block_id=blk_...`, `status=anomaly`로 필터링 가능.
- `GET /stats`: 상태별 저장 행 수. 여러 블록을 가진 한 줄이 여러 행이므로 원본 로그 줄 수와 다를 수 있다.

## 모델 담당자와 연결할 부분

`inference.py`의 `Predictor` 프로토콜을 구현하고 `create_app(predictor=adapter)`로 앱을 만든다.
`predict(events)`에는 **현재 이벤트를 포함한**, 오래된 순서의 어휘 인덱스 튜플을 전달한다.
길이는 최대 `window_size + 1`; 초기에는 더 짧다. 블록별 이전 이벤트는 DB에서 읽으므로 재시작 뒤에도 이어진다.
DeepLog는 마지막 이벤트를 목표로, 나머지를 문맥으로 사용한다. Autoencoder는 필요한 마지막 구간을 사용한다.
PAD 추가, 텐서 변환, 모델 로드, 임계치 적용은 모델 어댑터가 담당한다.
반환값은 `Prediction(anomaly_score=유한한 실수, is_anomaly=bool)`이다.
현재 기본 앱에는 모델이나 가짜 판정기를 연결하지 않았다.

각 응답은 해당 시점의 윈도우 판정이다. 세션 F1을 계산할 때는 모델 담당자가 블록별 OR 집계를 적용해야 한다.

## 로그 재생

```bash
PYTHONPATH=src .venv/bin/python -m logguard.api.replay data/raw/HDFS_v1/HDFS.log --rate 10 --limit 100
```

- `--url`, `--rate`(초당 로그 줄), `--timeout`, `--limit` 지정 가능.
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
- SQLite 데이터 보관/삭제 정책과 세션 종료 판정은 추후 합의해야 한다.
