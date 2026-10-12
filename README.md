# DeepLog-Guard — AI 기반 실시간 서버 로그 이상 탐지 및 알림 시스템

동의대학교 컴퓨터공학과 딥러닝 팀 프로젝트 **7조** (담당교수 임환섭, 2026.09.11 ~ 2026.12.21)

HDFS 로그를 Drain으로 파싱 → Block ID 세션 → 슬라이딩 윈도우 → **Autoencoder vs DeepLog(LSTM)** 이상 탐지 →
FastAPI 실시간 추론 → SQLite 저장 → Slack 알림 + Streamlit 대시보드.

## 목표 (계획서 기준)

| 항목 | 목표 |
|---|---|
| F1-score (HDFS, 세션 단위) | **≥ 0.85** (Precision·Recall 병기) |
| Rule-Based Baseline 대비 | **F1 +10%p 이상** |
| 추론 지연 (FastAPI 수신 → 판정) | p95 **≤ 1초** |
| End-to-End (Replay 전송 → Slack 요청 완료) | **≤ 2초** |
| 처리량 (Locust) | **500 logs/s × 1분 이상**, 오류율 < 1% |

## 팀 / 담당 폴더

| 이름 | 역할 | 주 작업 폴더 |
|---|---|---|
| 옥상민 (팀장) | AI/Data · 총괄 | `src/logguard/data`, `models`, `baseline`, `eval` |
| 안명근 | Backend | `src/logguard/api` (FastAPI, Log Replay, SQLite, Locust) |
| 이동준 | Dashboard | `src/logguard/dashboard` (Streamlit, Slack Webhook) |

## 폴더 구조

```
dplngproj/
├── configs/
│   ├── config.yaml          # 경로·분할 비율·윈도우·베이스라인 설정 (모든 코드가 여기서 읽음)
│   └── drain3.ini           # Drain3 마스킹 규칙 / sim_th / depth
├── src/logguard/
│   ├── config.py            # load_config(), 프로젝트 루트 경로
│   ├── data/
│   │   ├── download.py      # HDFS_v1 다운로드·압축 해제
│   │   ├── parse.py         # Drain3 파싱 → 라인/템플릿/세션 테이블 + Drain 상태 저장
│   │   ├── split.py         # Block ID 단위 train/val/test 분할 + 이벤트 어휘
│   │   └── windows.py       # 슬라이딩 윈도우 (DeepLog / Autoencoder 입력)
│   ├── baseline/rule_based.py   # 키워드·정규식 + 빈도 임계치 베이스라인
│   ├── eval/metrics.py      # Precision/Recall/F1, 백분위수 임계치
│   ├── models/              # Autoencoder, DeepLog (예정)
│   ├── api/                 # FastAPI 추론 서버 (예정)
│   └── dashboard/           # Streamlit + Slack (예정)
├── tests/                   # pytest (원본 데이터 없이 실행)
├── notebooks/               # EDA·실험 노트북
├── data/                    # ⚠️ git 미포함 — `make data` 로 생성
│   ├── raw/HDFS_v1/         # HDFS.log (1.5GB), preprocessed/anomaly_label.csv
│   ├── interim/             # 파싱 결과
│   └── processed/           # 분할 결과
├── artifacts/drain3/        # 온라인 추론용 Drain3 상태 (git 포함)
├── results/                 # 모델·베이스라인 평가 결과 json
└── docs/                    # 계획서 등 문서
```

## 시작하기

```bash
# 1) 환경 (uv 필요: brew install uv)
make setup

# 2) 데이터: 다운로드 → Drain 파싱 → 세션 분할  (파싱 수 분 소요)
make data

# 3) Rule-Based Baseline
make baseline

# 4) 테스트
make test
```

`make` 없이 실행: `.venv/bin/python -m logguard.data.parse --limit 100000` 처럼 모듈 단위 실행 가능 (`--limit` 은 빠른 테스트용).

## 데이터 규칙 (누수 방지 — 반드시 지킬 것)

- 분할 단위는 **Block ID 세션**. 같은 블록의 윈도우가 서로 다른 셋에 들어가면 안 된다.
- **train = 정상 세션만**. 이상 세션은 버리고 개수만 `split_summary.json` 에 기록.
- **임계치 = val 의 정상 세션(label==0) 이상 스코어의 백분위수** (`eval.metrics.percentile_threshold`).
- **test = 실제 클래스 분포 유지** (기본 `strategy: random` 은 라벨별 층화 분할, `chrono` 는 시간순).
- 최종 지표는 **세션 단위** (세션 내 윈도우 하나라도 이상이면 세션 이상)로 `eval.metrics.prf()` 로 계산.

## 데이터 파일 명세

| 파일 | 내용 |
|---|---|
| `data/interim/hdfs_lines.parquet` | 원본 라인 순서대로 `date, time, level, component, event_id` (행 번호 = HDFS.log 줄 번호, 0부터) |
| `data/interim/hdfs_templates.csv` | `event_id, event_key(E#), template, occurrences, levels` |
| `data/interim/hdfs_sessions.parquet` | `block_id, first_line, last_line, length, events(list[int]), label(0/1)` |
| `data/processed/{train,val,test}.parquet` | 위 세션 테이블의 분할본 |
| `data/processed/event_vocab.json` | 모델 입력 인덱스 (0=PAD, 1=UNK, train 에 나온 이벤트만 2부터) |
| `artifacts/drain3/drain3_state.bin` | FastAPI 에서 `TemplateMiner` 로 로드해 `match()` 로 Event ID 매칭 |

## 현재 결과 (HDFS test, 세션 단위)

데이터: 세션 575,061개 · 이상 16,838개(2.93%) · Drain 템플릿 46개 · random 층화 분할 6:2:2 (seed 42)

| 모델 | 설정 (val F1 최대로 선택) | Precision | Recall | F1 | 파일 |
|---|---|---|---|---|---|
| Rule-Based (목표 ② 비교 기준) | 키워드·레벨 + 세션 길이 p99 | 0.083 | 0.684 | 0.148 | `results/baseline_rule_based.json` |
| 통계 빈도 베이스라인 (참고) | 이벤트별 빈도 p99.95 | 0.955 | 0.828 | **0.887** | `results/baseline_statistical_frequency.json` |
| Autoencoder | — | | | | 예정 |
| DeepLog (LSTM) | — | | | | 예정 |

- 키워드 규칙은 정상 블록에서도 흔한 WARN(`Got exception while serving`) 때문에 오탐이 많다 → 계획서에서 말한 규칙 기반의 한계
- 통계 빈도 베이스라인이 이미 목표 F1 0.85 를 넘으므로, 딥러닝 모델은 **0.887 이상**을 실질적인 기준으로 삼는다

## 협업 규칙

- `main` 직접 push 금지 → `feat/<영역>-<내용>` 브랜치 + PR, 1명 이상 리뷰 후 merge
- 비밀값(Slack Webhook URL 등)은 `.env` 에만, **절대 커밋 금지**
- 설정값은 코드에 하드코딩하지 말고 `configs/config.yaml` 에 추가
- 실험 결과는 `results/<model>.json` 으로 저장해 베이스라인과 같은 형식으로 비교

## 참고

- Du et al., *DeepLog*, CCS 2017 · He et al., *Drain*, ICWS 2017 · He et al., *Loghub*, arXiv:2008.06448

## Streamlit 대시보드

```bash
# make setup으로 환경을 구성한 후
make dashboard
# 또는
.venv/bin/python -m streamlit run src/logguard/dashboard/app.py
```

기본은 **데모 데이터** 모드입니다. DB나 학습된 모델 없이 화면을 확인할 수 있습니다.
데모 로그·스코어·지연·Slack 이력은 합성 데이터이며, 모델 평가 탭은 실제 `results/*.json`을 읽습니다.
대시보드는 알림을 전송하지 않으며 SQLite를 읽기 전용으로 조회합니다.

- 상단: 관측 세션 수, 이상 세션 수·비율, Slack 전송 성공 수
- 실시간 모니터링: 모델별 스코어·임계치, 지연 p95, 최근 추론 윈도우, CSV 다운로드
- 세션 상세: Block ID별 로드된 윈도우 이력과 이벤트 템플릿
- 알림 이력: 전송 성공·실패·대기 기록
- 모델 평가: test 세션 단위 Precision/Recall/F1와 혼동행렬 집계값
- 사이드바: 데이터 소스, 조회 기간, Block ID 검색, 이상 세션 필터, 자동 갱신

### 백엔드 연동 규격

`src/logguard/dashboard/schema.sql`을 DB 규격으로 사용합니다. 백엔드에서 스키마를 생성하고 기록하며,
대시보드는 생성·수정하지 않습니다. 기본 DB 경로는 `configs/config.yaml`의
`dashboard.database_path` (`data/logguard.sqlite3`)입니다. 사이드바에서도 경로를 지정할 수 있습니다.

`predictions`는 **추론된 윈도우당 한 행**입니다. `event_id`와 `raw_log`는 해당 윈도우의
마지막 이벤트·원본 로그입니다. `timestamp`는 Replay 수신 시각을 UTC ISO-8601 형식으로 저장합니다.
`model_name`, `anomaly_score`, `threshold`, `is_anomaly`, `latency_ms`를 함께 저장합니다.
`is_anomaly`는 모델이 결정한 판정값을 사용하고, 세션 판정은 **동일 모델·Block ID의 윈도우 OR**로 집계합니다.
서로 다른 모델의 스코어는 같은 차트에 섞지 않습니다.

알림 전송은 백엔드/알림 작업자가 담당하고, `alerts` 테이블에 전송 시도별 시각·세션·모델·상태·메시지를 기록합니다.
중복 방지는 작업자에서 처리해야 합니다. `alerts` 테이블이 없으면 빈 알림 이력으로 표시합니다.
Slack Webhook 비밀값은 기존 규칙대로 환경변수 또는 `.env`에 보관합니다.

조회 기간은 **최신 기록 시각**을 기준으로 계산하여 과거 Replay도 조회할 수 있습니다.
조회 구간에 등장한 세션의 판정은 로드된 전체 이력으로 집계합니다. 같은 블록은 모델별로 별도 집계합니다.
최대 조회 행 수는 `dashboard.max_rows`이며 상한에 도달하면 경고합니다. 상한 이전 이력은 집계에 포함되지 않습니다.
세션 종료 신호가 아직 없으므로 실시간 화면은 관측된 세션의 잠정 상태이며, 최종 F1은 모델 평가 탭에서 별도로 제공합니다.

대시보드 검증: `.venv/bin/python -m pytest tests/test_dashboard.py -q`.
