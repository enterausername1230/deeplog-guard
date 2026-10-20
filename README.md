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
│   │   ├── drain_tuning.py  # Loghub 공식 템플릿 대비 Drain 검증·sim_th/depth 튜닝
│   │   ├── online.py        # 실시간 파서: 로그 1줄 → Event ID, Block ID (백엔드용)
│   │   └── windows.py       # 슬라이딩 윈도우 (DeepLog / Autoencoder 입력)
│   ├── baseline/rule_based.py   # 키워드·정규식 + 빈도 임계치 베이스라인
│   ├── eval/
│   │   ├── metrics.py       # Precision/Recall/F1, 백분위수 임계치
│   │   ├── final_eval.py    # 6단계 test 1회 평가·목표 달성 정리
│   │   └── figures.py       # 보고서 그래프 → docs/figures/
│   ├── models/
│   │   ├── dataset.py       # 세션 → 윈도우 (중복 압축, EOS/UNK)
│   │   ├── scoring.py       # 세션 점수·임계치 선택·0~1 정규화·결과 저장
│   │   ├── train_utils.py   # 시드·장치·가중 학습 루프
│   │   ├── inference.py     # Detector 인터페이스 (백엔드용, docs/interface_backend.md)
│   │   ├── deeplog.py       # DeepLog LSTM + EOS, 학습·val 평가·Detector
│   │   ├── autoencoder.py   # MLP Autoencoder (원-핫 윈도우 재구성), Detector
│   │   ├── seed_study.py    # 시드별 성능 편차 측정
│   │   └── tune.py          # 하이퍼파라미터 그리드 × 시드 튜닝 (val)
│   ├── api/                 # 백엔드 (안명근): FastAPI 수신·판정·저장, 로그 재생, Locust
│   │   ├── main.py          #   POST/GET /logs, POST /sessions/close, GET /sessions, /stats
│   │   ├── predictor.py     #   모델 연결 어댑터 (Detector → Predictor)
│   │   ├── schema.sql       #   ★ 통합 DB 규격 (백엔드·대시보드 공용, docs/db_schema.md)
│   │   └── replay.py        #   HDFS.log 를 순서대로 재생 (--test-sessions, --close-sessions)
│   └── dashboard/           # 대시보드 (이동준): Streamlit 화면, 읽기 전용 SQLite 조회
├── tests/                   # pytest (원본 데이터 없이 실행)
├── notebooks/               # EDA·실험 노트북 (01_eda.ipynb: 데이터 점검·Drain 검증)
├── data/                    # ⚠️ git 미포함 — `make data` 로 생성
│   ├── raw/HDFS_v1/         # HDFS.log (1.5GB), preprocessed/anomaly_label.csv
│   ├── interim/             # 파싱 결과
│   └── processed/           # 분할 결과
├── artifacts/               # drain3/ 상태, models/*.json Detector 설정 (git 포함)
├── results/                 # 모델·베이스라인 평가 결과 json
└── docs/                    # 개발 일정(schedule.md), 실행 계획(plan_week4-6.md), 백엔드 인터페이스(interface_backend.md), figures/
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

## 최종 결과 (HDFS test, 세션 단위) — 2026-10-12 6단계

데이터: 세션 575,061개 · 이상 16,838개(2.93%) · Drain 템플릿 52개(sim_th 0.85, Loghub 공식 시퀀스와 100% 일치) · Block ID 층화 분할 6:2:2 (seed 42).
모든 설정·임계치는 val 로만 선택했고 test 는 마지막에 1회 평가했다 (`results/final_test.{json,csv}`).

| 방법 | Precision | Recall | F1 (배포 모델) | F1 시드 3개 평균 ± 표준편차 |
|---|---|---|---|---|
| Rule-Based (키워드·정규식 + 세션 길이) | 0.976 | 0.406 | 0.573 | — |
| 통계 빈도 (이벤트별 빈도 임계치) | 0.955 | 0.828 | 0.887 | — |
| DeepLog + EOS (w=20, LSTM 2×128) | 0.898 | 0.956 | 0.926 | 0.924 ± 0.002 |
| **Autoencoder + EOS (w=20, 잠재 32) — 최종 채택** | **0.970** | **0.984** | **0.977** | **0.978 ± 0.001** |

| 계획서 목표 | 결과 | 달성 |
|---|---|---|
| ① F1 ≥ 0.85 | 0.977 | ✅ |
| ② Rule-Based 대비 F1 +10%p | +40.4%p (통계 빈도 대비로도 +9.0%p) | ✅ |
| ③ 추론 p95 ≤ 1초 | 로그 1줄 파싱 + 판정 p95 **0.10ms** (HTTP·DB 제외, E2E 2초는 백엔드 통합 후 측정) | ✅ (모델 구간) |
| ④ 500 logs/s | 단일 스레드 약 10,700 줄/초 (Locust 부하테스트는 통합 후) | ✅ (모델 구간) |
| ⑤ 두 모델 비교 후 채택 | Autoencoder 채택 (DeepLog 대비 F1 +5.4%p 시드 평균, 추론 약 5배 빠름) | ✅ |
| ⑥ 대시보드·Slack | 이동준 담당, 진행 예정 | — |

| 방법별 F1 | Precision-Recall |
|---|---|
| ![F1](docs/figures/f1_comparison.png) | ![PR](docs/figures/pr_curves.png) |
| **EOS 효과** | **Autoencoder 점수 분포** |
| ![EOS](docs/figures/eos_ablation.png) | ![dist](docs/figures/score_dist_ae.png) |

주요 발견
- **EOS(세션 종료) 토큰**: 이상 세션의 약 37%가 중간에 끊긴 세션이라, 다음 이벤트만 보는 기본 모델은 놓친다. EOS 추가로 test F1 DeepLog 0.821 → 0.926, Autoencoder 0.884 → 0.977
- **Drain 튜닝**: 기본 sim_th 0.5 는 이상 신호(`Interrupted`)를 정상 이벤트에 합쳐버림 → 0.85 로 수정
- **val ↔ test 일치**: Autoencoder val 0.972 / test 0.977 — val 기반 선택이 과적합되지 않음
- 과정 상세: `docs/plan_week4-6.md`, 실험 결과: `results/`

## 협업 규칙

- `main` 직접 push 금지 → `feat/<영역>-<내용>` 브랜치 + PR, 1명 이상 리뷰 후 merge
- 비밀값(Slack Webhook URL 등)은 `.env` 에만, **절대 커밋 금지**
- 설정값은 코드에 하드코딩하지 말고 `configs/config.yaml` 에 추가
- 실험 결과는 `results/<model>.json` 으로 저장해 베이스라인과 같은 형식으로 비교

## 참고

- Du et al., *DeepLog*, CCS 2017 · He et al., *Drain*, ICWS 2017 · He et al., *Loghub*, arXiv:2008.06448

## 전체 시스템 실행 (백엔드 + 재생 + 대시보드)

```bash
make api                         # 터미널 1: FastAPI 서버 (Autoencoder 판정, http://127.0.0.1:8000/docs)
make replay                      # 터미널 2: test 블록 로그 재생 → 끝나면 세션 종료 판정
make dashboard                   # 터미널 3: Streamlit → 사이드바에서 "SQLite" 선택
```

- 실시간 경보(로그마다 판정): test F1 0.880 / 세션 종료 판정 포함 최종: test F1 0.977
- DB 구조: `docs/db_schema.md` (규격), `docs/db_design.md` (설계 계획서), 백엔드 상세: `src/logguard/api/README.md`

## Streamlit 대시보드

```bash
# make setup으로 환경을 구성한 후
make dashboard
# 또는
.venv/bin/python -m streamlit run src/logguard/dashboard/app.py
```

기본은 **데모 데이터** 모드입니다. DB나 학습된 모델 없이 화면을 확인할 수 있습니다.
데모 로그·스코어·지연·Slack 이력은 합성 데이터이며, 모델 평가 탭은 실제 `results/final_test.csv`(6단계 test 결과)를 읽습니다.
대시보드는 알림을 전송하지 않으며 SQLite를 읽기 전용으로 조회합니다.

- 상단: 관측 세션 수, 이상 세션 수·비율, Slack 전송 성공 수
- 실시간 모니터링: 모델별 스코어·임계치(0.5), 지연 p95, 최근 추론 윈도우와 **판정 근거**, CSV 다운로드
- 세션 상세: Block ID별 윈도우 이력, 세션 종료 판정 여부(진행 중 / 정상 종료 / 끊긴 세션)
- 알림 이력: 전송 성공·실패·대기 기록
- 모델 평가: Rule-Based·통계 빈도·DeepLog·Autoencoder(+EOS 비교)의 test Precision/Recall/F1
- 사이드바: 데이터 소스, 조회 기간, Block ID 검색, 이상 세션 필터, 자동 갱신

### 백엔드 연동 규격 (2026-10-20 통합)

DB 규격은 백엔드와 공유하는 [`src/logguard/api/schema.sql`](src/logguard/api/schema.sql) 하나이며 설명은
[`docs/db_schema.md`](docs/db_schema.md)에 있습니다. 백엔드가 테이블을 만들고 기록하며 대시보드는 읽기만 합니다.
기본 DB 경로는 `configs/config.yaml`의 `dashboard.database_path` (`artifacts/api/logguard.sqlite`, 백엔드와 같은 파일)입니다.

- `logs`: 로그 1줄 × 블록 1개. 대시보드는 판정된 행(`status` normal/anomaly)만 읽습니다
- `session_results`: 세션 종료(EOS) 판정. 세션 판정 = **윈도우 OR + 종료 판정** (같은 모델·Block ID 기준)
- `alerts`: Slack 알림 이력. 전송·중복 방지는 Slack 알림 작업자(담당 이동준, 미구현)가 하며, 블록·모델당 1건만 저장됩니다

조회 기간은 **최신 기록 시각**을 기준으로 계산하여 과거 Replay도 조회할 수 있습니다.
최대 조회 행 수는 `dashboard.max_rows`이며 상한에 도달하면 경고합니다.

대시보드 검증: `.venv/bin/python -m pytest tests/test_dashboard.py -q`.
