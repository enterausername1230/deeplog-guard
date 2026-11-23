# 최종 테스트 시나리오 (2026-11-28 ~ 29)

- 작성: 옥상민 / 2026-11-20
- 목적: 계획서 목표 ①~⑥ 실측표 확정 (M3, 12/3), 시연 전 기능·안정성 최종 확인
- 기준 코드: 테스트 당일 `main` 최신 커밋 (시작 전에 커밋 번호를 결과표에 적는다)
- 결과 기록: 이 문서 맨 아래 결과표를 채우고, 요약을 `docs/devlog.md` 에 남긴다

## 0. 공통 준비

```bash
git pull && git log --oneline -1      # 커밋 번호 기록
make setup                            # 처음 한 번
make data                             # 데이터가 없을 때만 (약 10분)
```

- 각 시험 전에 이전 실행이 남아 있지 않은지 확인: `pgrep -fl "uvicorn|streamlit|notifier"` 결과가 비어 있어야 한다
- `start.sh` 는 실행할 때마다 DB 를 새로 만든다 (`KEEP_DB=1` 이면 유지)
- **전체 재생(T3) 은 Slack 없이** — 2026-11-23 부터 `./start.sh full` 은 기본으로 Slack 을 끈다 (`SLACK_FULL=1` 일 때만 켬). 아래 명령의 `NO_SLACK=1` 은 확실히 하려고 붙여 둔 것

## 1. 일정·담당

| 날짜 | 시간 | 시험 | 담당 |
|---|---|---|---|
| 11/28 (토) | 오전 | T1 새 clone 설치, T2 자동 테스트 | 전원 (각자 PC) |
| 11/28 | 오후 | T3 전체 재생 정확도 (약 1시간 실행), T4 지연 | 옥상민 |
| 11/28 | 오후 | T7 예외 입력, T8 세션 재판정, T11 재시작 | 안명근 |
| 11/29 (일) | 오전 | T5 Slack E2E, T9 알림 기능, T10 대시보드 | 이동준 |
| 11/29 | 오전 | T6 부하 (Locust) | 안명근 |
| 11/29 | 오후 | 결과 정리, 실측표 확정, 시연 리허설 (`./start.sh demo`, `docs/demo_scenario.md`) | 전원 |

---

## 2. 시험 항목

### T1. 새 clone 설치 (계획서: 팀원 누구나 실행 가능)

| 절차 | 합격 기준 |
|---|---|
| 빈 폴더에서 `git clone https://github.com/enterausername1230/deeplog-guard` → `make setup` → `make data` → `NO_SLACK=1 LIMIT=20000 ./start.sh` | 오류 없이 대시보드가 뜨고, 재생 후 이상 64행 · 이상 블록 12개 (실측 기준값) |

### T2. 자동 테스트

| 절차 | 합격 기준 |
|---|---|
| `make test` | 전부 통과 (11/20 기준 56개). 건너뜀 0 (데이터가 있을 때) |

### T3. 전체 재생 정확도 — 목표 ① (가장 중요)

전체 test 로그 약 223만 줄을 서버로 흘리고, 재생이 끝나면 세션 종료 판정까지 한다 (약 1시간).

```bash
NO_SLACK=1 NO_BROWSER=1 ./start.sh full
# 터미널에 "재생 완료" 와 "Closed sessions: N" 이 나오면 다른 터미널에서 아래 계산
```

```bash
PYTHONPATH=src .venv/bin/python - <<'EOF'
import sqlite3, pandas as pd
from logguard.eval.metrics import prf
db = sqlite3.connect("artifacts/api/logguard.sqlite")
q = lambda sql: {b for (b,) in db.execute(sql)}
realtime = q("SELECT DISTINCT block_id FROM logs WHERE status = 'anomaly'")
closed   = q("SELECT block_id FROM session_results WHERE is_anomaly = 1")
seen     = q("SELECT DISTINCT block_id FROM logs WHERE block_id IS NOT NULL")
test = pd.read_parquet("data/processed/test.parquet", columns=["block_id", "label"])
y = test["label"].to_numpy()
print("test 블록", len(test), "/ DB 에 없는 블록", int((~test.block_id.isin(seen)).sum()))
print("실시간 판정만        ", prf(y, test.block_id.isin(realtime)))
print("최종 (실시간 OR 종료)", prf(y, test.block_id.isin(realtime | closed)))
EOF
```

| 항목 | 합격 기준 | 참고값 (오프라인) |
|---|---|---|
| 최종 F1 | **≥ 0.85** (목표 ①) | 0.977 (P 0.970, R 0.984) |
| 실시간만 F1 | 기록만 | 0.880 |
| DB 에 없는 test 블록 | 기록 (여러 블록이 섞인 줄은 재생에서 빠지므로 0 이 아닐 수 있다) | — |
| 서버·대시보드 로그 오류 | `artifacts/run_logs/*.log` 에 Traceback 없음 (1시간 안정성) | — |

오프라인 값과 다르면 차이 나는 블록을 뽑아 원인을 적는다 (재생 필터, 종료 판정 시점 등).

### T4. 서버 지연 — 목표 ③ 앞부분

T3 이 끝난 DB 로 계산한다.

```bash
.venv/bin/python -c "
import sqlite3, numpy as np
l = np.array([r[0] for r in sqlite3.connect('artifacts/api/logguard.sqlite').execute('SELECT latency_ms FROM logs WHERE latency_ms IS NOT NULL')])
print('rows', len(l), 'p50/p95/p99 ms', np.percentile(l, [50, 95, 99]).round(3))"
```

| 합격 기준 | 참고값 |
|---|---|
| p95 ≤ 1,000ms | 0.52ms (재생 모듈 1개, 순차 전송) |

### T5. Slack E2E 지연 — 목표 ③ 뒷부분

```bash
NO_BROWSER=1 ./start.sh demo        # Slack 켜짐 상태 (.env 필요). 채점 결과가 나온 뒤 25초 기다렸다가 Ctrl+C (마지막 20초 요약까지)
.venv/bin/python -c "
import sqlite3, numpy as np
c = sqlite3.connect('artifacts/api/logguard.sqlite')
e = np.array([r[0] for r in c.execute(\"SELECT (julianday(a.completed_at)-julianday(l.sent_at))*86400000 FROM alerts a JOIN logs l ON l.id=a.log_id WHERE a.status='sent' AND l.sent_at IS NOT NULL AND a.message LIKE '🔴%'\")])
print('n', len(e), 'p50/p95/max ms', np.percentile(e, [50, 95]).round(0), e.max().round(0), '2초 이내', f'{(e <= 2000).mean():.0%}')"
```

| 합격 기준 | 참고값 |
|---|---|
| 긴급 알림 p95 ≤ 2,000ms, 실패 0 | 11/19 실제 Slack (이전 정책, 전부 즉시): 12건 p95 1,641ms / 11/23 가짜 Slack 0.3초 (등급 정책): 긴급 5건 × 3회 최대 1,603ms |

- 2026-11-23 알림 등급 정책 이후 E2E 는 **즉시 보내는 긴급 알림**(`message` 가 `🔴` 로 시작)으로 잰다. 경고는 정책상 최대 20초 모아서 보내므로 계산에서 뺀다
- 여러 번(3회) 반복해 결과를 모두 적는다. Slack 응답이 느린 날은 값이 커질 수 있다

### T6. 부하 — 목표 ④

```bash
NO_SLACK=1 ./start.sh serve                # 터미널 1 (재생 없이 서버만)
.venv/bin/locust -f src/logguard/api/locustfile.py --host http://127.0.0.1:8000 \
  --headless -u 50 -r 10 -t 70s --csv artifacts/locust/final50      # 터미널 2
.venv/bin/locust -f src/logguard/api/locustfile.py --host http://127.0.0.1:8000 \
  --headless -u 100 -r 25 -t 45s --csv artifacts/locust/final100    # 최대 처리량 확인
```

| 항목 | 합격 기준 | 참고값 (11/20, 실제 test 로그) |
|---|---|---|
| 50명·70초 실패율 | < 1% (목표 ④) | 0% (33,872건) |
| 50명 처리량 (상승 구간 이후 60초) | ≥ 약 480 req/s (설정 상한 500) | 약 487 req/s |
| 100명 최대 처리량 | 기록 | 약 890 req/s, 실패 0 |
| 응답 p95 | 기록 | 50ms (50명) / 120ms (100명) |

`final50_stats_history.csv` 에서 8~68초 구간 평균을 계산해 적는다.

### T7. 예외 입력

서버만 켠 상태(`NO_SLACK=1 ./start.sh serve`)에서 http://127.0.0.1:8000/docs 또는 curl 로 보낸다.

| # | 입력 | 기대 결과 |
|---|---|---|
| 7-1 | `{"raw_log": "hello world"}` (형식 오류) | 422 |
| 7-2 | `{"raw_log": ""}` | 422 |
| 7-3 | 16,384자보다 긴 줄 | 413 |
| 7-4 | 블록 ID 없는 정상 형식 줄 | 201, `status: no_block` (판정 안 함) |
| 7-5 | 처음 보는 형식 + 새 블록 (`… totally new disk failure on blk_999000111 …`) | 201, `anomaly`, `reason: unknown_event`, 점수 1.0 |
| 7-6 | 정상 이벤트 + 새 블록 (`… Receiving block blk_777000222 …`) | 201, `normal`, 점수 0 근처 |
| 7-7 | `GET /logs?limit=0` | 422 |
| 7-8 | `GET /logs?status=anomaly&limit=3` | 3건, 모두 anomaly |

### T8. 세션 종료 재판정

| 절차 | 기대 결과 |
|---|---|
| `NO_SLACK=1 LIMIT=5000 ./start.sh` 재생 후 `curl -X POST 127.0.0.1:8000/sessions/close` | `{"closed": 444}` 근처 |
| 같은 요청 한 번 더 | `{"closed": 0}` (새 로그 없으면 다시 판정 안 함) |
| 판정된 블록의 로그 1줄을 `POST /logs` 로 다시 보낸 뒤 종료 요청 | `{"closed": 1}`, `session_results` 행 수는 그대로, 그 블록 `last_log_id` 만 커짐 |

### T9. 알림 기능

| # | 절차 | 기대 결과 |
|---|---|---|
| 9-1 | T5 실행 후 '알림 이력' 탭과 `alerts` 테이블 | 이상 블록 수 = alerts 행 수 (블록당 1회), 모두 sent. 긴급(🔴) 행 수 = 나온 이상 유형 수 (시연 샘플 6개), 나머지는 경고(🟠) |
| 9-2 | 메시지 형식 확인 | 긴급 카드: 유형 이름·설명, 블록, 관련 노드(추정), 점수, KST 시각, 흐름, 권장 조치, 원본 로그, 대시보드 링크 / 경고 요약: 유형별 개수, 예시 블록 (Slack 메시지 약 8건) |
| 9-3 | 링크 클릭 | 그 블록만 걸러진 대시보드 (SQLite 모드) |
| 9-4 | 실패 처리: `NO_SLACK=1 LIMIT=20000 ./start.sh` 로 미알림 이상이 있는 DB 를 만든 뒤 (Ctrl+C), 형식은 맞지만 없는 주소로 한 번 실행 `SLACK_WEBHOOK_URL=https://hooks.slack.com/services/TEST/TEST/not-a-real-token PYTHONPATH=src .venv/bin/python -m logguard.dashboard.notifier --once` | 알림이 모두 `failed` 로 기록, 자동 재전송 없음. 실제 채널에는 아무것도 안 감 |
| 9-5 | 실패 건 재전송: `.env` 주소로 `PYTHONPATH=src .venv/bin/python -m logguard.dashboard.notifier --retry-failed --once` | failed 건이 sent 로 바뀜 |
| 9-6 | 20초 요약: T5 실행 중 Slack 메시지 시각 | 같은 유형 반복은 즉시 오지 않고 약 20초 간격 요약으로 옴 |
| 9-7 | 재시작: 알림 작업자만 껐다 켬 (`make slack`) 후 같은 유형 이상 추가 | 이미 알린 유형은 다시 '새 유형 첫 발생' 으로 오지 않음 |

### T10. 대시보드

| # | 확인 | 기대 결과 |
|---|---|---|
| 10-1 | `start.sh` 로 열었을 때 데이터 소스 | 처음부터 SQLite 선택, 지표 4개(관측 세션, 이상 세션, 비율, Slack 성공) 표시, 2만 줄 재생에서 '조회 상한' 경고 없음 |
| 10-2 | 자동 갱신 | 재생 중 5초마다 숫자 증가 |
| 10-3 | 점수 추이 그래프 | 짧은 재생에서도 선이 그려짐 (`1s 단위 최댓값`) |
| 10-4 | 판정 근거 칸 | `실제: … → 모델 예상: …` 또는 `학습에 없던 이벤트 등장` |
| 10-5 | 세션 상세 | 세션 선택 시 이력 표시, 종료 판정 여부 문구 |
| 10-6 | 모델 평가 탭 | 표 + 막대그래프 (지표 3개 나란히, 축 0~1) |
| 10-7 | Block ID 검색, 이상 세션만 보기, CSV 다운로드 | 필터 동작, CSV 열림 |
| 10-8 | 실행 로그 | `artifacts/run_logs/dashboard.log` 에 Traceback 없음 |

### T11. 재시작·내구성

| 절차 | 기대 결과 |
|---|---|
| 재생 중간에 `Ctrl+C` → `KEEP_DB=1 NO_SLACK=1 ./start.sh serve` | 이전 기록 유지, 서버 정상 기동 |
| 알림 작업자 재시작 (`make slack` 종료 후 다시 실행) | 이미 보낸 블록은 다시 보내지 않음 |

---

## 3. 결과표 (테스트 당일 작성)

- 기준 커밋: `________` / 실행 PC: `________` / 날짜: `________`

| 시험 | 결과 | 합격 | 비고 |
|---|---|---|---|
| T1 새 clone 설치 | | | |
| T2 자동 테스트 | __ passed | | |
| T3 최종 F1 (실시간만 F1) | ____ (____) | | |
| T4 서버 지연 p95 | ____ ms | | |
| T5 Slack E2E p95 (3회) | ____ / ____ / ____ ms | | |
| T6 부하 50명 실패율·처리량 / 100명 최대 | ___% · ___ req/s / ___ req/s | | |
| T7 예외 입력 (8개) | __/8 | | |
| T8 세션 재판정 | | | |
| T9 알림 기능 (7개) | __/7 | | |
| T10 대시보드 (8개) | __/8 | | |
| T11 재시작 | | | |

### 계획서 목표 실측표 (M3 제출용)

| 목표 | 기준 | 최종 측정값 | 달성 |
|---|---|---|---|
| ① F1 | ≥ 0.85 | (T3) | |
| ② Rule-Based 대비 | +10%p 이상 | +40.4%p (오프라인) | ✅ |
| ③ 추론 p95 / E2E p95 | ≤ 1초 / ≤ 2초 | (T4) / (T5) | |
| ④ 처리량·오류율 | 500 logs/s × 1분, < 1% | (T6) | |
| ⑤ 두 모델 비교 | 2종 구현·채택 | Autoencoder 채택 | ✅ |
| ⑥ 대시보드·Slack | 구현 | (T9, T10) | |
