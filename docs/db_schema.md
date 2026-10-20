# DB 구조 (통합 규격)

- 작성: 옥상민 / 2026-10-20 (백엔드 `logs` + 대시보드 `predictions`·`alerts` 를 하나로 통합)
- 정의 파일: [`src/logguard/api/schema.sql`](../src/logguard/api/schema.sql) — **이 파일 하나만** 진짜 규격이다. 백엔드가 서버 시작 시 실행하고, 대시보드 테스트도 이 파일로 테이블을 만든다.
- DB 파일: `artifacts/api/logguard.sqlite` (`configs/api.yaml` 의 `database`, git 제외). 예전 구조로 만든 파일은 칸이 달라 **지우고 서버를 다시 켜야 한다.**

## 누가 쓰고 누가 읽나

| 테이블 | 한 행 | 쓰는 쪽 | 읽는 쪽 |
|---|---|---|---|
| `logs` | 로그 1줄 × 블록 1개 (실시간 판정) | 백엔드 `POST /logs` | 대시보드 (실시간 모니터링·세션 상세), 알림 작업자 |
| `session_results` | 끝난 세션 1개 (세션 종료·EOS 판정) | 백엔드 (자동 idle / `POST /sessions/close`) | 대시보드, 알림 작업자 |
| `alerts` | Slack 알림 1건 | **Slack 알림 작업자 (담당 이동준, 미구현)** | 대시보드 (알림 이력) |

## 세션 최종 판정

```
세션 이상 = (logs 에 is_anomaly = 1 인 행이 하나라도 있음)  OR  (session_results.is_anomaly = 1)
```

- `logs` 만으로 판정(실시간 경보): test F1 **0.880** (Precision 0.967, Recall 0.807)
- `session_results` 까지 포함(세션 종료 후 최종): test F1 **0.977** — 이상 세션의 약 18%(중간에 끊긴 세션)는 종료 판정으로만 잡힌다
- 세션 종료 시점
  - 자동(idle): 그 블록 마지막 행 뒤로 `session_idle_lines`(기본 150만) 행이 더 들어오면. HDFS 정상 블록은 이벤트 사이가 매우 길어서(test 재생 스트림 최대 약 130만 줄) 짧게 잡으면 정상 세션을 오탐한다
  - 수동: 로그 재생이 끝나면 `replay --close-sessions` (= `POST /sessions/close`)

## `logs` 칸

| 칸 | 타입 | 설명 |
|---|---|---|
| `id` | INTEGER PK | 저장 순서 (= 서버 수신 순서). idle 판정의 "줄 시계" |
| `timestamp` | TEXT | 서버 수신 시각 (UTC ISO-8601) |
| `sent_at` | TEXT? | 재생 모듈 전송 시각 (UTC ISO). E2E 지연 계산용 |
| `log_date`, `log_time` | TEXT | 원본 로그 날짜·시각 (`081109`, `203518`) |
| `level`, `component` | TEXT | `INFO`, `dfs.DataNode$DataXceiver` … |
| `block_id` | TEXT? | 블록이 없는 줄이면 NULL (`status = no_block`) |
| `event_id` | INTEGER? | Drain 템플릿 번호. **처음 보는 형식이면 NULL** (→ 즉시 이상) |
| `event_index` | INTEGER | 모델 입력 번호 (모르는 이벤트 = 1 = UNK) |
| `model_name` | TEXT? | `autoencoder` 등. 모델 미연결이면 NULL |
| `anomaly_score` | REAL? | 0~1 정규화 점수 |
| `threshold` | REAL? | 항상 0.5 (대시보드 기준선) |
| `is_anomaly` | INTEGER? | 1 이상 / 0 정상 / NULL 판정 안 함 |
| `reason` | TEXT? | `unknown_event` / `model` / `normal` |
| `details` | TEXT? | 판정 근거 JSON (Autoencoder: `worst.actual` vs `worst.reconstructed`) — 형식은 `docs/interface_backend.md` |
| `latency_ms` | REAL? | 서버 수신 → 판정 완료 (ms) |
| `status` | TEXT | `pending_model` / `no_block` / `normal` / `anomaly` |
| `raw_log` | TEXT | 원본 한 줄 |
| `alerted` | INTEGER | Slack 알림을 보냈으면 1 (알림 작업자가 갱신) |

## `session_results` 칸

`block_id, model_name` (복합 PK, 블록당 한 번만 판정), `closed_at`, `close_reason`(`idle`/`manual`), `last_log_id`, `anomaly_score`, `threshold`, `is_anomaly`, `reason`, `details`

## `alerts` 칸 — Slack 알림 작업자 구현 가이드 (이동준)

`id, timestamp, block_id, model_name, log_id, status(pending/sent/failed), message, completed_at, error`, **UNIQUE(block_id, model_name)**

권장 흐름 (별도 프로세스 또는 스레드, 예: `src/logguard/dashboard/notifier.py`)

1. 몇 초마다 새 이상을 찾는다
   - 실시간: `SELECT … FROM logs WHERE is_anomaly = 1 AND alerted = 0 AND id > 마지막으로 본 id`
   - 세션 종료: `SELECT … FROM session_results WHERE is_anomaly = 1` 중 아직 alerts 에 없는 블록
2. `INSERT OR IGNORE INTO alerts (…, status) VALUES (…, 'pending')` — UNIQUE 덕분에 **같은 블록은 한 번만** 알림
3. Slack Incoming Webhook 으로 전송 (URL 은 `.env` 의 `SLACK_WEBHOOK_URL`, 커밋 금지)
4. 결과로 `UPDATE alerts SET status = 'sent' | 'failed', completed_at = 지금, error = …` 그리고 `UPDATE logs SET alerted = 1 WHERE id = log_id`

메시지 예: `[이상] blk_-160… · autoencoder 점수 0.83 · 실제: Deleting block … → 모델 예상: Receiving block …` (`dashboard.data.explain()` 재사용 가능)

## 계획서 목표 ③ E2E 지연 계산

```sql
SELECT (julianday(a.completed_at) - julianday(l.sent_at)) * 86400000 AS e2e_ms
FROM alerts a JOIN logs l ON l.id = a.log_id
WHERE a.status = 'sent' AND l.sent_at IS NOT NULL;
```

목표: p95 ≤ 2,000ms. 서버 안쪽 지연(`logs.latency_ms`)은 2026-10-20 측정 기준 Autoencoder 연결 시 요청당 p95 약 1.8ms.
