-- ============================================================================================
-- DeepLog-Guard 통합 DB 구조 (SQLite) — 백엔드(쓰기)와 대시보드(읽기)가 함께 쓰는 "단 하나의" 규격
--
--   쓰는 쪽 : 백엔드 FastAPI (src/logguard/api/db.py 가 서버 시작 시 이 파일을 실행해 테이블을 만든다)
--             Slack 알림 작업자 (alerts 테이블, 담당: 이동준)
--   읽는 쪽 : Streamlit 대시보드 (src/logguard/dashboard/data.py, 읽기 전용)
--   설명 문서: docs/db_schema.md
--
--   2026-10-20 통합: 백엔드의 logs 테이블을 기준으로, 대시보드 schema(predictions/alerts)에 있던 칸을 합쳤다.
--   기존 개발용 DB 파일(artifacts/api/logguard.sqlite)은 칸이 다르므로 지우고 새로 만들어야 한다.
--
--   "CREATE TABLE IF NOT EXISTS" : 테이블이 이미 있으면 아무것도 하지 않는다 (서버를 다시 켜도 데이터 유지)
-- ============================================================================================


-- --------------------------------------------------------------------------------------------
-- logs : 로그 1줄 × 블록 1개 = 1행  (한 줄에 블록이 여러 개면 블록마다 1행)
--   실시간 판정(스트리밍) 결과가 여기에 저장된다. 대시보드의 점수 추이·로그 표가 이 테이블을 읽는다.
-- --------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- 저장 순서 번호 (= 서버가 받은 순서). 세션 종료 판정의 "줄 시계"로도 쓴다
    timestamp     TEXT    NOT NULL,                   -- 서버가 받은 시각 (UTC, ISO-8601 예: 2026-10-20T08:00:00+00:00)
    sent_at       TEXT,                               -- 로그 재생 모듈이 보낸 시각 (UTC ISO). E2E 지연 계산용, 없으면 NULL
    log_date      TEXT    NOT NULL,                   -- 원본 로그 날짜 "081109" (YYMMDD)
    log_time      TEXT    NOT NULL,                   -- 원본 로그 시각 "203518" (HHMMSS)
    level         TEXT    NOT NULL,                   -- INFO / WARN …
    component     TEXT    NOT NULL,                   -- dfs.DataNode$DataXceiver …
    block_id      TEXT,                               -- 블록 ID. 블록이 없는 줄이면 NULL (status = no_block)
    event_id      INTEGER,                            -- Drain 템플릿 번호. 처음 보는 형식의 로그면 NULL (→ 즉시 이상 판정)
    event_index   INTEGER NOT NULL,                   -- 모델 입력 번호 (어휘 event_vocab.json 기준, 모르는 이벤트 = 1 = UNK)
    model_name    TEXT,                               -- 판정한 모델 이름 (autoencoder 등). 모델 미연결이면 NULL
    anomaly_score REAL,                               -- 0~1 정규화 이상 점수 (모델 미연결·블록 없음이면 NULL)
    threshold     REAL,                               -- 판정 기준선. 정규화 점수라 항상 0.5 (대시보드 기준선 표시용)
    is_anomaly    INTEGER,                            -- 1 = 이상, 0 = 정상, NULL = 판정 안 함
    reason        TEXT,                               -- unknown_event(학습에 없던 이벤트) / model(모델 판정) / normal
    details       TEXT,                               -- 판정 근거 JSON 문자열 (예: 복원 실패 이벤트 vs 모델이 복원한 이벤트)
    latency_ms    REAL,                               -- 서버 수신 → 판정 완료까지 걸린 시간 (밀리초)
    status        TEXT    NOT NULL                    -- pending_model / no_block / normal / anomaly
                  CHECK (status IN ('pending_model', 'no_block', 'normal', 'anomaly')),
    raw_log       TEXT    NOT NULL,                   -- 원본 로그 한 줄 그대로
    alerted       INTEGER NOT NULL DEFAULT 0          -- Slack 알림을 보냈으면 1 (알림 작업자가 갱신)
);
-- 블록별 최근 이력 조회(모델 입력 만들기)를 빠르게
CREATE INDEX IF NOT EXISTS logs_block_id_id ON logs(block_id, id);
-- 상태별 조회(이상만 보기, 알림 대상 찾기)를 빠르게
CREATE INDEX IF NOT EXISTS logs_status_id   ON logs(status, id);


-- --------------------------------------------------------------------------------------------
-- session_results : 끝난 세션(블록) 1개 = 1행
--   세션이 끝났다고 판단될 때(오래 소식 없음 또는 재생 종료) "이렇게 끝나도 정상인가(EOS 판정)" 결과를 저장한다.
--   이상 세션의 약 18% 는 이 판정으로만 잡힌다 (중간에 끊긴 세션). 실시간 경보 F1 0.88 → 세션 종료 판정 포함 0.977
--   최종 세션 판정 = (logs 에 이상 행이 하나라도 있음) OR (session_results.is_anomaly = 1)
-- --------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_results (
    block_id      TEXT    NOT NULL,
    model_name    TEXT    NOT NULL,
    closed_at     TEXT    NOT NULL,                   -- 종료 판정 시각 (UTC ISO)
    close_reason  TEXT    NOT NULL                    -- idle(오래 소식 없음) / manual(재생 종료 등 수동 종료)
                  CHECK (close_reason IN ('idle', 'manual')),
    last_log_id   INTEGER NOT NULL,                   -- 그 블록의 마지막 logs.id
    anomaly_score REAL    NOT NULL,                   -- EOS 판정 점수 (0~1, 0.5 초과면 이상)
    threshold     REAL    NOT NULL,
    is_anomaly    INTEGER NOT NULL CHECK (is_anomaly IN (0, 1)),
    reason        TEXT    NOT NULL,
    details       TEXT,                               -- 판정 근거 JSON
    PRIMARY KEY (block_id, model_name)                -- 한 블록은 모델마다 한 번만 종료 판정
);


-- --------------------------------------------------------------------------------------------
-- alerts : Slack 알림 1건 = 1행  (쓰는 쪽: Slack 알림 작업자, 담당 이동준)
--   UNIQUE(block_id, model_name) → 같은 블록은 모델마다 알림이 한 번만 저장된다 (중복 알림을 DB 가 막아 줌)
--   E2E 지연 = completed_at(Slack 요청 완료) − logs.sent_at(재생 모듈 전송)  ← 계획서 목표 ③ (2초 이내)
-- --------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,                   -- 알림을 만든 시각 (UTC ISO)
    block_id      TEXT    NOT NULL,
    model_name    TEXT    NOT NULL,
    log_id        INTEGER,                            -- 알림 원인이 된 logs.id (세션 종료 판정이 원인이면 NULL)
    status        TEXT    NOT NULL CHECK (status IN ('sent', 'failed', 'pending')),
    message       TEXT    NOT NULL,                   -- Slack 으로 보낸(보낼) 메시지 본문
    completed_at  TEXT,                               -- Slack 요청이 끝난 시각 (성공·실패 모두). pending 이면 NULL
    error         TEXT,                               -- 실패 이유 (status = failed 일 때)
    UNIQUE (block_id, model_name)
);
CREATE INDEX IF NOT EXISTS alerts_time ON alerts(timestamp);
