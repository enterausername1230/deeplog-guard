-- Backend integration contract. Dashboard never runs this schema or writes to DB.
-- One row per scored window, linked to a Block ID session.
-- timestamp: UTC ISO-8601 replay reception time, e.g. 2026-10-12T08:00:00+00:00.
-- is_anomaly: model decision; any true window makes the session anomalous.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    block_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    threshold REAL NOT NULL,
    is_anomaly INTEGER NOT NULL CHECK (is_anomaly IN (0, 1)),
    raw_log TEXT NOT NULL,
    latency_ms REAL
);
CREATE INDEX IF NOT EXISTS predictions_time ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS predictions_session ON predictions(model_name, block_id);
-- Backend/notification worker owns sending and records every attempt here.
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    block_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('sent', 'failed', 'pending')),
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_time ON alerts(timestamp);
