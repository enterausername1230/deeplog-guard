"""Run: uvicorn logguard.api.main:app --app-dir src --host 127.0.0.1"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
import logging
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .db import Database
from .inference import LogMatcher, Predictor
from .settings import load_settings

logger = logging.getLogger(__name__)


class LogInput(BaseModel):
    raw_log: str = Field(min_length=1, max_length=65536)


class LogRecord(BaseModel):
    id: int
    timestamp: str
    log_date: str
    log_time: str
    level: str
    component: str
    block_id: str | None
    event_id: int | None
    event_index: int
    anomaly_score: float | None
    is_anomaly: bool | None
    status: Literal['pending_model', 'no_block', 'normal', 'anomaly']
    raw_log: str
    alerted: bool


class IngestResult(BaseModel):
    records: list[LogRecord]


def create_app(settings=None, predictor: Predictor | None = None):
    @asynccontextmanager
    async def lifespan(app):
        cfg = settings if settings is not None else load_settings()
        matcher = LogMatcher(cfg)
        db = Database(cfg['database'], cfg['sqlite_timeout_seconds'])
        db.initialize()
        app.state.settings, app.state.matcher, app.state.db = cfg, matcher, db
        app.state.lock = Lock()
        yield

    app = FastAPI(title='DeepLog-Guard Backend', version='0.1.0', lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model_ready': predictor is not None,
                'mode': 'inference' if predictor is not None else 'collection_only'}

    @app.post('/logs', response_model=IngestResult, status_code=201)
    def ingest(body: LogInput):
        cfg = app.state.settings
        if len(body.raw_log) > cfg['max_log_chars']:
            raise HTTPException(413, 'Log exceeds configured max_log_chars')
        try:
            # One worker is required: parsing and per-block sequence updates are ordered.
            with app.state.lock:
                parsed, blocks = app.state.matcher.parse(body.raw_log)
                timestamp = datetime.now(timezone.utc).isoformat()
                records = []
                with app.state.db.connect() as con:
                    for block in blocks or [None]:
                        record = {**parsed, 'timestamp': timestamp, 'block_id': block,
                                  'raw_log': body.raw_log, 'anomaly_score': None,
                                  'is_anomaly': None, 'status': 'pending_model' if block else 'no_block'}
                        if block and predictor is not None:
                            events = app.state.db.history(con, block, cfg['window_size'])
                            events.append(parsed['event_index'])
                            try:
                                prediction = predictor.predict(tuple(events))
                            except Exception as exc:
                                logger.exception('Model inference failed')
                                raise HTTPException(503, 'Model inference failed; request was not stored') from exc
                            record.update(anomaly_score=prediction.anomaly_score,
                                          is_anomaly=prediction.is_anomaly,
                                          status='anomaly' if prediction.is_anomaly else 'normal')
                        records.append(app.state.db.insert(con, record))
                return {'records': records}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except sqlite3.OperationalError as exc:
            logger.exception('Database unavailable')
            raise HTTPException(503, 'Database temporarily unavailable') from exc

    @app.get('/logs', response_model=list[LogRecord])
    def logs(after_id: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
             block_id: str | None = None,
             status: Literal['pending_model', 'no_block', 'normal', 'anomaly'] | None = None):
        return app.state.db.list_logs(after_id, limit, block_id, status)

    @app.get('/stats')
    def stats():
        return app.state.db.stats()

    return app


app = create_app()
