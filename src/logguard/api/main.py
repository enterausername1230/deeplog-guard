"""Run: uvicorn logguard.api.main:app --app-dir src --host 127.0.0.1

DeepLog-Guard 실시간 추론 서버 (원작성: 안명근 / 2026-10-20 모델 연결·DB 통합: 옥상민)

요청 흐름 (로그 1줄마다)
    POST /logs {"raw_log": "...", "sent_at": "..."}
      → LogMatcher: Drain 매칭으로 Event ID·블록 ID 구하기
      → 블록마다: DB 에서 최근 이력 꺼내기 → 판정기(predictor.predict) → logs 테이블에 저장
      → (주기적으로) 오래 소식 없는 블록은 세션 종료 판정(predict_close) → session_results 에 저장

엔드포인트
    GET  /health          서버·모델 상태
    POST /logs            로그 1줄 수신·판정·저장
    GET  /logs            저장된 행 조회 (after_id 페이지 넘기기, block_id·status 필터)
    POST /sessions/close  세션 종료 판정 실행 (idle_lines 생략 = 열린 세션 전부, 재생이 끝났을 때 호출)
    GET  /sessions        종료 판정 결과 조회
    GET  /stats           상태별 행 수, 종료된 세션 수

모델 선택: configs/api.yaml 의 model (autoencoder / deeplog / null=수집만). 테스트에서는 create_app(predictor=...) 로 주입.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
import json
import logging
import sqlite3
import time

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .db import Database
from .inference import LogMatcher, Predictor
from .predictor import DetectorPredictor
from .settings import load_settings

logger = logging.getLogger(__name__)


class LogInput(BaseModel):
    """POST /logs 요청 본문."""
    raw_log: str = Field(min_length=1, max_length=65536)    # HDFS 로그 한 줄
    # 로그 재생 모듈이 보낸 시각 (UTC ISO). E2E 지연(전송 → Slack 완료) 계산용. 직접 보낼 때는 생략 가능
    sent_at: str | None = Field(default=None, max_length=64)


class LogRecord(BaseModel):
    """logs 테이블 한 행 = API 응답 한 건 (칸 설명은 schema.sql)."""
    id: int
    timestamp: str
    sent_at: str | None = None
    log_date: str
    log_time: str
    level: str
    component: str
    block_id: str | None
    event_id: int | None
    event_index: int
    model_name: str | None = None
    anomaly_score: float | None
    threshold: float | None = None
    is_anomaly: bool | None
    reason: str | None = None
    details: str | None = None          # 판정 근거 JSON 문자열
    latency_ms: float | None = None
    status: Literal['pending_model', 'no_block', 'normal', 'anomaly']
    raw_log: str
    alerted: bool


class IngestResult(BaseModel):
    records: list[LogRecord]


def _details_json(prediction) -> str | None:
    """판정 근거 dict → DB 저장용 JSON 문자열. 근거가 없는 판정기(테스트용 등)면 None."""
    details = getattr(prediction, 'details', None)
    return None if details is None else json.dumps(details, ensure_ascii=False)


def create_app(settings=None, predictor: Predictor | None = None):
    """FastAPI 앱을 만든다.

    Args:
        settings : 설정 dict (None 이면 configs/api.yaml). 테스트는 임시 DB 경로를 넣은 설정을 준다
        predictor: 판정기를 직접 넣을 때 (테스트용). None 이면 설정의 model 로 DetectorPredictor 를 만든다
    """
    @asynccontextmanager
    async def lifespan(app):
        """서버 시작 시 1회: 설정·파서·DB·판정기 준비 (yield 이후는 서버 종료 시 실행)."""
        cfg = settings if settings is not None else load_settings()
        matcher = LogMatcher(cfg)
        db = Database(cfg['database'], cfg['sqlite_timeout_seconds'])
        db.initialize()
        app.state.settings, app.state.matcher, app.state.db = cfg, matcher, db
        # 판정기: 직접 넣은 것이 우선, 없으면 api.yaml 의 model 로 생성 (model 이 null 이면 None = 수집만)
        #   DetectorPredictor 는 window_size·어휘 검사를 통과하지 못하면 ValueError → 서버가 켜지지 않음 (의도된 동작)
        app.state.predictor = predictor if predictor is not None else DetectorPredictor.from_settings(cfg)
        app.state.lock = Lock()
        yield

    app = FastAPI(title='DeepLog-Guard Backend', version='0.2.0', lifespan=lifespan)

    def close_sessions(max_last_id=None, close_reason='manual'):
        """아직 종료 판정을 안 받은 블록들에 세션 종료(EOS) 판정을 내리고 session_results 에 저장한다.

        Args:
            max_last_id : 마지막 로그 id 가 이 값 이하인 블록만 (= 오래 소식 없음). None 이면 열린 블록 전부
            close_reason: 'idle'(자동) / 'manual'(POST /sessions/close)
        Returns:
            이번에 종료 판정한 블록 수
        """
        pred, cfg, db = app.state.predictor, app.state.settings, app.state.db
        # 종료 판정 기능이 없는 판정기(수집 모드, 테스트용 가짜 판정기)는 건너뛴다
        if pred is None or not hasattr(pred, 'predict_close'):
            return 0
        model_name = getattr(pred, 'model_name', 'model')
        now = datetime.now(timezone.utc).isoformat()
        with db.connect() as con:                       # 전부 한 트랜잭션: 중간 실패 시 이번 종료 판정 전체 취소
            todo = db.sessions_to_close(con, model_name, max_last_id)
            for block, last_id in todo:
                # 그 블록의 최근 window_size 개만 꺼내 EOS 판정 (score_close 는 최근 w 개면 정확하다)
                p = pred.predict_close(tuple(db.history(con, block, cfg['window_size'])))
                db.insert_session_result(con, {
                    'block_id': block, 'model_name': model_name, 'closed_at': now, 'close_reason': close_reason,
                    'last_log_id': last_id, 'anomaly_score': p.anomaly_score,
                    'threshold': p.threshold if p.threshold is not None else 0.5,
                    'is_anomaly': int(p.is_anomaly), 'reason': p.reason or ('model' if p.is_anomaly else 'normal'),
                    'details': _details_json(p)})
        return len(todo)

    @app.get('/health')
    def health():
        pred = app.state.predictor
        return {'status': 'ok', 'model_ready': pred is not None,
                'model': getattr(pred, 'model_name', None),
                'mode': 'inference' if pred is not None else 'collection_only'}

    @app.post('/logs', response_model=IngestResult, status_code=201)
    def ingest(body: LogInput):
        t0 = time.perf_counter()                        # 지연 측정 시작: 서버가 요청을 받은 시점
        cfg, pred = app.state.settings, app.state.predictor
        if len(body.raw_log) > cfg['max_log_chars']:
            raise HTTPException(413, 'Log exceeds configured max_log_chars')
        try:
            # One worker is required: parsing and per-block sequence updates are ordered.
            # (블록별 이력 순서가 섞이면 판정이 틀어지므로 한 번에 한 요청만 처리)
            with app.state.lock:
                parsed, blocks = app.state.matcher.parse(body.raw_log)
                timestamp = datetime.now(timezone.utc).isoformat()
                records = []
                with app.state.db.connect() as con:
                    for block in blocks or [None]:
                        record = {**parsed, 'timestamp': timestamp, 'sent_at': body.sent_at, 'block_id': block,
                                  'raw_log': body.raw_log, 'anomaly_score': None,
                                  'is_anomaly': None, 'status': 'pending_model' if block else 'no_block'}
                        if block and pred is not None:
                            events = app.state.db.history(con, block, cfg['window_size'])
                            events.append(parsed['event_index'])
                            try:
                                prediction = pred.predict(tuple(events))
                            except Exception as exc:
                                logger.exception('Model inference failed')
                                raise HTTPException(503, 'Model inference failed; request was not stored') from exc
                            record.update(anomaly_score=prediction.anomaly_score,
                                          is_anomaly=prediction.is_anomaly,
                                          status='anomaly' if prediction.is_anomaly else 'normal',
                                          # 판정 근거·모델 정보 (대시보드 표시용). 테스트용 가짜 판정기는 None 일 수 있음
                                          model_name=prediction.model_name or getattr(pred, 'model_name', None),
                                          threshold=prediction.threshold, reason=prediction.reason,
                                          details=_details_json(prediction),
                                          latency_ms=(time.perf_counter() - t0) * 1000)
                        records.append(app.state.db.insert(con, record))
                # ── 자동 세션 종료 판정 (idle) ──────────────────────────────────────────────
                #   session_sweep_every 행마다 한 번, "session_idle_lines 행 넘게 소식 없는 블록" 을 종료 판정한다.
                #   HDFS 정상 블록은 이벤트 사이가 수십만 줄 벌어지기도 해서(test 스트림 최대 약 130만 줄)
                #   기본값을 150만 줄로 크게 잡았다. 시연·평가에서는 재생이 끝난 뒤 POST /sessions/close 로 한꺼번에 닫는다.
                idle, every = cfg.get('session_idle_lines'), cfg.get('session_sweep_every')
                last_id = records[-1]['id']
                if idle and every and last_id % every < len(records):
                    try:
                        close_sessions(max_last_id=last_id - idle, close_reason='idle')
                    except Exception:                    # 종료 판정 실패가 로그 수신까지 막지 않도록 기록만 남김
                        logger.exception('Idle session close failed')
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

    @app.post('/sessions/close')
    def sessions_close(idle_lines: int | None = Query(None, ge=1)):
        """세션 종료 판정 실행. idle_lines 생략 = 열린 세션 전부 (로그 재생이 끝났을 때 호출 → 최종 F1 0.977 재현)."""
        if app.state.predictor is None or not hasattr(app.state.predictor, 'predict_close'):
            raise HTTPException(409, 'Session close needs a model with predict_close (configs/api.yaml model)')
        with app.state.lock:                            # 수신 처리와 섞이지 않게 같은 잠금 사용
            max_last_id = None
            if idle_lines is not None:
                with app.state.db.connect() as con:
                    max_last_id = app.state.db.max_id(con) - idle_lines
            closed = close_sessions(max_last_id, 'manual')
        return {'closed': closed}

    @app.get('/sessions')
    def sessions(limit: int = Query(100, ge=1, le=10000), anomalies_only: bool = False):
        return app.state.db.list_sessions(limit, anomalies_only)

    @app.get('/stats')
    def stats():
        return app.state.db.stats()

    return app


app = create_app()
