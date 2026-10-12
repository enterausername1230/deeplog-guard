"""실시간 추론 API (담당: 안명근).

예정 모듈
  main.py   : FastAPI + Uvicorn. 로그 수신 → Drain 매칭 → 윈도우 → 모델 추론 → 판정 → SQLite 저장
  replay.py : Log Replay. data/raw/HDFS_v1/HDFS.log 를 원본 순서·일정 속도로 전송 (test 블록만 재생 권장)
  db.py     : SQLite 스키마 (timestamp, block_id, event_id, anomaly_score, raw_log, alerted)
온라인 파싱은 artifacts/drain3/drain3_state.bin 을 로드해 TemplateMiner.match() 로 Event ID 를 구한다
(학습 때와 같은 템플릿을 써야 하므로 add_log_message 로 새로 학습시키지 않는다).
"""
