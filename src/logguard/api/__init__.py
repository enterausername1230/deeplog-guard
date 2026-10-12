"""실시간 추론 API (담당: 안명근).

예정 모듈
  main.py   : FastAPI + Uvicorn. 로그 수신 → Drain 매칭 → 윈도우 → 모델 추론 → 판정 → SQLite 저장
  replay.py : Log Replay. data/raw/HDFS_v1/HDFS.log 를 원본 순서·일정 속도로 전송 (test 블록만 재생 권장)
  db.py     : SQLite 스키마 (timestamp, block_id, event_id, anomaly_score, raw_log, alerted)
파싱은 logguard.data.online.OnlineParser, 판정은 logguard.models.inference.load_detector() 사용.
사용법과 반환 형식은 docs/interface_backend.md 참고.
"""
