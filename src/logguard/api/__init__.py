"""실시간 추론 API (담당: 안명근).

이 폴더는 무엇인가?
    (아직 코드 없음 — 3단계 API 구현에서 만들 예정)
    서버 로그가 실시간으로 들어오면 즉시 이상 여부를 판정하고 결과를 저장하는 웹 서버(FastAPI)가 들어갈 자리다.

예정 모듈
  main.py   : FastAPI + Uvicorn. 로그 수신 → Drain 매칭 → 윈도우 → 모델 추론 → 판정 → SQLite 저장
  replay.py : Log Replay. data/raw/HDFS_v1/HDFS.log 를 원본 순서·일정 속도로 전송 (test 블록만 재생 권장)
              (공개 데이터셋은 이미 다 쌓인 파일이므로, 한 줄씩 시간 간격을 두고 보내 "실시간" 처럼 흉내낸다)
  db.py     : SQLite 스키마 (timestamp, block_id, event_id, anomaly_score, raw_log, alerted)

이미 준비된 부품 (옥상민 제공, 그대로 가져다 쓰면 됨)
  파싱 : logguard.data.online.OnlineParser       — 로그 1줄 → Event ID, Block ID
  판정 : logguard.models.inference.load_detector() — Event ID 목록 → 이상 점수(0~1)·이상 여부·판정 근거
사용법과 반환 형식은 docs/interface_backend.md 참고.
"""
