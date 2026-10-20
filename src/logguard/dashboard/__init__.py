"""Streamlit 모니터링 대시보드 (담당: 이동준).

app.py: 세션 KPI, 스코어 추이, 추론 윈도우·세션 상세, 알림 이력, 모델 평가.
data.py: 읽기 전용 SQLite 조회와 합성 데모 데이터.
DB 규격: src/logguard/api/schema.sql (백엔드와 공유하는 단일 규격, 설명 docs/db_schema.md).
    logs(윈도우 판정) + session_results(세션 종료 판정) → 세션 판정은 OR.
알림 전송·중복 방지: Slack 알림 작업자 (담당 이동준, 2026-10-20 결정 — 미구현). 대시보드는 이력만 조회한다.
"""
