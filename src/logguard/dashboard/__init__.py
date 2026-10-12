"""모니터링 대시보드·알림 (담당: 이동준).

예정 모듈
  app.py   : Streamlit. KPI 카드 / 이상 스코어 추이 / Slack 알림 이력 / 실시간 로그 스트림 테이블
  slack.py : Slack Incoming Webhook (URL 은 .env 의 SLACK_WEBHOOK_URL, 절대 커밋 금지)
SQLite 이력을 읽기 전용으로 조회한다. 템플릿 텍스트는 data/interim/hdfs_templates.csv 참조.
"""
