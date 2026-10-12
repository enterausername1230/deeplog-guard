"""모니터링 대시보드·알림 (담당: 이동준).

이 폴더는 무엇인가?
    (아직 코드 없음 — 3~4단계에서 만들 예정)
    이상 탐지 결과를 사람이 눈으로 확인하는 웹 화면(Streamlit)과, 이상 발생 시 Slack 으로 알림을 보내는
    코드가 들어갈 자리다.

예정 모듈
  app.py   : Streamlit. KPI 카드 / 이상 스코어 추이 / Slack 알림 이력 / 실시간 로그 스트림 테이블
  slack.py : Slack Incoming Webhook (URL 은 .env 의 SLACK_WEBHOOK_URL, 절대 커밋 금지)
             (Webhook URL 이 유출되면 누구나 우리 Slack 채널에 메시지를 보낼 수 있으므로 비밀로 관리)
SQLite 이력을 읽기 전용으로 조회한다. 템플릿 텍스트는 data/interim/hdfs_templates.csv 참조.

화면에 보여 줄 값
  - 이상 점수: Detector 가 0~1 로 정규화해서 주며, 0.5 가 모든 모델 공통 임계치(기준선)
  - 판정 근거: DetectionResult.details (어느 이벤트가 이상했는지, 모델이 대신 예상한 이벤트는 무엇인지)
"""
