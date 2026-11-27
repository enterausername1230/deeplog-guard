"""1. 시스템 구성도 — 로그 1줄이 들어와서 알림·대시보드까지 가는 구성 요소와 연결.

작성: 옥상민 / 2026-11-27 (코드 기준 main a42d0e4)
근거 코드: api/main.py(엔드포인트·처리 순서), api/inference.py(LogMatcher), api/predictor.py(어댑터),
          models/autoencoder.py(탐지기), api/schema.sql(테이블), dashboard/notifier.py(알림), dashboard/app.py(대시보드)
실행: .venv/bin/python docs/diagrams/1_system_architecture.py  → 1_system_architecture.png
"""
from pathlib import Path

from _draw import arrow, box, canvas, cylinder, group, note, save

OUT = Path(__file__).with_suffix(".png")
BLUE, ORANGE, PURPLE, CYAN, GRAY = "#2563EB", "#D97706", "#7C3AED", "#0891B2", "#94A3B8"

fig, ax = canvas(12, 9.2, "DeepLog-Guard 시스템 구성도",
                 "HDFS 로그 1줄 수신 → 실시간 이상 판정 → 저장 → Slack 알림 · 대시보드")

# ── ① 로그 발생원 (부하 시험은 서버 입구 바로 위에 두어 선이 다른 박스를 가로지르지 않게) ─────
group(ax, 0.2, 7.25, 5.6, 8.3, "① 로그 발생원", "source")
box(ax, 0.4, 7.4, 1.65, 7.95, "부하 시험", "Locust · 50명", dashed=True)
box(ax, 1.95, 7.4, 3.45, 7.95, "HDFS 로그", "test 블록 약 223만 줄", fill="#E2E8F0")
box(ax, 3.8, 7.4, 5.4, 7.95, "로그 재생 모듈", "replay.py · start.sh 기본 초당 1000줄")
arrow(ax, (3.45, 7.67), (3.8, 7.67))

# ── 학습 산출물 (오프라인 학습에서 만든 파일, 서버 시작 시 읽음) ─────────────
group(ax, 6.2, 7.25, 11.8, 8.3, "학습 산출물", "artifact", "오프라인 학습 결과 · 서버 시작 시 읽음", dashed=True)
note(ax, 6.4, 7.4, 8.0, 7.9, "drain3_state.bin", "로그 템플릿 52개")
note(ax, 8.2, 7.4, 9.8, 7.9, "event_vocab.json", "Event ID → 입력 번호")
note(ax, 10.0, 7.4, 11.6, 7.9, "autoencoder.pt", "가중치 · 임계치 6.78")

# ── ② 백엔드 서버 ───────────────────────────────────────────────
group(ax, 0.2, 4.95, 11.8, 6.8, "② 백엔드 서버", "backend", "FastAPI · uvicorn 단일 워커 (블록별 이벤트 순서 보장)")
y0, y1 = 5.8, 6.35
box(ax, 0.4, y0, 1.95, y1, "POST /logs", "로그 1줄 수신·형식 검사", fill="#DBEAFE")
box(ax, 2.65, y0, 4.7, y1, "LogMatcher", "Drain3 템플릿 매칭\n→ Event ID · Block ID", sub_size=8.2)
box(ax, 5.0, y0, 6.65, y1, "블록 이력 조회", "같은 블록 최근 이벤트 20개")
box(ax, 6.95, y0, 8.95, y1, "DetectorPredictor", "백엔드 ↔ 모델 연결 어댑터\n(시작 시 윈도우·어휘 검사)", sub_size=8.2)
box(ax, 9.25, y0, 11.6, y1, "Autoencoder 탐지기", "재구성 오차 > 임계치 → 이상\n처음 보는 로그 → 즉시 이상",
    fill="#BFDBFE", strong=True, sub_size=8.2)
for a, b in [(1.95, 2.65), (4.7, 5.0), (6.65, 6.95), (8.95, 9.25)]:
    arrow(ax, (a, 6.07), (b, 6.07), lw=1.5)
box(ax, 1.0, 5.08, 4.3, 5.55, "세션 종료 판정",
    "재생이 끝나면 POST /sessions/close · 오래 소식 없는 블록은 자동", sub_size=7.8)
box(ax, 9.55, 5.08, 11.6, 5.52, "조회 API", "GET /logs · /sessions · /stats · /health", dashed=True, sub_size=7.6)

# ── ③ 저장소 ───────────────────────────────────────────────────
group(ax, 2.0, 3.25, 10.0, 4.45, "③ 저장소", "db", "SQLite 파일 1개 · WAL 모드 (쓰는 중에도 동시에 읽기)")
cylinder(ax, 2.3, 3.37, 4.6, 4.13, "session_results", "세션 종료 판정 (블록당 1행)")
cylinder(ax, 4.85, 3.37, 7.15, 4.13, "logs", "로그·판정 결과 (줄 × 블록 = 1행)")
cylinder(ax, 7.4, 3.37, 9.7, 4.13, "alerts", "Slack 알림 이력·전송 결과")

# ── ④ Slack 알림 작업자 / ⑤ 대시보드 ──────────────────────────────
group(ax, 0.2, 1.3, 6.3, 2.6, "④ Slack 알림 작업자", "notify", "notifier.py · 0.5초마다 새 이상 확인")
box(ax, 0.4, 1.45, 3.05, 2.15, "이상 유형 분류", "event_guide.yaml\n템플릿 52개 → 유형 10개 · 권장 조치", sub_size=8.2)
box(ax, 3.35, 1.45, 6.1, 2.15, "알림 등급 정책", "긴급: 새 유형 첫 발생 · 이상 급증 → 즉시\n경고: 20초마다 모아서 요약 1건",
    fill="#EDE9FE", strong=True, kind="notify", sub_size=8.2)
arrow(ax, (3.05, 1.8), (3.35, 1.8), color=PURPLE)
group(ax, 6.9, 1.3, 11.8, 2.6, "⑤ 모니터링 대시보드", "dash", "Streamlit · 읽기 전용 · 5초 자동 갱신")
box(ax, 7.2, 1.45, 11.5, 2.15, "대시보드 화면", "실시간 점수 추이 · 세션 상세(판정 근거)\n알림 이력 · 모델 평가 결과",
    fill="#CFFAFE", kind="dash", strong=True, sub_size=8.4)

# ── 바깥 ──────────────────────────────────────────────────────
box(ax, 1.8, 0.2, 4.4, 0.85, "Slack 채널", "Incoming Webhook · 카드 메시지", fill="#EDE9FE", kind="notify",
    strong=True)
box(ax, 7.9, 0.25, 9.9, 0.8, "운영자", "", lw=1.8, title_size=11.5)

# ── 연결선 ────────────────────────────────────────────────────
# 로그 수신
arrow(ax, (4.1, 7.4), (1.5, 6.35), "HTTP POST · 로그 1줄 (보낸 시각 sent_at 포함)", BLUE, lw=1.9, lpos=0.42,
      loff=(0.15, 0.06))
arrow(ax, (1.02, 7.4), (1.02, 6.35), "실제 로그\n순환", dashed=True, lpos=0.42, loff=(0, 0))
# 학습 산출물 → 서버 부품 (시작 시 읽기)
arrow(ax, (7.2, 7.4), (4.2, 6.35), color=GRAY, dashed=True, head=False, lw=1.0)
arrow(ax, (9.0, 7.4), (4.5, 6.35), color=GRAY, dashed=True, head=False, lw=1.0)
arrow(ax, (10.8, 7.4), (10.6, 6.35), color=GRAY, dashed=True, head=False, lw=1.0)
# 세션 종료 판정 → 어댑터 (EOS)
arrow(ax, (4.3, 5.35), (7.7, 5.8), "EOS 판정 (마지막 이벤트 뒤 '세션 끝')", dashed=True, rad=0.12, lpos=0.3,
      loff=(0.25, -0.13), lsize=8)
# 저장소 읽기·쓰기
arrow(ax, (5.75, 4.13), (5.75, 5.8), "블록 이력 읽기", ORANGE, dashed=True, lpos=0.32)
arrow(ax, (9.7, 5.8), (6.75, 4.13), "판정 결과 저장 (점수 · 근거 · 지연)", ORANGE, lw=1.7, lpos=0.66, loff=(0.9, 0))
arrow(ax, (2.3, 5.08), (3.2, 4.13), "종료 판정 저장\n(재판정 시 갱신)", ORANGE, lpos=0.4, loff=(-0.6, 0))
# 알림 (이름표는 저장소와 알림 영역 사이 빈 줄 y≈2.9 에)
arrow(ax, (2.9, 3.37), (1.6, 2.15), "끊긴 세션", PURPLE, lpos=0.385, loff=(-0.4, 0))
arrow(ax, (5.4, 3.37), (2.5, 2.15), "새 이상 블록 조회", PURPLE, lpos=0.385, loff=(0.1, 0))
arrow(ax, (5.3, 2.15), (8.0, 3.37), "알림 기록 (블록당 1회)\n+ logs.alerted = 1", PURPLE, lpos=0.615, loff=(0.1, 0))
arrow(ax, (4.6, 1.45), (3.2, 0.85), "Webhook 전송\n(긴급 2초 이내)", PURPLE, lw=1.9, lpos=0.55, loff=(-0.75, -0.02))
# 대시보드 읽기
arrow(ax, (6.9, 3.37), (8.4, 2.15), "", CYAN, dashed=True)
arrow(ax, (9.0, 3.37), (9.6, 2.15), "세 테이블 읽기 전용 조회", CYAN, dashed=True, lpos=0.385, loff=(1.0, 0))
# 운영자
arrow(ax, (4.4, 0.52), (7.9, 0.52), "알림 확인", PURPLE, lpos=0.62)
arrow(ax, (9.35, 1.45), (9.0, 0.8), "화면 확인", CYAN, lpos=0.5, loff=(0.45, 0))
arrow(ax, (4.2, 0.85), (7.3, 1.45), "카드의 블록 링크 (?block=블록ID)", PURPLE, dashed=True, lpos=0.72, loff=(-0.1, -0.13),
      lsize=8)

save(fig, OUT)
