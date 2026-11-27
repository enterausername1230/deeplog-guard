"""3. 유스케이스 다이어그램 — 누가(액터) 시스템으로 무엇을 하는가(유스케이스).

작성: 옥상민 / 2026-11-27 (코드 기준 main a42d0e4)
근거 코드
    로그 수집기 : api/replay.py, api/locustfile.py → POST /logs, POST /sessions/close (api/main.py)
    Slack       : dashboard/notifier.py (긴급 즉시 · 경고 20초 요약 · 실패 재전송 --retry-failed)
    운영자       : dashboard/app.py (탭 4개, Block ID 검색, CSV, ?block= 링크), start.sh, GET /health·/stats
    개발자       : data/parse.py·split.py, models/autoencoder.py, eval/final_eval.py (make 명령)
UML 표기
    실선            : 액터와 유스케이스의 연결
    점선 «include»  : 기본 유스케이스가 항상 포함하는 동작 (화살표 = 포함되는 쪽)
    점선 «extend»   : 조건이 맞을 때만 덧붙는 동작 (화살표 = 기본 유스케이스 쪽)
    빈 삼각형 실선   : 일반화 (자식 → 부모, "~의 한 종류")
실행: .venv/bin/python docs/diagrams/3_usecase.py  → 3_usecase.png
"""
import math
from pathlib import Path

from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

from _draw import COLORS, INK, SUB, canvas, save

OUT = Path(__file__).with_suffix(".png")
fig, ax = canvas(13, 9.6, "DeepLog-Guard 유스케이스 다이어그램", "액터 4종 · 유스케이스 24개 (UML)")

UC = {}            # 이름 → (cx, cy, 반지름a, 반지름b)


def usecase(key, cx, cy, text, w=2.2, h=0.56, kind="backend", strong=False):
    fc, ec = COLORS[kind]
    ax.add_patch(Ellipse((cx, cy), w, h, fc="white" if not strong else fc, ec=ec, lw=1.8 if strong else 1.2, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.6, color=INK, zorder=4, linespacing=1.15,
            weight="bold" if strong else "normal")
    UC[key] = (cx, cy, w / 2, h / 2)


def edge_point(key, tx, ty):
    """유스케이스 타원 테두리 위에서 (tx, ty) 쪽을 향한 점."""
    cx, cy, a, b = UC[key]
    dx, dy = tx - cx, ty - cy
    t = 1 / math.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    return cx + dx * t, cy + dy * t


def actor(x, y, name, sub=""):
    """사람 모양 액터. 연결점(손 높이) 좌표 반환."""
    kw = dict(color=INK, lw=1.6, zorder=5, solid_capstyle="round")
    ax.add_patch(Circle((x, y + 0.42), 0.13, fc="white", ec=INK, lw=1.6, zorder=5))
    ax.plot([x, x], [y + 0.29, y - 0.1], **kw)
    ax.plot([x - 0.24, x + 0.24], [y + 0.16, y + 0.16], **kw)
    ax.plot([x, x - 0.2], [y - 0.1, y - 0.42], **kw)
    ax.plot([x, x + 0.2], [y - 0.1, y - 0.42], **kw)
    ax.text(x, y - 0.6, name, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
    if sub:
        ax.text(x, y - 0.86, sub, ha="center", va="center", fontsize=8, color=SUB, linespacing=1.2)
    return x, y + 0.05


def system_actor(x0, y0, x1, y1, name, sub):
    """외부 시스템 액터 (상자 + «system»)."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#EDE9FE", ec=COLORS["notify"][1], lw=1.6, zorder=5))
    ax.text((x0 + x1) / 2, y1 - 0.18, "«외부 시스템»", ha="center", va="center", fontsize=8, color=SUB, zorder=6)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 0.02, name, ha="center", va="center", fontsize=10.5, weight="bold",
            color=INK, zorder=6)
    ax.text((x0 + x1) / 2, y0 + 0.17, sub, ha="center", va="center", fontsize=7.8, color=SUB, zorder=6)


def assoc(p, key, color=INK):
    """액터 ↔ 유스케이스 실선."""
    q = edge_point(key, *p)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.1, zorder=2)


def rel(src, dst, label, color="#475569", lpos=0.5, loff=(0, 0)):
    """«include»/«extend» 점선 화살표 (src → dst)."""
    p = edge_point(src, *UC[dst][:2])
    q = edge_point(dst, *UC[src][:2])
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, ls=(0, (4, 3)), mutation_scale=11,
                                shrinkA=0, shrinkB=0))
    x, y = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
    ax.text(x, y, label, ha="center", va="center", fontsize=7.6, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))


def generalize(child, parent, color="#475569"):
    """일반화: 자식 → 부모, 빈 삼각형 머리."""
    p = edge_point(child, *UC[parent][:2])
    q = edge_point(parent, *UC[child][:2])
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle="-|>", fc="white", ec=color, lw=1.1, mutation_scale=16,
                                shrinkA=0, shrinkB=0))


def band(x0, y0, x1, y1, title, kind):
    fc, ec = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fc, ec="none", alpha=0.7, zorder=0.5))
    ax.text(x0 + 0.12, y1 - 0.17, title, ha="left", va="center", fontsize=9.5, weight="bold", color=ec, zorder=1)


# ── 시스템 경계 ────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((2.3, 0.4), 8.55, 8.35, boxstyle="round,pad=0,rounding_size=0.12",
                            fc="white", ec=INK, lw=1.8, zorder=0.2))
ax.text(6.575, 8.55, "DeepLog-Guard 시스템", ha="center", va="center", fontsize=12, weight="bold", color=INK)
band(2.5, 4.85, 6.45, 8.3, "로그 수집 · 실시간 판정 (백엔드)", "backend")
band(6.65, 4.85, 10.65, 8.3, "이상 알림 (Slack 알림 작업자)", "notify")
band(2.5, 0.55, 6.45, 4.65, "오프라인 학습 · 평가", "model")
band(6.65, 0.55, 10.65, 4.65, "모니터링 · 운영 (대시보드 · 실행)", "dash")

# ── 로그 수집 · 실시간 판정 ──────────────────────────────────────
usecase("send", 3.5, 7.05, "로그 전송\n(POST /logs)", w=1.8, strong=True)
usecase("match", 5.5, 7.72, "템플릿 매칭 ·\n블록 ID 추출", w=1.75)
usecase("judge", 5.5, 7.02, "실시간 이상 판정\n(Autoencoder)", w=1.75)
usecase("store", 5.5, 6.32, "판정 결과 저장", w=1.75)
usecase("close", 3.5, 5.3, "세션 종료 판정 요청\n(재생 종료 시)", w=1.95)
usecase("idle", 3.5, 6.18, "오래 소식 없는 블록\n자동 종료 판정", w=1.8)
rel("send", "match", "«include»", lpos=0.45, loff=(-0.05, 0.12))
rel("send", "judge", "«include»", lpos=0.5, loff=(0, 0.1))
rel("send", "store", "«include»", lpos=0.5, loff=(0.05, -0.1))
rel("idle", "send", "«extend»", lpos=0.5, loff=(0.42, 0))

# ── 이상 알림 ───────────────────────────────────────────────────
usecase("alert", 9.55, 7.5, "이상 알림 전송", w=1.8, kind="notify", strong=True)
usecase("classify", 7.45, 7.6, "이상 유형 분류 ·\n등급 결정", w=1.6, kind="notify")
usecase("urgent", 7.55, 6.7, "긴급 알림 (즉시)\n새 유형 첫 발생 · 급증", w=2.05, kind="notify")
usecase("digest", 7.55, 5.85, "경고 요약\n(20초마다 1건)", w=1.85, kind="notify")
usecase("retry", 9.4, 5.3, "실패 알림 재전송\n(--retry-failed)", w=1.95, kind="notify")
rel("alert", "classify", "«include»", lpos=0.5, loff=(0, 0.13))
generalize("urgent", "alert")
generalize("digest", "alert")

# ── 오프라인 학습 · 평가 ────────────────────────────────────────
usecase("prep", 4.5, 3.85, "로그 전처리\n(Drain 파싱 · 블록 세션 · 분할)", w=2.7, kind="model")
usecase("train", 4.5, 3.0, "모델 학습 · 임계치 선택\n(검증 데이터)", w=2.6, kind="model")
usecase("eval", 4.5, 2.15, "최종 평가\n(test 1회)", w=2.1, kind="model")
usecase("export", 4.5, 1.3, "학습 산출물 내보내기\n(모델 · 템플릿 · 어휘)", w=2.5, kind="model")

# ── 모니터링 · 운영 ─────────────────────────────────────────────
dash = [("live", 4.05, "실시간 모니터링 조회\n(점수 추이 · 지표)"), ("detail", 3.4, "세션 상세 ·\n판정 근거 확인"),
        ("history", 2.75, "알림 이력 조회"), ("evalview", 2.1, "모델 평가 결과 조회"),
        ("run", 1.45, "시스템 실행\n(start.sh 모드 선택)"), ("health", 0.85, "서버 상태 조회\n(/health · /stats)")]
for key, y, text in dash:
    usecase(key, 9.55, y, text, w=1.95, h=0.5, kind="dash")
usecase("search", 7.5, 4.12, "Block ID 검색 ·\n이상만 보기", w=1.75, h=0.5, kind="dash")
usecase("csv", 7.5, 3.45, "조회 결과\nCSV 다운로드", w=1.6, h=0.5, kind="dash")
usecase("link", 7.5, 2.7, "Slack 카드 링크로\n블록 열기", w=1.75, h=0.5, kind="dash")
rel("search", "live", "«extend»", lpos=0.5, loff=(0, 0.11))
rel("csv", "live", "«extend»", lpos=0.5, loff=(0.1, -0.12))
rel("link", "detail", "«extend»", lpos=0.5, loff=(0.05, -0.12))

# ── 액터 ───────────────────────────────────────────────────────
collector = actor(1.15, 6.55, "로그 수집기", "HDFS 서버 ·\n로그 재생 모듈 · Locust")
developer = actor(1.15, 2.55, "개발자", "모델 담당")
operator = actor(11.95, 3.3, "운영자", "서버 관리자")
system_actor(11.25, 6.95, 12.7, 7.95, "Slack", "Incoming Webhook")
slack = (11.25, 7.45)

for k in ("send", "close"):
    assoc(collector, k)
for k in ("prep", "train", "eval", "export"):
    assoc(developer, k)
assoc(slack, "alert", COLORS["notify"][1])
for k in ("retry", "live", "detail", "history", "evalview", "run", "health"):
    assoc(operator, k)
# 운영자는 Slack 채널에서 알림을 본다 (시스템 바깥의 액터끼리 관계)
ax.annotate("", xy=(11.95, 3.95), xytext=(11.95, 6.95), zorder=2,
            arrowprops=dict(arrowstyle="-|>", color=COLORS["notify"][1], lw=1.2, ls=(0, (4, 3)), mutation_scale=12))
ax.text(12.05, 5.45, "알림 확인", ha="left", va="center", fontsize=8.4, color=COLORS["notify"][1])

# 범례
ax.text(6.575, 0.17, "실선: 액터 연결   점선 «include»: 항상 포함   점선 «extend»: 조건부로 덧붙음   빈 삼각형: 일반화(~의 한 종류)",
        ha="center", va="center", fontsize=8, color=SUB)

save(fig, OUT)
