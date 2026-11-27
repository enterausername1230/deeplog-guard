"""보고서용 다이어그램을 그리는 공통 도구 (matplotlib). 작성: 옥상민 / 2026-11-27

왜 직접 그리나?
    Graphviz(dot)는 자동 배치라서 "박스 칸 안에서 왼쪽 → 오른쪽 순서" 같은 배치를 고정하기 어렵다.
    시스템 구성도·유스케이스처럼 위치가 의미를 갖는 그림은 좌표를 직접 정해서 그린다.

좌표계: 그림 왼쪽 아래가 (0, 0), 단위는 인치. 박스는 (왼쪽 x, 아래 y, 오른쪽 x, 위 y) 로 지정한다.
사용: 각 그림 파일(예: 1_system_architecture.py)에서 import 해서 쓴다. 실행하면 같은 이름의 .png 가 생긴다.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")                      # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle

FONT = "Apple SD Gothic Neo"               # macOS 기본 한글 글꼴 (Graphviz 그림과 같은 글꼴)
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False

# 영역별 색 (채움, 테두리) — 다섯 그림에서 같은 의미는 같은 색
COLORS = {
    "source":   ("#F1F5F9", "#64748B"),    # 로그 발생원·외부
    "backend":  ("#EFF6FF", "#2563EB"),    # 백엔드 서버
    "db":       ("#FFFBEB", "#D97706"),    # 저장소
    "notify":   ("#F5F3FF", "#7C3AED"),    # Slack 알림
    "dash":     ("#ECFEFF", "#0891B2"),    # 대시보드
    "artifact": ("#F8FAFC", "#94A3B8"),    # 학습 산출물·파일
    "model":    ("#F0FDF4", "#16A34A"),    # 모델·학습
}
INK, SUB = "#0F172A", "#475569"            # 본문 글자색, 보조 글자색


def canvas(w: float, h: float, title: str, subtitle: str = ""):
    """w × h 인치 그림을 만들고 제목을 단다. (fig, ax) 반환."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(w / 2, h - 0.32, title, ha="center", va="center", fontsize=17, weight="bold", color=INK)
    if subtitle:
        ax.text(w / 2, h - 0.62, subtitle, ha="center", va="center", fontsize=10.5, color=SUB)
    return fig, ax


def group(ax, x0, y0, x1, y1, title, kind, note="", dashed=False, align="left"):
    """영역(묶음) 배경 상자. 왼쪽 위에 제목, 그 옆에 작은 설명."""
    fc, ec = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc if not dashed else "none", ec=ec, lw=1.3, ls="--" if dashed else "-", zorder=0))
    # 제목은 연결선보다 위(zorder 8)에, 영역 배경색 바탕으로 → 선이 제목 뒤로 지나가 글자가 가려지지 않는다
    bg = dict(boxstyle="round,pad=0.12", fc=fc if not dashed else "white", ec="none")
    right = align == "right"                    # 오른쪽 정렬: 왼쪽 위로 연결선이 지나갈 때 제목을 피하게
    t = ax.text(x1 - 0.16 if right else x0 + 0.16, y1 - 0.2, title, ha="right" if right else "left", va="center",
                fontsize=11.5, weight="bold", color=INK, zorder=8, bbox=bg)
    if note:
        ax.annotate(note, xy=(0, 0.5) if right else (1, 0.5), xycoords=t, xytext=(-7 if right else 7, 0),
                    textcoords="offset points", ha="right" if right else "left", va="center", fontsize=9,
                    color=SUB, zorder=8, bbox=bg)


def box(ax, x0, y0, x1, y1, title, sub="", kind="backend", fill=None, lw=1.2, dashed=False, strong=False,
        title_size=10.5, sub_size=8.6):
    """둥근 사각형 박스: 굵은 제목 + 아래 설명 (여러 줄은 \\n). 가운데 좌표 반환."""
    fc, ec = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill or "white", ec=ec if strong else "#334155",
                                lw=2.0 if strong else lw, ls="--" if dashed else "-", zorder=3))
    _label(ax, (x0 + x1) / 2, (y0 + y1) / 2, title, sub, title_size, sub_size)
    return (x0 + x1) / 2, (y0 + y1) / 2


def note(ax, x0, y0, x1, y1, title, sub=""):
    """파일 모양(오른쪽 위 모서리가 접힌) 박스."""
    f = 0.13
    ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1 - f), (x1 - f, y1), (x0, y1)], closed=True,
                         fc="#F8FAFC", ec="#334155", lw=1.1, zorder=3))
    ax.add_patch(Polygon([(x1 - f, y1), (x1 - f, y1 - f), (x1, y1 - f)], closed=True, fc="#E2E8F0",
                         ec="#334155", lw=1.0, zorder=4))
    _label(ax, (x0 + x1) / 2, (y0 + y1) / 2, title, sub, 9.5, 8.2, bold=False)


def cylinder(ax, x0, y0, x1, y1, title, sub=""):
    """DB 테이블 원통."""
    fc, ec = COLORS["db"]
    e = 0.11                                            # 위·아래 타원 높이의 절반
    ax.add_patch(Rectangle((x0, y0 + e), x1 - x0, y1 - y0 - 2 * e, fc="#FEF3C7", ec="none", zorder=3))
    ax.plot([x0, x0], [y0 + e, y1 - e], color="#334155", lw=1.2, zorder=4)
    ax.plot([x1, x1], [y0 + e, y1 - e], color="#334155", lw=1.2, zorder=4)
    ax.add_patch(Ellipse(((x0 + x1) / 2, y0 + e), x1 - x0, 2 * e, fc="#FEF3C7", ec="#334155", lw=1.2, zorder=2.9))
    ax.add_patch(Rectangle((x0 + 0.01, y0 + e), x1 - x0 - 0.02, 0.02, fc="#FEF3C7", ec="none", zorder=3.1))
    ax.add_patch(Ellipse(((x0 + x1) / 2, y1 - e), x1 - x0, 2 * e, fc="#FDE68A", ec="#334155", lw=1.2, zorder=5))
    _label(ax, (x0 + x1) / 2, (y0 + y1) / 2 - 0.05, title, sub, 10.5, 8.4)


def _label(ax, cx, cy, title, sub, ts, ss, bold=True):
    if sub:
        n = sub.count("\n") + 1
        gap = 0.075 * n
        ax.text(cx, cy + gap + 0.02, title, ha="center", va="center", fontsize=ts,
                weight="bold" if bold else "normal", color=INK, zorder=6)
        ax.text(cx, cy - 0.09 * (n - 1) / 2 - 0.08 + 0.03 * (n - 1), sub, ha="center", va="center",
                fontsize=ss, color=SUB, zorder=6, linespacing=1.25)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=ts, weight="bold" if bold else "normal",
                color=INK, zorder=6)


def arrow(ax, p, q, label="", color="#334155", lw=1.3, dashed=False, rad=0.0, lpos=0.5, loff=(0, 0),
          head=True, lsize=8.4, both=False):
    """p → q 화살표. rad: 곡률(0 = 직선). label 은 선 위 lpos 위치(0~1)에 흰 바탕으로."""
    style = "<|-|>" if both else ("-|>" if head else "-")
    ax.annotate("", xy=q, xytext=p, zorder=2,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=(0, (4, 3)) if dashed else "-",
                                shrinkA=2, shrinkB=2, mutation_scale=13,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]
        y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=lsize, color=color if color != "#334155" else INK,
                zorder=7, linespacing=1.2, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92))


def save(fig, path, dpi=220):
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print("saved", path)
