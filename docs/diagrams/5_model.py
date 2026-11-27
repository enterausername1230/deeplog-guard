"""5. 모델 구조도 — Autoencoder 가 윈도우 하나를 판정하는 과정.

작성: 옥상민 / 2026-11-27 (코드 기준 main a42d0e4)
근거: models/autoencoder.py
    WindowAutoencoder : 원-핫(20 × 22 = 440) → Linear 440→64 + ReLU → Linear 64→32(잠재, 선형)
                        → Linear 32→64 + ReLU → Linear 64→440 → (20, 22) 위치별 점수
    recon_nll         : 위치별 −log P(원래 이벤트), 20칸 합 = 재구성 오차
    _judge            : 판정 위치에 UNK 가 있으면 즉시 이상(점수 1.0), 아니면 오차 > 임계치 6.78 → 이상
    score_event       : 방금 들어온 이벤트로 끝나는 윈도우 1개 (앞이 모자라면 PAD)
    score_close       : 이력 끝에 EOS 를 붙이고 EOS 로 끝나는 윈도우 1개
    models/scoring.py : 정규화 점수 = s / (s + 임계치) → 기준선 항상 0.5
    results/autoencoder.json : 파라미터 61,016개, 임계치 6.776 (val 정상 세션 99.9 백분위수)
실행: .venv/bin/python docs/diagrams/5_model.py  → 5_model.png
"""
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from _draw import COLORS, INK, SUB, arrow, box, canvas, group, save

OUT = Path(__file__).with_suffix(".png")
BLUE, GREEN, ORANGE, RED = "#2563EB", "#16A34A", "#D97706", "#DC2626"
fig, ax = canvas(12.5, 8.2, "Autoencoder 모델 구조와 판정 과정",
                 "이벤트 20개 윈도우를 32개 숫자로 압축했다가 복원 → 복원이 안 되는 정도(재구성 오차)로 이상 판정")

rng = np.random.default_rng(3)


def onehot(x0, y0, w, h, seq, color, title, sub):
    """원-핫 행렬 그림: 가로 = 윈도우 20칸, 세로 = 이벤트 22종. 칸마다 해당 이벤트 한 칸만 칠한다."""
    cw, ch = w / 20, h / 22
    ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec="#334155", lw=1.1, zorder=3))
    for i, e in enumerate(seq):
        ax.add_patch(Rectangle((x0 + i * cw, y0 + (21 - e) * ch), cw, ch, fc=color, ec="none", zorder=4))
    ax.text(x0 + w / 2, y0 + h + 0.2, title, ha="center", va="center", fontsize=10, weight="bold", color=INK)
    ax.text(x0 + w / 2, y0 - 0.22, sub, ha="center", va="center", fontsize=8, color=SUB, linespacing=1.2)


# ── 위쪽: 신경망 ────────────────────────────────────────────────
group(ax, 0.2, 3.55, 12.3, 7.45, "모델 (WindowAutoencoder · 파라미터 61,016개)", "model", "정상 윈도우만으로 학습")
seq = [0, 0, 0, 3, 4, 4, 5, 5, 6, 7, 6, 8, 9, 6, 10, 11, 6, 12, 13, 14]      # 예시: 앞 PAD 3칸 + 이벤트들
box(ax, 0.4, 5.45, 1.95, 6.55, "블록 이벤트 순서", "blk_… 의 최근 이벤트\n앞이 모자라면 PAD", kind="model",
    sub_size=8)
onehot(2.3, 4.35, 1.35, 2.25, seq, BLUE, "입력: 원-핫", "윈도우 20칸 × 이벤트 22종\n= 440개 숫자")
arrow(ax, (1.95, 6.0), (2.3, 6.0), color=INK)

# 층 막대: 높이를 크기에 비례하게 (440 은 길게, 32 는 짧게)
layers = [(4.25, 440, BLUE), (5.25, 64, BLUE), (6.25, 32, "#7C3AED"), (7.25, 64, GREEN), (8.25, 440, GREEN)]
cy = 5.47
hmap = {440: 2.25, 64: 1.05, 32: 0.6}
for (xa, na, ca), (xb, nb, cb) in zip(layers, layers[1:]):
    ha, hb = hmap[na] / 2, hmap[nb] / 2
    ax.add_patch(Polygon([(xa + 0.14, cy - ha), (xb - 0.14, cy - hb), (xb - 0.14, cy + hb), (xa + 0.14, cy + ha)],
                         closed=True, fc=ca if xa < 6.25 else cb, alpha=0.12, ec="none", zorder=2))
for x, n, c in layers:
    h = hmap[n]
    ax.add_patch(FancyBboxPatch((x - 0.14, cy - h / 2), 0.28, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=c, ec="none", alpha=0.85, zorder=3))
    ax.text(x, cy + h / 2 + 0.15, str(n), ha="center", va="center", fontsize=10, weight="bold", color=c)
ax.text(4.75, 4.05, "Linear 440→64\n+ ReLU", ha="center", va="center", fontsize=8, color=SUB, linespacing=1.15)
ax.text(5.75, 4.05, "Linear\n64→32", ha="center", va="center", fontsize=8, color=SUB, linespacing=1.15)
ax.text(6.75, 4.05, "Linear\n32→64 + ReLU", ha="center", va="center", fontsize=8, color=SUB, linespacing=1.15)
ax.text(7.75, 4.05, "Linear\n64→440", ha="center", va="center", fontsize=8, color=SUB, linespacing=1.15)
ax.text(5.25, 6.98, "인코더 (압축)", ha="center", va="center", fontsize=10, weight="bold", color=BLUE)
ax.text(6.25, 6.62, "잠재 32", ha="center", va="center", fontsize=9.5, weight="bold", color="#7C3AED")
ax.text(7.25, 6.98, "디코더 (복원)", ha="center", va="center", fontsize=10, weight="bold", color=GREEN)
arrow(ax, (3.65, cy), (4.1, cy), color=INK)

# 출력: 위치마다 22종 확률 (복원 결과). 마지막 칸은 원래와 다른 이벤트를 복원했다고 가정한 예시
recon = seq[:-1] + [6]
onehot(8.75, 4.35, 1.35, 2.25, recon, GREEN, "출력: 복원 결과", "위치마다 22종 점수 → softmax 확률\n빨간 칸: 원래와 다르게 복원한 위치(예)")
ax.add_patch(Rectangle((8.75 + 19 * 1.35 / 20, 4.35), 1.35 / 20, 2.25, fc="none", ec=RED, lw=1.3, zorder=5))
arrow(ax, (8.4, cy), (8.75, cy), color=INK)

box(ax, 10.45, 4.75, 12.1, 6.2, "재구성 오차 s", "위치마다 -log P(원래 이벤트)\n20칸을 더한 값\n잘 복원 ≈ 0 · 못 하면 큼",
    kind="model", fill="#DCFCE7", strong=True, sub_size=8)
arrow(ax, (10.1, cy), (10.45, cy), color=INK)

# ── 아래쪽: 판정 ────────────────────────────────────────────────
group(ax, 0.2, 0.3, 12.3, 3.2, "판정", "backend", "실시간 판정과 세션 종료 판정이 같은 모델을 쓴다")
box(ax, 0.4, 1.75, 3.9, 2.75, "① 실시간 판정 (로그 1줄마다)", "방금 들어온 이벤트로 끝나는 윈도우 1개\n[직전 19개, 현재] · 약 0.09ms",
    kind="backend", sub_size=8.2)
box(ax, 0.4, 0.5, 3.9, 1.5, "② 세션 종료 판정 (EOS)", "이력 끝에 EOS 를 붙인 윈도우 1개\n[마지막 19개, EOS] → 끊긴 세션 탐지",
    kind="backend", sub_size=8.2)
box(ax, 4.55, 1.95, 7.45, 2.75, "처음 보는 이벤트(UNK)?", "학습에 없던 로그 → 모델 계산 없이", kind="backend",
    fill="#FEF2F2", sub_size=8.2)
box(ax, 4.55, 0.55, 7.45, 1.55, "재구성 오차 s > 6.78 ?", "s: 위 모델이 계산한 값 · 6.78: val 정상 세션\n점수의 99.9 백분위수 (정상 1,000개 중 1개 수준)",
    kind="backend", fill="#EFF6FF", strong=True, sub_size=8.2)
box(ax, 8.15, 1.95, 10.2, 2.75, "이상 (즉시)", "점수 1.0 · unknown_event", fill="#FEE2E2", sub_size=8.2)
box(ax, 8.15, 0.55, 10.2, 1.55, "이상 / 정상", "reason = model / normal", fill="#FEF3C7", sub_size=8.2)
box(ax, 10.55, 0.55, 12.1, 2.75, "저장 · 화면 점수", "정규화 점수\ns ÷ (s + 6.78)\n→ 0~1, 기준선 0.5\n\n판정 근거도 저장\n(실제 vs 복원 이벤트)",
    kind="backend", sub_size=8.2)
arrow(ax, (3.9, 2.25), (4.55, 2.3), color=INK)
arrow(ax, (3.9, 1.0), (4.55, 2.15), color=INK)
arrow(ax, (6.0, 1.95), (6.0, 1.55), "아니오", color=INK, lpos=0.5, loff=(0.3, 0))
arrow(ax, (7.45, 2.35), (8.15, 2.35), "예", color=RED, lpos=0.5, loff=(0, 0.12))
arrow(ax, (7.45, 1.05), (8.15, 1.05), color=INK)
arrow(ax, (10.2, 2.35), (10.55, 2.35), color=INK)
arrow(ax, (10.2, 1.05), (10.55, 1.05), color=INK)

save(fig, OUT)
