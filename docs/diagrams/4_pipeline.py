"""4. 학습·평가 파이프라인 — 원본 로그에서 배포 모델까지 12단계.

작성: 옥상민 / 2026-11-27 (코드 기준 main a42d0e4)
근거: data/download.py, data/parse.py, data/drain_tuning.py, data/split.py(+ data/processed/split_summary.json),
      data/windows.py, models/autoencoder.py(+ results/autoencoder.json), models/tune.py, models/seed_study.py,
      eval/final_eval.py(+ results/final_test.csv)
실행: .venv/bin/python docs/diagrams/4_pipeline.py  → 4_pipeline.png
"""
from pathlib import Path

from _draw import COLORS, INK, SUB, arrow, box, canvas, group, save

OUT = Path(__file__).with_suffix(".png")
fig, ax = canvas(12, 8.4, "학습 · 평가 파이프라인",
                 "원본 로그 → 구조화 → 블록 세션 → 학습(val로 모든 설정 선택) → test 1회 평가 → 실시간 서버에 배포")

XS = [(0.4, 2.95), (3.3, 5.85), (6.2, 8.75), (9.1, 11.65)]     # 한 줄에 단계 4개


def row(y0, y1, steps, kind, fill, reverse=False):
    """단계 4개를 한 줄에. reverse=True 면 오른쪽 → 왼쪽 (지그재그 배치로 줄 사이 연결을 짧게)."""
    xs = XS[::-1] if reverse else XS
    for (x0, x1), (num, title, sub, src) in zip(xs, steps):
        box(ax, x0, y0, x1, y1, f"{num}  {title}", sub, kind=kind, fill=fill, sub_size=8.2, title_size=10.2)
        ax.text((x0 + x1) / 2, y0 - 0.14, src, ha="center", va="center", fontsize=7.4, color="#64748B",
                family="monospace")
    ym = (y0 + y1) / 2
    for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
        arrow(ax, (a0, ym) if reverse else (a1, ym), (b1, ym) if reverse else (b0, ym), lw=1.6)


# ── ① 데이터 준비 ──────────────────────────────────────────────
group(ax, 0.2, 5.75, 11.8, 7.65, "① 데이터 준비", "source", "원본 로그를 모델이 읽을 수 있는 번호 순서로")
row(6.15, 7.15, [
    ("1", "HDFS_v1 다운로드", "로그 11,175,629줄 (1.5GB)\n블록 575,061개 · 블록별 정답 라벨", "data/download.py"),
    ("2", "Drain3 로그 파싱", "바뀌는 값 마스킹 → 템플릿 52개\nsim_th 0.85 · 공식 정답과 100% 일치", "data/parse.py · drain_tuning.py"),
    ("3", "블록 세션 만들기", "같은 blk_ID 로그의 Event ID를\n순서대로 모음 (블록 1개 = 세션 1개)", "data/parse.py"),
    ("4", "세션 단위 분할 6:2:2", "train 정상만 334,933 (이상 제외)\nval 115,011 · test 115,015 (이상 2.9%)", "data/split.py"),
], "source", "white")

# ── ② 학습 · 설정 선택 ─────────────────────────────────────────
group(ax, 0.2, 3.3, 11.8, 5.2, "② 학습 · 설정 선택", "model", "모든 선택은 val(검증)만 사용 — test 는 보지 않음")
row(3.7, 4.7, [
    ("5", "어휘 · 윈도우", "이벤트 19종 + PAD·UNK·EOS = 22\n세션 끝에 EOS · 길이 20 윈도우", "data/windows.py"),
    ("6", "Autoencoder 학습", "정상 윈도우만 · 300 에폭 · cosine\n고유 윈도우 89,445개 (빈도 가중)", "models/autoencoder.py"),
    ("7", "임계치 선택", "val 정상 세션 점수 백분위수 후보 중\nval F1 최대 → 99.9 백분위 = 6.78", "models/scoring.py"),
    ("8", "튜닝 · 시드 반복", "윈도우 10/15/20 × 잠재 16/32\n시드 3개 평균 val F1 → 20, 32 채택", "models/tune.py"),
], "model", "white", reverse=True)

# ── ③ 최종 평가 · 배포 ─────────────────────────────────────────
group(ax, 0.2, 0.85, 11.8, 2.75, "③ 최종 평가 · 배포", "backend", "설정을 고정한 뒤 test 는 단 한 번", align="right")
row(1.25, 2.25, [
    ("9", "test 1회 평가", "Autoencoder F1 0.977\n(정밀도 0.970 · 재현율 0.984)", "eval/final_eval.py"),
    ("10", "비교 실험", "규칙 0.573 · 통계 빈도 0.887\nDeepLog 0.926 · EOS 없는 AE 0.884", "results/final_test.csv"),
    ("11", "산출물 내보내기", "autoencoder.pt (가중치 · 임계치)\ndrain3_state.bin · event_vocab.json", "artifacts/ · data/processed/"),
    ("12", "실시간 서버 연결", "FastAPI 가 시작할 때 읽음\n윈도우 · 어휘 일치 검사 후 판정", "api/predictor.py"),
], "backend", "#DBEAFE")

# 줄 사이 연결: ①의 마지막(오른쪽) → ②의 처음(오른쪽), ②의 마지막(왼쪽) → ③의 처음(왼쪽)
arrow(ax, (11.4, 6.15), (11.4, 4.7), color=COLORS["model"][1], lw=1.8)
arrow(ax, (0.65, 3.7), (0.65, 2.25), color=COLORS["backend"][1], lw=1.8)

# 아래 설명 줄
ax.text(6.0, 0.45, "데이터 누수 방지: 블록 단위로 먼저 나눈 뒤 각 셋 안에서만 윈도우를 만든다  ·  "
        "과적합 확인: val F1 0.972 ↔ test F1 0.977, 시드 3개 평균 0.978 ± 0.001",
        ha="center", va="center", fontsize=8.6, color=SUB)

save(fig, OUT)
