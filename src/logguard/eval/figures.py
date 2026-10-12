"""보고서용 그래프 (6단계). final_eval 실행 후 사용.

출력 (docs/figures/):
  f1_comparison.png    방법별 test F1 (모델은 시드 3개 평균 ± 표준편차)
  pr_curves.png        test Precision-Recall 곡선 + 채택한 운영점
  score_dist_ae.png    Autoencoder test 세션 점수 분포 (정상 vs 이상) + 임계치
  eos_ablation.png     EOS 유무에 따른 test F1

사용법: python -m logguard.eval.figures

그래프 읽는 법
    PR 곡선 : 임계치를 바꿔 가며 (Recall, Precision) 을 이은 선. 오른쪽 위 모서리(1,1)에 가까울수록 좋은 모델.
              AP(Average Precision) = 곡선 아래 넓이 (1 에 가까울수록 좋음). 점은 val 로 고른 실제 운영 임계치.
    점수 분포: 파란색(정상)과 빨간색(이상)이 임계치(점선)를 사이에 두고 잘 갈라질수록 좋은 모델.
"""
from __future__ import annotations

import json

import matplotlib
import matplotlib.ticker

# "Agg" = 화면 창 없이 파일로만 그리는 모드 (서버·터미널 환경에서도 동작). pyplot 을 불러오기 "전에" 설정해야 한다
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402   (E402: import 가 파일 맨 위가 아니라는 경고를 무시)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from logguard.config import ROOT, load_config  # noqa: E402

# 그래프 공통 스타일
#   font.family: 한글이 깨지지 않도록 한글 글꼴을 우선 (Mac: AppleGothic, 리눅스: NanumGothic, Windows: Malgun Gothic)
#   axes.unicode_minus=False: 축의 마이너스 기호를 일반 하이픈으로 (한글 글꼴에 수학 마이너스 기호가 없어서)
plt.rcParams.update({"font.family": ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
# 결과 표의 방법 이름 → 그래프에 표시할 이름
LABEL = {"rule_based": "Rule-Based", "statistical_frequency": "통계 빈도", "deeplog": "DeepLog",
         "autoencoder": "Autoencoder", "deeplog_noeos": "DeepLog (EOS 없음)", "autoencoder_noeos": "Autoencoder (EOS 없음)"}
# 방법별 색 (모든 그래프에서 같은 방법은 같은 색)
COLOR = {"rule_based": "#9e9e9e", "statistical_frequency": "#607d8b", "deeplog": "#1f77b4", "autoencoder": "#d62728"}


def finite(s: np.ndarray) -> np.ndarray:
    """UNK 세션(+inf)은 유한 최댓값보다 큰 값으로 바꿔 곡선 계산에 포함.

    sklearn 의 PR 곡선 함수는 무한대 값을 처리하지 못한다. UNK 세션은 "가장 확실한 이상" 이므로
    다른 어떤 점수보다 큰 유한한 값(최댓값 × 10 + 1)으로 바꿔도 순서(=곡선 모양)는 그대로다.
    """
    s = s.astype(float).copy()          # 원본 배열을 바꾸지 않도록 복사
    fin = np.isfinite(s)
    s[~fin] = (s[fin].max() if fin.any() else 1.0) * 10 + 1
    return s


def statistical_scores(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """통계 빈도 탐지기의 test 세션 점수를 한 번에 계산한다 (PR 곡선용).

    점수 = 세션 안 이벤트들의 (실제 횟수 / 허용 횟수) 중 최댓값. 1 을 넘으면 이상.
    StatisticalFrequencyDetector.score_session() 과 같은 계산을 행렬로 빠르게 한 것.
    Returns:
        (정답 라벨 배열, 점수 배열)
    """
    from logguard.baseline.rule_based import count_matrix
    det = json.loads((cfg["paths"]["artifacts_dir"] / "models" / "statistical_frequency.json").read_text())
    th = {int(k): v for k, v in det["thresholds"].items()}       # Event ID → 허용 횟수
    test = pd.read_parquet(cfg["paths"]["processed_dir"] / "test.parquet")
    # 행렬 열 개수: test 에 나온 가장 큰 Event ID 와 임계치 목록의 가장 큰 ID 중 큰 쪽 + 1
    n = max(max(int(e) for s in test["events"] for e in s), max(th)) + 1
    C = count_matrix(test, n).astype(float)                      # (세션, 이벤트) 횟수 행렬
    T = np.array([th.get(e, -1.0) for e in range(n)])            # 이벤트별 허용 횟수 (학습에 없던 이벤트는 -1 표시)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(T > 0, C / T, np.where(C > 0, np.inf, 0.0))   # 학습에 없거나 임계치 0 인 이벤트 등장 = inf
    return test["label"].to_numpy(), ratio.max(1)                # 세션별 최댓값


def main() -> None:
    """그래프 4개를 그려 docs/figures/ 에 PNG 로 저장한다."""
    cfg = load_config()
    res, ev = cfg["paths"]["results_dir"], cfg["paths"]["artifacts_dir"] / "eval" / "final"
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(res / "final_test.csv").set_index("method")    # 방법 이름으로 행을 찾을 수 있게

    # 1) F1 비교 ─────────────────────────────────────────────────────────────────────────────
    order = ["rule_based", "statistical_frequency", "deeplog", "autoencoder"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))                    # figsize: 인치 단위 크기
    vals, errs = table.loc[order, "test_f1_mean"], table.loc[order, "test_f1_std"]
    # yerr: 막대 위의 오차 막대(표준편차), capsize: 오차 막대 끝의 가로선 길이
    bars = ax.bar([LABEL[m] for m in order], vals, yerr=errs, capsize=4, color=[COLOR[m] for m in order])
    for b, v in zip(bars, vals):
        # 막대 위에 숫자 표시 (막대 가운데 x, 막대 높이보다 살짝 위 y)
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.85, ls="--", c="k", lw=1, label="목표 F1 0.85")                                  # 목표 ① 선
    ax.axhline(table.loc["rule_based", "test_f1"] + 0.10, ls=":", c="gray", lw=1, label="Rule-Based + 10%p")  # 목표 ② 선
    ax.set(ylim=(0, 1.08), ylabel="test F1", title="방법별 test F1 (딥러닝 모델: 시드 3개 평균 ± 표준편차)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()            # 글자가 잘리지 않도록 여백 자동 조정
    fig.savefig(out / "f1_comparison.png")

    # 2) PR 곡선 ─────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    curves = {"statistical_frequency": statistical_scores(cfg)}
    for m in ("deeplog", "autoencoder"):
        d = np.load(ev / f"{m}_s42.npz")              # final_eval 이 저장한 test 점수 (배포 모델 = 시드 42)
        curves[m] = (d["labels"], d["scores"])
    for m, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, finite(s))      # 임계치를 바꿔 가며 계산한 (Precision, Recall) 점들
        ax.plot(r, p, color=COLOR[m], lw=1.6, label=f"{LABEL[m]} (AP {average_precision_score(y, finite(s)):.3f})")
        # 실제로 채택한 임계치에서의 성능 = 운영점 (zorder=5: 선보다 위에 그리기)
        ax.scatter(table.loc[m, "test_recall"], table.loc[m, "test_precision"], color=COLOR[m], s=40, zorder=5,
                   edgecolor="k")
    # Rule-Based 는 점수가 아니라 참/거짓 판정이라 곡선 없이 운영점 하나만 표시
    ax.scatter(table.loc["rule_based", "test_recall"], table.loc["rule_based", "test_precision"],
               color=COLOR["rule_based"], marker="s", s=40, edgecolor="k", label="Rule-Based (운영점)")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02),
           title="test Precision-Recall (점 = val 로 고른 임계치)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png")

    # 3) Autoencoder 점수 분포 ─────────────────────────────────────────────────────────────────
    d = np.load(ev / "autoencoder_s42.npz")
    y, s, thr = d["labels"], d["scores"], float(d["threshold"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    fin = np.isfinite(s)                              # UNK 세션(무한대 점수)은 히스토그램에서 제외
    # 로그 간격 구간 60개: 점수가 0.0001 ~ 수백까지 넓게 퍼져 있어 일반 간격으로는 안 보인다
    bins = np.logspace(np.log10(max(s[fin & (s > 0)].min(), 1e-4)), np.log10(s[fin].max()), 60)
    for lab, name, c in [(0, "정상", "#1f77b4"), (1, "이상", "#d62728")]:
        # clip: 첫 구간보다 작은 점수(거의 0)는 첫 구간에 모아서 표시
        ax.hist(np.clip(s[fin & (y == lab)], bins[0], None), bins=bins, alpha=0.6, color=c, label=name)
    ax.axvline(thr, c="k", ls="--", label=f"임계치 {thr:.2f}")
    ax.set(xscale="log", yscale="log", xlabel="세션 재구성 오차 (최댓값)", ylabel="세션 수",
           title=f"Autoencoder test 점수 분포 (UNK 즉시 이상 {int((~fin & (y == 1)).sum())}개 세션 제외)")
    # 한글 글꼴에 수학 마이너스(U+2212)가 없어 10^-4 표기가 깨지므로 x축은 일반 숫자로 표기 (xscale 설정 뒤에 지정)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "score_dist_ae.png")

    # 4) EOS 효과 ────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(2)                                   # 막대 묶음 위치: DeepLog(0), Autoencoder(1)
    no = [table.loc[f"{m}_noeos", "test_f1"] for m in ("deeplog", "autoencoder")]
    yes = [table.loc[m, "test_f1"] for m in ("deeplog", "autoencoder")]
    # 묶음마다 왼쪽(−0.18) 회색 = EOS 없음, 오른쪽(+0.18) 색 = EOS 추가. 0.36 = 막대 너비
    ax.bar(x - 0.18, no, 0.36, color="#bdbdbd", label="EOS 없음")
    ax.bar(x + 0.18, yes, 0.36, color=["#1f77b4", "#d62728"], label="EOS 추가")
    for i in range(2):
        ax.text(x[i] - 0.18, no[i] + 0.01, f"{no[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.18, yes[i] + 0.01, f"{yes[i]:.3f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=["DeepLog", "Autoencoder"], ylim=(0, 1.08), ylabel="test F1 (seed 42)",
           title="세션 종료 토큰(EOS) 효과")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "eos_ablation.png")
    print("[figures] saved:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
