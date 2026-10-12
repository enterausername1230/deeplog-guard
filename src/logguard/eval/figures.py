"""보고서용 그래프 (6단계). final_eval 실행 후 사용.

출력 (docs/figures/):
  f1_comparison.png    방법별 test F1 (모델은 시드 3개 평균 ± 표준편차)
  pr_curves.png        test Precision-Recall 곡선 + 채택한 운영점
  score_dist_ae.png    Autoencoder test 세션 점수 분포 (정상 vs 이상) + 임계치
  eos_ablation.png     EOS 유무에 따른 test F1

사용법: python -m logguard.eval.figures
"""
from __future__ import annotations

import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from logguard.config import ROOT, load_config  # noqa: E402

plt.rcParams.update({"font.family": ["AppleGothic", "NanumGothic", "Malgun Gothic", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
LABEL = {"rule_based": "Rule-Based", "statistical_frequency": "통계 빈도", "deeplog": "DeepLog",
         "autoencoder": "Autoencoder", "deeplog_noeos": "DeepLog (EOS 없음)", "autoencoder_noeos": "Autoencoder (EOS 없음)"}
COLOR = {"rule_based": "#9e9e9e", "statistical_frequency": "#607d8b", "deeplog": "#1f77b4", "autoencoder": "#d62728"}


def finite(s: np.ndarray) -> np.ndarray:
    """UNK 세션(+inf)은 유한 최댓값보다 큰 값으로 바꿔 곡선 계산에 포함."""
    s = s.astype(float).copy()
    fin = np.isfinite(s)
    s[~fin] = (s[fin].max() if fin.any() else 1.0) * 10 + 1
    return s


def statistical_scores(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    from logguard.baseline.rule_based import count_matrix
    det = json.loads((cfg["paths"]["artifacts_dir"] / "models" / "statistical_frequency.json").read_text())
    th = {int(k): v for k, v in det["thresholds"].items()}
    test = pd.read_parquet(cfg["paths"]["processed_dir"] / "test.parquet")
    n = max(max(int(e) for s in test["events"] for e in s), max(th)) + 1
    C = count_matrix(test, n).astype(float)
    T = np.array([th.get(e, -1.0) for e in range(n)])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(T > 0, C / T, np.where(C > 0, np.inf, 0.0))   # 학습에 없거나 임계치 0 인 이벤트 등장 = inf
    return test["label"].to_numpy(), ratio.max(1)


def main() -> None:
    cfg = load_config()
    res, ev = cfg["paths"]["results_dir"], cfg["paths"]["artifacts_dir"] / "eval" / "final"
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(res / "final_test.csv").set_index("method")

    # 1) F1 비교
    order = ["rule_based", "statistical_frequency", "deeplog", "autoencoder"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    vals, errs = table.loc[order, "test_f1_mean"], table.loc[order, "test_f1_std"]
    bars = ax.bar([LABEL[m] for m in order], vals, yerr=errs, capsize=4, color=[COLOR[m] for m in order])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.85, ls="--", c="k", lw=1, label="목표 F1 0.85")
    ax.axhline(table.loc["rule_based", "test_f1"] + 0.10, ls=":", c="gray", lw=1, label="Rule-Based + 10%p")
    ax.set(ylim=(0, 1.08), ylabel="test F1", title="방법별 test F1 (딥러닝 모델: 시드 3개 평균 ± 표준편차)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "f1_comparison.png")

    # 2) PR 곡선
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    curves = {"statistical_frequency": statistical_scores(cfg)}
    for m in ("deeplog", "autoencoder"):
        d = np.load(ev / f"{m}_s42.npz")
        curves[m] = (d["labels"], d["scores"])
    for m, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, finite(s))
        ax.plot(r, p, color=COLOR[m], lw=1.6, label=f"{LABEL[m]} (AP {average_precision_score(y, finite(s)):.3f})")
        ax.scatter(table.loc[m, "test_recall"], table.loc[m, "test_precision"], color=COLOR[m], s=40, zorder=5,
                   edgecolor="k")
    ax.scatter(table.loc["rule_based", "test_recall"], table.loc["rule_based", "test_precision"],
               color=COLOR["rule_based"], marker="s", s=40, edgecolor="k", label="Rule-Based (운영점)")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02),
           title="test Precision-Recall (점 = val 로 고른 임계치)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png")

    # 3) Autoencoder 점수 분포
    d = np.load(ev / "autoencoder_s42.npz")
    y, s, thr = d["labels"], d["scores"], float(d["threshold"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    fin = np.isfinite(s)
    bins = np.logspace(np.log10(max(s[fin & (s > 0)].min(), 1e-4)), np.log10(s[fin].max()), 60)
    for lab, name, c in [(0, "정상", "#1f77b4"), (1, "이상", "#d62728")]:
        ax.hist(np.clip(s[fin & (y == lab)], bins[0], None), bins=bins, alpha=0.6, color=c, label=name)
    ax.axvline(thr, c="k", ls="--", label=f"임계치 {thr:.2f}")
    ax.set(xscale="log", yscale="log", xlabel="세션 재구성 오차 (최댓값)", ylabel="세션 수",
           title=f"Autoencoder test 점수 분포 (UNK 즉시 이상 {int((~fin & (y == 1)).sum())}개 세션 제외)")
    # 한글 글꼴에 수학 마이너스(U+2212)가 없어 10^-4 표기가 깨지므로 x축은 일반 숫자로 표기 (xscale 설정 뒤에 지정)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "score_dist_ae.png")

    # 4) EOS 효과
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(2)
    no = [table.loc[f"{m}_noeos", "test_f1"] for m in ("deeplog", "autoencoder")]
    yes = [table.loc[m, "test_f1"] for m in ("deeplog", "autoencoder")]
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
