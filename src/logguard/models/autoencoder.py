"""Autoencoder: 정상 윈도우를 압축·복원하도록 학습하고 재구성 오차로 이상을 판정한다.

입력 : 세션의 각 이벤트(EOS 포함)로 끝나는 길이 w 윈도우 → 원-핫 (w × C) 을 펼친 벡터
구조 : MLP  w·C → 64 → 16(잠재) → 64 → w·C   (출력은 위치별 C 클래스 logits)
손실 : 위치별 Cross-Entropy 합 = 각 위치에서 원래 이벤트를 복원할 확률의 −log 합 (원-핫에는 MSE 보다 적합)
판정 : 윈도우 재구성 오차 → 세션 최댓값 → val 정상 세션 p 백분위수 임계치 (계획서 방식), UNK 세션 즉시 이상
평가 : 이 단계에서는 val 만. test 는 6단계에서 1회

출력 : results/autoencoder.json (val), artifacts/models/autoencoder.pt

사용법:
    python -m logguard.models.autoencoder
    python -m logguard.models.autoencoder --no-eos     # EOS 제거 비교 → results/autoencoder_noeos.json

────────────────────────────────────────────────────────────────────────────────────────────
Autoencoder 를 한 문장으로: "정상 패턴만 외운 압축기는 처음 보는 패턴을 잘 복원하지 못한다"
────────────────────────────────────────────────────────────────────────────────────────────
    1) 윈도우(이벤트 20개)를 아주 작은 요약(32개 숫자)으로 압축한 뒤(인코더),
    2) 그 요약만 보고 원래 윈도우를 다시 그려 낸다(디코더).
    3) 정상 윈도우만으로 학습하면, 정상 패턴은 거의 완벽하게 복원하게 된다.
    4) 이상한 윈도우가 들어오면 요약에 담을 수 없는 패턴이라 복원이 틀린다 → 복원 오차가 크면 이상.

예시
    정상 윈도우 [할당, 수신, 수신, 수신완료, …] → 복원 [할당, 수신, 수신, 수신완료, …]  오차 ≈ 0.001
    이상 윈도우 [할당, 수신, 삭제, EOS, …]       → 복원 [할당, 수신, 수신, 수신완료, …]  오차 ≈ 25 (임계치 6.78 초과)

최종 설정·결과 (5~6단계)
    w=20, 잠재 32 (440 → 64 → 32 → 64 → 440), 300 에폭, cosine 학습률
    test F1 0.977 (시드 3개 평균 0.978 ± 0.001) → 최종 채택 모델 (inference.RECOMMENDED_DETECTOR)
"""
from __future__ import annotations

import argparse
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from logguard.config import load_config
from logguard.data.windows import EOS, PAD, UNK, encode, load_vocab, make_lookup
from logguard.eval.metrics import prf
from logguard.models.dataset import build_windows, encode_sessions, unk_mask
from logguard.models.inference import DetectionResult, Detector
from logguard.models.scoring import normalize, save_result, save_test_scores, search_threshold, session_scores
from logguard.models.train_utils import get_device, sample_weights, seed_everything, train_weighted


class WindowAutoencoder(nn.Module):
    """윈도우(이벤트 번호 w개) → 원-핫 → 압축 → 복원 → 위치별 클래스 점수.

    원-핫(one-hot)이란? 번호 하나를 "그 칸만 1, 나머지는 0" 인 벡터로 표현하는 것.
        클래스 22개일 때 번호 3 → [0,0,0,1,0,…,0] (길이 22)
        윈도우 20칸 × 22 = 440 개 숫자로 된 하나의 긴 벡터가 모델 입력이 된다.
    """

    def __init__(self, window_size: int, num_classes: int, hidden_dims: list[int]):
        """
        Args:
            window_size: 윈도우 길이 w (최종 20)
            num_classes: 클래스 수 C (22)
            hidden_dims: 인코더 층 크기들. [64, 32] → 440 → 64 → 32(잠재) 로 압축, 디코더는 거꾸로 32 → 64 → 440
        """
        super().__init__()
        self.w, self.C = window_size, num_classes
        dims = [window_size * num_classes, *hidden_dims]     # 예: [440, 64, 32]
        enc, dec = [], []
        # 인코더: 이웃한 크기끼리 Linear + ReLU 를 쌓는다 → [Linear(440,64), ReLU, Linear(64,32), ReLU]
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        # 디코더: 크기를 거꾸로 → [32, 64, 440]
        rdims = dims[::-1]
        for i, (a, b) in enumerate(zip(rdims[:-1], rdims[1:])):
            # 마지막 층(→440) 뒤에는 ReLU 를 붙이지 않는다: 출력은 음수도 가능한 점수(logits)여야 하기 때문
            dec += [nn.Linear(a, b)] + ([nn.ReLU()] if i < len(rdims) - 2 else [])
        # enc[:-1]: 인코더 마지막 ReLU 를 뺀다 → 잠재 벡터가 음수도 가질 수 있게 (표현 범위를 넓힘)
        self.encoder = nn.Sequential(*enc[:-1])        # 잠재층은 선형
        self.decoder = nn.Sequential(*dec)
        # ReLU: 음수는 0 으로, 양수는 그대로 통과시키는 활성화 함수 (층을 쌓아 복잡한 패턴을 표현하게 해 줌)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, w) int -> (B, w, C) logits
        """x: (배치 B, w) 이벤트 번호 → (B, w, C) 위치별·클래스별 복원 점수."""
        # one_hot: (B, w) → (B, w, C) / .float(): 신경망 계산은 실수형 / .flatten(1): (B, w*C) 로 펼침
        oh = nn.functional.one_hot(x, self.C).float().flatten(1)
        # 압축 → 복원 → (B, w*C) 를 다시 (B, w, C) 모양으로 접는다 (.view)
        return self.decoder(self.encoder(oh)).view(-1, self.w, self.C)


def recon_nll(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """위치별 −log P(원래 이벤트) (B, w).

    "복원 결과에서 원래 이벤트에 준 확률" 이 높을수록 잘 복원한 것. 그 확률의 −log 가 위치별 복원 오차다.
    (정확히 복원하면 P≈1 → 오차≈0, 틀리게 복원하면 P≈0 → 오차가 크다)
    원-핫 입력에는 MSE(제곱오차)보다 이 방식(Cross-Entropy)이 분류 문제의 성격에 맞다.
    """
    # log_softmax(…, -1): 위치마다 22개 클래스 점수 → 로그 확률
    # gather(2, x[..., None]): 각 위치에서 "원래 이벤트" 칸의 로그 확률만 뽑기 → (B, w, 1) → squeeze → (B, w)
    return -torch.log_softmax(logits, -1).gather(2, x[..., None]).squeeze(2)


@torch.no_grad()
def window_errors(model: WindowAutoencoder, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """고유 윈도우별 재구성 오차 (위치별 오차의 합). 클수록 이상."""
    model.eval().cpu()
    out = []
    for i in range(0, len(X), batch_size):
        x = torch.as_tensor(X[i:i + batch_size])
        out.append(recon_nll(model(x), x).sum(1).double().numpy())     # .sum(1): 위치 w 개의 오차를 합산
    return np.concatenate(out)


# --------------------------------------------------------------------------- Detector (백엔드용)

class AutoencoderDetector(Detector):
    """학습된 Autoencoder 를 백엔드용 Detector 인터페이스로 감싼 것. ← 최종 채택 모델"""
    name = "autoencoder"

    def __init__(self, model: WindowAutoencoder, meta: dict):
        """meta: 윈도우 크기, 임계치(threshold), 어휘, 템플릿 문장 등 판정에 필요한 정보 (.pt 파일에 함께 저장됨)."""
        self.model = model.eval().cpu()
        self.meta = meta
        self.w = meta["window_size"]
        self.lut = make_lookup(meta["event_to_index"])
        self.index_to_event = {int(i): int(e) for e, i in meta["event_to_index"].items()}
        self.templates = {int(k): v for k, v in meta["templates"].items()}

    @classmethod
    def load(cls, path) -> "AutoencoderDetector":
        """artifacts/models/autoencoder.pt 를 불러와 Detector 를 만든다 (구조는 DeepLogDetector.load 와 같음)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ckpt["meta"]
        model = WindowAutoencoder(m["window_size"], m["num_classes"], m["hidden_dims"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, m)

    def _label(self, idx: int) -> dict:
        """모델 번호 → {"event_id", "template"}. AE 는 윈도우 앞쪽 PAD 도 보여 줄 수 있어 PAD 처리가 추가됨."""
        if idx == EOS:
            return {"event_id": None, "template": "<EOS: 세션 종료>"}
        if idx == PAD:
            return {"event_id": None, "template": "<PAD>"}
        e = self.index_to_event.get(int(idx))
        return {"event_id": e, "template": self.templates.get(e) if e is not None else None}

    def _judge(self, events: Sequence[int | None], seq: np.ndarray, positions: np.ndarray) -> DetectionResult:
        """positions 위치에서 끝나는 윈도우들의 복원 오차를 계산하고, 가장 큰 것을 기준으로 판정한다."""
        # 1) 판정 대상 위치에 UNK 가 있으면 즉시 이상
        unk_pos = [int(p) for p in positions if seq[p] == UNK]
        if unk_pos:
            unknown = [events[p] for p in unk_pos]
            return DetectionResult(1.0, True, "unknown_event", self.name,
                                   {"unknown_events": unknown,
                                    "templates": [self.templates.get(e) for e in unknown if e is not None]})
        # 2) 위치 p 에서 "끝나는" 길이 w 윈도우 만들기 (앞에 PAD w-1 개를 붙이면 padded[p : p+w] 의 마지막이 seq[p])
        padded = np.concatenate([np.full(self.w - 1, PAD, dtype=np.int64), seq])
        X = torch.as_tensor(np.stack([padded[p:p + self.w] for p in positions]))
        with torch.no_grad():
            logits = self.model(X)                        # (윈도우 수, w, C) 복원 점수
            nll = recon_nll(logits, X).double().numpy()   # (윈도우 수, w) 위치별 복원 오차
        err = nll.sum(1)                                  # 윈도우별 재구성 오차
        j = int(np.argmax(err))                           # 가장 이상한 윈도우
        thr = float(self.meta["threshold"])               # val 로 정한 임계치 (최종 6.78)
        # 3) 판정 근거: 그 윈도우 안에서 가장 복원이 안 된 위치의 "실제 이벤트" vs "모델이 복원한 이벤트"
        worst_pos = int(np.argmax(nll[j]))                       # 윈도우 안에서 가장 복원이 안 된 위치
        recon = int(logits[j, worst_pos].argmax())               # 그 위치에서 모델이 가장 그럴듯하다고 본 이벤트
        details = {"position": int(positions[j]), "reconstruction_error": float(err[j]), "threshold": thr,
                   "window": [self._label(i)["event_id"] for i in X[j].tolist()],
                   # offset: 판정 위치 기준 몇 칸 앞인지 (0 = 판정 대상 이벤트 자체, -3 = 3칸 앞)
                   "worst": {"offset": worst_pos - (self.w - 1), "actual": self._label(int(X[j, worst_pos])),
                             "reconstructed": self._label(recon)}}
        is_anom = bool(err[j] > thr)
        return DetectionResult(float(normalize(err[j], thr)), is_anom, "model" if is_anom else "normal", self.name, details)

    def _encode(self, events: Sequence[int | None], closed: bool) -> np.ndarray:
        """Event ID 목록 → 모델 번호 (None 은 UNK 로, 세션이 끝났으면 EOS 추가). DeepLogDetector 와 동일."""
        ev = np.asarray([10**9 if e is None else int(e) for e in events], dtype=np.int64)
        return encode(ev, self.lut, add_eos=closed and self.meta["add_eos"])

    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        """세션 전체(모든 위치, EOS 포함)를 판정."""
        seq = self._encode(events, closed)
        return self._judge(list(events), seq, np.arange(len(seq)))

    def score_event(self, history: Sequence[int | None]) -> DetectionResult:
        """스트리밍: 마지막 이벤트로 끝나는 윈도우 하나만 판정 (로그 1줄당 약 0.09ms)."""
        seq = self._encode(history, closed=False)
        return self._judge(list(history), seq, np.array([len(seq) - 1]))


# --------------------------------------------------------------------------- 학습·평가

def run(cfg: dict, add_eos: bool | None = None, tag: str = "autoencoder", export: bool = True,
        evaluate_test: bool = False) -> dict:
    """evaluate_test=True 는 6단계 최종 평가에서만 사용. val 로 고른 임계치를 test 에 그대로 적용.

    순서와 인자의 의미는 deeplog.run() 과 같다. 차이점:
        - 윈도우에 "다음 이벤트" 가 없다 (with_next=False): 윈도우 자체를 복원하는 문제
        - 판정 방식은 백분위수 임계치 하나뿐 (top-k 개념이 없음)
        - 참고 지표로 "val 정상 세션 중 학습에 없던 윈도우를 포함한 비율" 을 기록 (AE 오탐 원인 분석용)
    """
    ac = dict(cfg["autoencoder"])
    if add_eos is not None:
        ac["add_eos"] = add_eos
    seed_everything(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    vocab = load_vocab(proc / "event_vocab.json")
    w, C = ac["window_size"], vocab["num_classes"]

    # ── 데이터 준비 ───────────────────────────────────────────────────────────────────────
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    tr_seqs = encode_sessions(train, vocab, add_eos=ac["add_eos"])
    va_seqs = encode_sessions(val, vocab, add_eos=ac["add_eos"])

    # ── 학습 ─────────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    tr = build_windows(tr_seqs, w, with_next=False)
    print(f"[{tag}] train windows {tr.n_total:,} -> unique {len(tr.X):,}  (eos={ac['add_eos']}, w={w})")
    model = WindowAutoencoder(w, C, ac["hidden_dims"])
    n_params = sum(p.numel() for p in model.parameters())
    # 손실 = 윈도우의 재구성 오차 (위치별 오차의 합). 정상 윈도우를 잘 복원하도록 학습한다
    loss_fn = lambda m, xb: recon_nll(m(xb), xb).sum(1)
    history = train_weighted(model, (tr.X,), sample_weights(tr.weight, ac["weight_power"]), loss_fn,
                             epochs=ac["epochs"], batch_size=ac["batch_size"], lr=ac["lr"],
                             device=get_device(ac["device"]), seed=cfg["seed"],
                             lr_schedule=ac.get("lr_schedule", "none"), log_every=10)
    train_sec = time.time() - t0

    # ── val 채점 → 임계치 선택 ──────────────────────────────────────────────────────────────
    va = build_windows(va_seqs, w, with_next=False)
    err = window_errors(model, va.X)
    unk = unk_mask(va_seqs)
    y_val = val["label"].to_numpy()
    scores = session_scores(err, va, unk)                       # 세션 점수 = 윈도우 오차 최댓값
    table, best = search_threshold(scores, y_val, cfg["threshold"]["percentile_grid"])

    # 학습에 없던(처음 보는) 정상 윈도우 비율 — AE 오탐의 주원인 후보
    #   AE 는 "외운 패턴" 만 잘 복원하므로, 정상인데도 학습에 없던 윈도우가 있으면 오탐이 날 수 있다.
    tr_keys = {tuple(r) for r in tr.X.tolist()}                  # 학습 윈도우 집합 (빠른 포함 검사용)
    unseen = np.fromiter((tuple(r) not in tr_keys for r in va.X.tolist()), dtype=bool, count=len(va.X))
    normal_sess = np.zeros(va.n_sessions, bool)
    # 세션 안에 "처음 보는 윈도우" 가 하나라도 있으면 True (logical_or.at: 세션별로 OR 누적)
    np.logical_or.at(normal_sess, va.session_index, unseen[va.inverse])
    unseen_rate = float(normal_sess[y_val == 0].mean())          # 정상 세션 중 그런 세션의 비율

    mode = {"mode": "percentile", "percentile": best["percentile"], "threshold": best["threshold"]}
    result = {
        "stage": "val_only (test 는 6단계에서 1회 평가)",
        "config": ac, "num_params": n_params, "train_seconds": round(train_sec, 1),
        "train_windows": tr.n_total, "train_unique_windows": len(tr.X), "loss_history": history,
        "selected_on": "val_f1", "best_config": mode,
        "val": {k: best[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")},
        "val_percentile_grid": table.to_dict(orient="records"),
        "val_normal_sessions_with_unseen_window": round(unseen_rate, 5),
    }

    # ── (6단계 전용) test 평가: val 로 정한 임계치를 그대로 적용 ──────────────────────────────
    if evaluate_test:
        test = pd.read_parquet(proc / "test.parquet")
        te_seqs = encode_sessions(test, vocab, add_eos=ac["add_eos"])
        te = build_windows(te_seqs, w, with_next=False)
        te_scores = session_scores(window_errors(model, te.X), te, unk_mask(te_seqs))
        result["test"] = prf(test["label"], te_scores > mode["threshold"])
        result["stage"] = "final (val 로 선택한 설정을 test 에 1회 적용)"
        save_test_scores(cfg, tag, test["label"].to_numpy(), te_scores, mode)

    # ── 모델 저장 + 저장본 검증 (deeplog.run 과 같은 방식) ──────────────────────────────────
    if export:
        tpl = pd.read_csv(cfg["paths"]["interim_dir"] / "hdfs_templates.csv")
        meta = {**ac, **mode, "num_classes": C, "event_to_index": vocab["event_to_index"],
                "templates": dict(zip(tpl["event_id"].astype(int).astype(str), tpl["template"]))}
        out = cfg["paths"]["artifacts_dir"] / "models" / "autoencoder.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": meta}, out)
        det = AutoencoderDetector.load(out)
        # 저장본 Detector 판정 == 일괄 계산 판정 인지(1.0 이어야 함), 세션·이벤트당 판정 시간
        sample = val.sample(2000, random_state=cfg["seed"])
        t1 = time.perf_counter()
        pred = [det.score_session(s.tolist()).is_anomaly for s in sample["events"]]
        sess_ms = (time.perf_counter() - t1) / len(sample) * 1e3
        ref = scores[sample.index.to_numpy()] > mode["threshold"]
        t2 = time.perf_counter()
        for s in sample["events"].head(500):
            det.score_event(s.tolist())
        event_ms = (time.perf_counter() - t2) / 500 * 1e3
        result["detector_check"] = {"agree_with_batch_eval": float(np.mean(np.asarray(pred) == ref)),
                                    "ms_per_session": round(sess_ms, 3), "ms_per_event": round(event_ms, 3)}
        result["artifact"] = str(out.relative_to(cfg["paths"]["artifacts_dir"].parent))

    # ── 결과 저장·출력 ─────────────────────────────────────────────────────────────────────
    save_result(cfg, tag, result)
    print(f"[{tag}] params {n_params:,}, train {train_sec:.0f}s, mode={mode}")
    print(f"[{tag}] VAL P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}")
    print(f"[{tag}] val 정상 세션 중 학습에 없던 윈도우 포함 비율: {unseen_rate:.4%}")
    if "test" in result:
        t = result["test"]
        print(f"[{tag}] TEST P={t['precision']:.4f} R={t['recall']:.4f} F1={t['f1']:.4f}")
    if "detector_check" in result:
        print(f"[{tag}] detector {result['detector_check']}")
    return result


def main() -> None:
    """명령줄 진입점 (`make autoencoder`). --no-eos 는 EOS 없는 비교 실험 (모델 저장 안 함)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-eos", action="store_true", help="EOS 없이 학습 (비교 실험, 모델 내보내기 안 함)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.no_eos:
        run(cfg, add_eos=False, tag="autoencoder_noeos", export=False)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
