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
from logguard.models.dataset import build_windows, encode_sessions, unk_mask
from logguard.models.inference import DetectionResult, Detector
from logguard.models.scoring import normalize, save_result, search_threshold, session_scores
from logguard.models.train_utils import get_device, sample_weights, seed_everything, train_weighted


class WindowAutoencoder(nn.Module):
    def __init__(self, window_size: int, num_classes: int, hidden_dims: list[int]):
        super().__init__()
        self.w, self.C = window_size, num_classes
        dims = [window_size * num_classes, *hidden_dims]
        enc, dec = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        rdims = dims[::-1]
        for i, (a, b) in enumerate(zip(rdims[:-1], rdims[1:])):
            dec += [nn.Linear(a, b)] + ([nn.ReLU()] if i < len(rdims) - 2 else [])
        self.encoder = nn.Sequential(*enc[:-1])        # 잠재층은 선형
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, w) int -> (B, w, C) logits
        oh = nn.functional.one_hot(x, self.C).float().flatten(1)
        return self.decoder(self.encoder(oh)).view(-1, self.w, self.C)


def recon_nll(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """위치별 −log P(원래 이벤트) (B, w)."""
    return -torch.log_softmax(logits, -1).gather(2, x[..., None]).squeeze(2)


@torch.no_grad()
def window_errors(model: WindowAutoencoder, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    model.eval().cpu()
    out = []
    for i in range(0, len(X), batch_size):
        x = torch.as_tensor(X[i:i + batch_size])
        out.append(recon_nll(model(x), x).sum(1).double().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- Detector (백엔드용)

class AutoencoderDetector(Detector):
    name = "autoencoder"

    def __init__(self, model: WindowAutoencoder, meta: dict):
        self.model = model.eval().cpu()
        self.meta = meta
        self.w = meta["window_size"]
        self.lut = make_lookup(meta["event_to_index"])
        self.index_to_event = {int(i): int(e) for e, i in meta["event_to_index"].items()}
        self.templates = {int(k): v for k, v in meta["templates"].items()}

    @classmethod
    def load(cls, path) -> "AutoencoderDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ckpt["meta"]
        model = WindowAutoencoder(m["window_size"], m["num_classes"], m["hidden_dims"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, m)

    def _label(self, idx: int) -> dict:
        if idx == EOS:
            return {"event_id": None, "template": "<EOS: 세션 종료>"}
        if idx == PAD:
            return {"event_id": None, "template": "<PAD>"}
        e = self.index_to_event.get(int(idx))
        return {"event_id": e, "template": self.templates.get(e) if e is not None else None}

    def _judge(self, events: Sequence[int | None], seq: np.ndarray, positions: np.ndarray) -> DetectionResult:
        unk_pos = [int(p) for p in positions if seq[p] == UNK]
        if unk_pos:
            unknown = [events[p] for p in unk_pos]
            return DetectionResult(1.0, True, "unknown_event", self.name,
                                   {"unknown_events": unknown,
                                    "templates": [self.templates.get(e) for e in unknown if e is not None]})
        padded = np.concatenate([np.full(self.w - 1, PAD, dtype=np.int64), seq])
        X = torch.as_tensor(np.stack([padded[p:p + self.w] for p in positions]))
        with torch.no_grad():
            logits = self.model(X)
            nll = recon_nll(logits, X).double().numpy()
        err = nll.sum(1)
        j = int(np.argmax(err))
        thr = float(self.meta["threshold"])
        worst_pos = int(np.argmax(nll[j]))                       # 윈도우 안에서 가장 복원이 안 된 위치
        recon = int(logits[j, worst_pos].argmax())
        details = {"position": int(positions[j]), "reconstruction_error": float(err[j]), "threshold": thr,
                   "window": [self._label(i)["event_id"] for i in X[j].tolist()],
                   "worst": {"offset": worst_pos - (self.w - 1), "actual": self._label(int(X[j, worst_pos])),
                             "reconstructed": self._label(recon)}}
        is_anom = bool(err[j] > thr)
        return DetectionResult(float(normalize(err[j], thr)), is_anom, "model" if is_anom else "normal", self.name, details)

    def _encode(self, events: Sequence[int | None], closed: bool) -> np.ndarray:
        ev = np.asarray([10**9 if e is None else int(e) for e in events], dtype=np.int64)
        return encode(ev, self.lut, add_eos=closed and self.meta["add_eos"])

    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        seq = self._encode(events, closed)
        return self._judge(list(events), seq, np.arange(len(seq)))

    def score_event(self, history: Sequence[int | None]) -> DetectionResult:
        seq = self._encode(history, closed=False)
        return self._judge(list(history), seq, np.array([len(seq) - 1]))


# --------------------------------------------------------------------------- 학습·평가

def run(cfg: dict, add_eos: bool | None = None, tag: str = "autoencoder", export: bool = True) -> dict:
    ac = dict(cfg["autoencoder"])
    if add_eos is not None:
        ac["add_eos"] = add_eos
    seed_everything(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    vocab = load_vocab(proc / "event_vocab.json")
    w, C = ac["window_size"], vocab["num_classes"]

    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    tr_seqs = encode_sessions(train, vocab, add_eos=ac["add_eos"])
    va_seqs = encode_sessions(val, vocab, add_eos=ac["add_eos"])

    t0 = time.time()
    tr = build_windows(tr_seqs, w, with_next=False)
    print(f"[{tag}] train windows {tr.n_total:,} -> unique {len(tr.X):,}  (eos={ac['add_eos']}, w={w})")
    model = WindowAutoencoder(w, C, ac["hidden_dims"])
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = lambda m, xb: recon_nll(m(xb), xb).sum(1)
    history = train_weighted(model, (tr.X,), sample_weights(tr.weight, ac["weight_power"]), loss_fn,
                             epochs=ac["epochs"], batch_size=ac["batch_size"], lr=ac["lr"],
                             device=get_device(ac["device"]), seed=cfg["seed"],
                             lr_schedule=ac.get("lr_schedule", "none"), log_every=10)
    train_sec = time.time() - t0

    va = build_windows(va_seqs, w, with_next=False)
    err = window_errors(model, va.X)
    unk = unk_mask(va_seqs)
    y_val = val["label"].to_numpy()
    scores = session_scores(err, va, unk)
    table, best = search_threshold(scores, y_val, cfg["threshold"]["percentile_grid"])

    # 학습에 없던(처음 보는) 정상 윈도우 비율 — AE 오탐의 주원인 후보
    tr_keys = {tuple(r) for r in tr.X.tolist()}
    unseen = np.fromiter((tuple(r) not in tr_keys for r in va.X.tolist()), dtype=bool, count=len(va.X))
    normal_sess = np.zeros(va.n_sessions, bool)
    np.logical_or.at(normal_sess, va.session_index, unseen[va.inverse])
    unseen_rate = float(normal_sess[y_val == 0].mean())

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

    if export:
        tpl = pd.read_csv(cfg["paths"]["interim_dir"] / "hdfs_templates.csv")
        meta = {**ac, **mode, "num_classes": C, "event_to_index": vocab["event_to_index"],
                "templates": dict(zip(tpl["event_id"].astype(int).astype(str), tpl["template"]))}
        out = cfg["paths"]["artifacts_dir"] / "models" / "autoencoder.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": meta}, out)
        det = AutoencoderDetector.load(out)
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

    save_result(cfg, tag, result)
    print(f"[{tag}] params {n_params:,}, train {train_sec:.0f}s, mode={mode}")
    print(f"[{tag}] VAL P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}")
    print(f"[{tag}] val 정상 세션 중 학습에 없던 윈도우 포함 비율: {unseen_rate:.4%}")
    if "detector_check" in result:
        print(f"[{tag}] detector {result['detector_check']}")
    return result


def main() -> None:
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
