"""DeepLog (Du et al., CCS 2017): 직전 w개 이벤트로 다음 이벤트를 예측하는 LSTM.

학습 : train(정상 세션)의 고유 (윈도우, 다음 이벤트) 쌍 + 등장 횟수 가중 Cross-Entropy
       시퀀스 끝에 EOS 를 붙여 '세션이 여기서 끝나도 되는가'까지 학습 (EDA 결론 #4)
판정 : 윈도우마다 실제 다음 이벤트의
       (a) topk       : 예측 순위(rank, 0=1등) ≥ k 이면 이상                       — 논문 방식
       (b) percentile : −log P(실제) 가 val 정상 세션 점수의 p 백분위수 초과면 이상 — 계획서 방식
           (계획서의 1 − P 와 순서가 같은 단조 변환. P 가 매우 작을 때 1 − P 는 float32 에서 1 로 붙어버려 −log P 사용)
       세션 점수 = 윈도우 점수 최댓값, UNK 포함 세션은 즉시 이상. 두 방식 중 val F1 이 높은 쪽 채택.
평가 : 이 단계에서는 val 만 본다. test 는 6단계에서 1회 (docs/plan_week4-6.md 공통 원칙 1)

출력 : results/deeplog.json (val), artifacts/models/deeplog.pt (+ Detector 설정)

사용법:
    python -m logguard.models.deeplog                 # 기본 설정 (configs/config.yaml 의 deeplog)
    python -m logguard.models.deeplog --no-eos        # EOS 제거 비교 실험 → results/deeplog_noeos.json
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
from logguard.data.windows import EOS, NUM_SPECIAL, PAD, UNK, encode, load_vocab, make_lookup
from logguard.eval.metrics import prf
from logguard.models.dataset import build_windows, encode_sessions, unk_mask
from logguard.models.inference import DetectionResult, Detector
from logguard.models.scoring import normalize, save_result, save_test_scores, search_threshold, session_scores
from logguard.models.train_utils import get_device, sample_weights, seed_everything, train_weighted


class DeepLogModel(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, w) -> (B, C) logits
        h, _ = self.lstm(self.embed(x))
        return self.out(h[:, -1])


@torch.no_grad()
def window_scores(model: DeepLogModel, X: np.ndarray, y: np.ndarray, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """고유 윈도우별 (−log P(실제 다음 이벤트), 실제 이벤트의 예측 순위 0=1등)."""
    model.eval().cpu()
    nlls, ranks = [], []
    for i in range(0, len(X), batch_size):
        lp = torch.log_softmax(model(torch.as_tensor(X[i:i + batch_size])), dim=-1)
        yy = torch.as_tensor(y[i:i + batch_size])
        la = lp.gather(1, yy[:, None]).squeeze(1)
        nlls.append((-la).double().numpy())
        ranks.append((lp > la[:, None]).sum(1).numpy())
    return np.concatenate(nlls), np.concatenate(ranks)


# --------------------------------------------------------------------------- Detector (백엔드용)

class DeepLogDetector(Detector):
    name = "deeplog"

    def __init__(self, model: DeepLogModel, meta: dict):
        self.model = model.eval().cpu()
        self.meta = meta
        self.w = meta["window_size"]
        self.lut = make_lookup(meta["event_to_index"])
        self.index_to_event = {int(i): int(e) for e, i in meta["event_to_index"].items()}
        self.templates = {int(k): v for k, v in meta["templates"].items()}

    @classmethod
    def load(cls, path) -> "DeepLogDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ckpt["meta"]
        model = DeepLogModel(m["num_classes"], m["embed_dim"], m["hidden_size"], m["num_layers"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, m)

    def _label(self, idx: int) -> dict:
        if idx == EOS:
            return {"event_id": None, "template": "<EOS: 세션 종료>"}
        e = self.index_to_event.get(int(idx))
        return {"event_id": e, "template": self.templates.get(e) if e is not None else None}

    def _encode(self, events: Sequence[int | None], closed: bool) -> np.ndarray:
        ev = np.asarray([10**9 if e is None else int(e) for e in events], dtype=np.int64)
        return encode(ev, self.lut, add_eos=closed and self.meta["add_eos"])

    def _judge(self, events: Sequence[int | None], seq: np.ndarray, positions: np.ndarray) -> DetectionResult:
        unk_pos = [int(p) for p in positions if seq[p] == UNK]
        if unk_pos:
            unknown = [events[p] for p in unk_pos]
            return DetectionResult(1.0, True, "unknown_event", self.name,
                                   {"unknown_events": unknown,
                                    "templates": [self.templates.get(e) for e in unknown if e is not None]})
        padded = np.concatenate([np.full(self.w, PAD, dtype=np.int64), seq])
        X = np.stack([padded[p:p + self.w] for p in positions])
        y = seq[positions]
        with torch.no_grad():
            lp = torch.log_softmax(self.model(torch.as_tensor(X)), -1).double().numpy()
        la = lp[np.arange(len(y)), y]
        prob, pa = np.exp(lp), np.exp(la)
        rank = (lp > la[:, None]).sum(1)
        if self.meta["mode"] == "topk":
            s, thr = rank + 0.5, float(self.meta["k"])
        else:
            s, thr = -la, float(self.meta["threshold"])
        j = int(np.argmax(s))
        score = float(normalize(s[j], thr))
        top = np.argsort(-prob[j])[:max(3, int(self.meta.get("k", 3)))]
        details = {"position": int(positions[j]), "actual": {**self._label(y[j]), "prob": float(pa[j]), "rank": int(rank[j])},
                   "topk": [{**self._label(i), "prob": float(prob[j, i])} for i in top]}
        is_anom = bool(s[j] > thr)
        return DetectionResult(score, is_anom, "model" if is_anom else "normal", self.name, details)

    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        seq = self._encode(events, closed)
        return self._judge(list(events), seq, np.arange(len(seq)))

    def score_event(self, history: Sequence[int | None]) -> DetectionResult:
        """스트리밍: 마지막 이벤트만 판정 (앞 이벤트는 이전 호출에서 이미 판정됨)."""
        seq = self._encode(history, closed=False)
        return self._judge(list(history), seq, np.array([len(seq) - 1]))


# --------------------------------------------------------------------------- 학습·평가

def run(cfg: dict, add_eos: bool | None = None, tag: str = "deeplog", export: bool = True,
        evaluate_test: bool = False) -> dict:
    """evaluate_test=True 는 6단계 최종 평가에서만 사용. val 로 고른 판정 방식·임계치를 test 에 그대로 적용."""
    dc = dict(cfg["deeplog"])
    if add_eos is not None:
        dc["add_eos"] = add_eos
    seed_everything(cfg["seed"])
    proc = cfg["paths"]["processed_dir"]
    vocab = load_vocab(proc / "event_vocab.json")
    w, C = dc["window_size"], vocab["num_classes"]

    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    tr_seqs = encode_sessions(train, vocab, add_eos=dc["add_eos"])
    va_seqs = encode_sessions(val, vocab, add_eos=dc["add_eos"])

    t0 = time.time()
    tr = build_windows(tr_seqs, w, with_next=True)
    print(f"[{tag}] train windows {tr.n_total:,} -> unique {len(tr.X):,}  (eos={dc['add_eos']}, w={w})")

    model = DeepLogModel(C, dc["embed_dim"], dc["hidden_size"], dc["num_layers"])
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = lambda m, xb, yb: nn.functional.cross_entropy(m(xb), yb, reduction="none")
    history = train_weighted(model, (tr.X, tr.y), sample_weights(tr.weight, dc["weight_power"]), loss_fn,
                             epochs=dc["epochs"], batch_size=dc["batch_size"], lr=dc["lr"],
                             device=get_device(dc["device"]), seed=cfg["seed"],
                             lr_schedule=dc.get("lr_schedule", "none"), log_every=5)
    train_sec = time.time() - t0

    # val 채점 (고유 윈도우만 한 번씩)
    va = build_windows(va_seqs, w, with_next=True)
    nll, rank = window_scores(model, va.X, va.y)
    unk = unk_mask(va_seqs)
    y_val = val["label"].to_numpy()

    rank_s = session_scores(rank + 0.5, va, unk)                    # (a) top-k
    topk = pd.DataFrame([{"k": k, **prf(y_val, rank_s > k)} for k in dc["topk_grid"]]).sort_values("f1", ascending=False, kind="stable")
    nll_s = session_scores(nll, va, unk)                             # (b) percentile
    pct_table, pct_best = search_threshold(nll_s, y_val, cfg["threshold"]["percentile_grid"])
    best_k = topk.iloc[0].to_dict()

    if best_k["f1"] >= pct_best["f1"]:
        mode = {"mode": "topk", "k": int(best_k["k"])}
        chosen = best_k
    else:
        mode = {"mode": "percentile", "percentile": pct_best["percentile"], "threshold": pct_best["threshold"]}
        chosen = pct_best

    # UNK 단독 기여 (해석용)
    unk_only = prf(y_val, unk)

    result = {
        "stage": "val_only (test 는 6단계에서 1회 평가)",
        "config": dc, "num_params": n_params, "train_seconds": round(train_sec, 1),
        "train_windows": tr.n_total, "train_unique_windows": len(tr.X), "loss_history": history,
        "selected_on": "val_f1", "best_config": mode,
        "val": {k: chosen[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")},
        "val_topk_grid": topk.to_dict(orient="records"),
        "val_percentile_grid": pct_table.to_dict(orient="records"),
        "val_unk_only": unk_only,
    }

    if evaluate_test:
        test = pd.read_parquet(proc / "test.parquet")
        te_seqs = encode_sessions(test, vocab, add_eos=dc["add_eos"])
        te = build_windows(te_seqs, w, with_next=True)
        te_nll, te_rank = window_scores(model, te.X, te.y)
        te_unk = unk_mask(te_seqs)
        te_rank_s, te_nll_s = session_scores(te_rank + 0.5, te, te_unk), session_scores(te_nll, te, te_unk)
        pred = te_rank_s > mode["k"] if mode["mode"] == "topk" else te_nll_s > mode["threshold"]
        result["test"] = prf(test["label"], pred)
        result["stage"] = "final (val 로 선택한 설정을 test 에 1회 적용)"
        save_test_scores(cfg, tag, test["label"].to_numpy(), te_nll_s, mode)

    if export:
        tpl = pd.read_csv(cfg["paths"]["interim_dir"] / "hdfs_templates.csv")
        meta = {**dc, **mode, "num_classes": C, "event_to_index": vocab["event_to_index"],
                "templates": dict(zip(tpl["event_id"].astype(int).astype(str), tpl["template"]))}
        out = cfg["paths"]["artifacts_dir"] / "models" / "deeplog.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": meta}, out)
        det = DeepLogDetector.load(out)
        # 저장본 재현 검증 + 지연시간 (val 세션 2,000개, CPU 단일 호출)
        sample = val.sample(2000, random_state=cfg["seed"])
        t1 = time.perf_counter()
        pred = [det.score_session(s.tolist()).is_anomaly for s in sample["events"]]
        sess_ms = (time.perf_counter() - t1) / len(sample) * 1e3
        idx = sample.index.to_numpy()
        ref = (rank_s[idx] > mode["k"]) if mode["mode"] == "topk" else (nll_s[idx] > mode["threshold"])
        t2 = time.perf_counter()
        for s in sample["events"].head(500):
            det.score_event(s.tolist())
        event_ms = (time.perf_counter() - t2) / 500 * 1e3
        result["detector_check"] = {"agree_with_batch_eval": float(np.mean(np.asarray(pred) == ref)),
                                    "ms_per_session": round(sess_ms, 3), "ms_per_event": round(event_ms, 3)}
        result["artifact"] = str(out.relative_to(cfg["paths"]["artifacts_dir"].parent))

    save_result(cfg, tag, result)
    print(f"[{tag}] params {n_params:,}, train {train_sec:.0f}s, mode={mode}")
    print(f"[{tag}] VAL P={chosen['precision']:.4f} R={chosen['recall']:.4f} F1={chosen['f1']:.4f}")
    print(f"[{tag}] top-k  best: k={int(best_k['k'])} F1={best_k['f1']:.4f} | percentile best: p={pct_best['percentile']} F1={pct_best['f1']:.4f}")
    if "test" in result:
        t = result["test"]
        print(f"[{tag}] TEST P={t['precision']:.4f} R={t['recall']:.4f} F1={t['f1']:.4f}")
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
        run(cfg, add_eos=False, tag="deeplog_noeos", export=False)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
