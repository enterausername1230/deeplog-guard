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

────────────────────────────────────────────────────────────────────────────────────────────
DeepLog 를 한 문장으로: "다음에 올 로그를 예측해 보고, 실제가 예상 밖이면 이상"
────────────────────────────────────────────────────────────────────────────────────────────
휴대폰 자판의 "다음 단어 추천" 과 같은 원리다.
    정상 블록에서는 "블록 할당 → 데이터 수신 → 수신 완료 → …" 순서가 거의 정해져 있다.
    모델은 정상 세션만 보고 "이 다음엔 보통 무엇이 오는가" 를 배운다.
    판정 때 실제로 온 이벤트의 예측 확률 P 가 아주 낮으면(= 모델이 전혀 예상 못 함) 이상이라고 본다.

예시 (w=3)
    직전 이벤트: [할당, 수신, 수신]  →  모델 예측: 수신완료 60%, 수신 30%, 삭제 0.01%, …
    실제로 "삭제" 가 왔다 → P = 0.0001 → 점수 −log(0.0001) = 9.2 → 임계치(약 6.9)보다 크므로 이상

모델 구조
    이벤트 번호 → [Embedding: 번호를 16차원 벡터로] → [LSTM 2층: 순서 정보를 요약] → [Linear: 22개 클래스 점수]
    LSTM(Long Short-Term Memory): 시퀀스를 앞에서부터 읽으며 "지금까지의 흐름" 을 기억하는 순환 신경망

최종 결과 (6단계 test): F1 0.926 (Autoencoder 0.977 에 비해 낮아 최종 모델로는 채택되지 않음)
"""
from __future__ import annotations

import argparse
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn            # nn: 신경망 층(Embedding, LSTM, Linear 등) 모음

from logguard.config import load_config
from logguard.data.windows import EOS, NUM_SPECIAL, PAD, UNK, encode, load_vocab, make_lookup
from logguard.eval.metrics import prf
from logguard.models.dataset import build_windows, encode_sessions, unk_mask
from logguard.models.inference import DetectionResult, Detector
from logguard.models.scoring import normalize, save_result, save_test_scores, search_threshold, session_scores
from logguard.models.train_utils import get_device, sample_weights, seed_everything, train_weighted


class DeepLogModel(nn.Module):
    """직전 w 개 이벤트 → 다음 이벤트가 각 클래스일 점수(logits) 를 내는 LSTM 모델.

    nn.Module 을 상속한 PyTorch 모델은 __init__ 에서 층을 만들고, forward 에서 계산 순서를 정한다.
    """

    def __init__(self, num_classes: int, embed_dim: int, hidden_size: int, num_layers: int):
        """
        Args:
            num_classes: 클래스 수 = 어휘 크기 (특수 토큰 3 + 이벤트 19 = 22)
            embed_dim  : 이벤트 번호를 표현할 벡터 길이 (16)
            hidden_size: LSTM 이 기억하는 상태 벡터 길이 (최종 128, 클수록 표현력↑ 계산량↑)
            num_layers : LSTM 층 수 (2, DeepLog 논문 설정)
        """
        super().__init__()        # 부모 클래스(nn.Module) 초기화 — PyTorch 모델에서 필수
        # Embedding: 번호 하나를 학습 가능한 벡터로 바꾸는 표(22행 × 16열).
        #   padding_idx=PAD: PAD(0) 의 벡터는 항상 0 이고 학습되지 않는다 (빈칸은 의미가 없으므로)
        self.embed = nn.Embedding(num_classes, embed_dim, padding_idx=PAD)
        # LSTM: (배치, 시간, 특징) 순서의 입력을 받도록 batch_first=True
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        # 마지막 상태(128차원) → 22개 클래스 점수
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, w) -> (B, C) logits
        """x: (배치 B, 윈도우 길이 w) 의 이벤트 번호 → (B, 클래스 수 C) 의 점수(logits, 소프트맥스 전 값)."""
        # self.embed(x): (B, w) → (B, w, 16)
        # self.lstm(...): 각 시점의 출력 h (B, w, 128) 와 마지막 상태들을 돌려준다 (상태는 _ 로 버림)
        h, _ = self.lstm(self.embed(x))
        # h[:, -1]: 마지막 시점(윈도우의 마지막 이벤트까지 다 읽은 뒤)의 출력만 사용 → (B, 128) → (B, 22)
        return self.out(h[:, -1])


@torch.no_grad()      # 이 함수 안에서는 기울기 계산을 끈다 (평가만 하므로 메모리·속도 절약)
def window_scores(model: DeepLogModel, X: np.ndarray, y: np.ndarray, batch_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """고유 윈도우별 (−log P(실제 다음 이벤트), 실제 이벤트의 예측 순위 0=1등).

    Args:
        X: (n, w) 윈도우들, y: (n,) 각 윈도우 다음에 실제로 온 이벤트
    Returns:
        nll  : 이상 점수 −log P. 실제 이벤트를 잘 예측했으면(P≈1) 0 에 가깝고, 예상 밖이면 크다.
        ranks: 실제 이벤트가 예측 확률 순위로 몇 등이었는지 (0 = 1등). top-k 판정에 사용
    """
    model.eval().cpu()       # 평가 모드 + CPU 로 이동
    nlls, ranks = [], []
    # 수만~수십만 개 윈도우를 8192 개씩 나눠서 계산 (한 번에 하면 메모리가 부족할 수 있음)
    for i in range(0, len(X), batch_size):
        # log_softmax: 점수(logits) → 로그 확률. 확률을 직접 쓰면 아주 작은 값이 0 으로 뭉개지므로 로그로 다룬다
        lp = torch.log_softmax(model(torch.as_tensor(X[i:i + batch_size])), dim=-1)    # (배치, 22)
        yy = torch.as_tensor(y[i:i + batch_size])
        # gather: 각 행에서 "실제 이벤트" 칸의 로그 확률만 뽑기 → (배치,)
        la = lp.gather(1, yy[:, None]).squeeze(1)
        nlls.append((-la).double().numpy())                   # 부호를 바꿔 −log P (클수록 이상)
        # 순위 = 실제 이벤트보다 확률이 높은 클래스의 개수 (0 개면 1등)
        ranks.append((lp > la[:, None]).sum(1).numpy())
    return np.concatenate(nlls), np.concatenate(ranks)


# --------------------------------------------------------------------------- Detector (백엔드용)

class DeepLogDetector(Detector):
    """학습된 DeepLog 모델을 백엔드용 Detector 인터페이스로 감싼 것. load() 로 .pt 파일에서 불러온다."""
    name = "deeplog"

    def __init__(self, model: DeepLogModel, meta: dict):
        """
        Args:
            model: 가중치가 들어 있는 DeepLogModel
            meta : 판정에 필요한 정보 (윈도우 크기, 판정 방식·임계치, 어휘, 템플릿 문장 등). .pt 파일에 함께 저장됨
        """
        self.model = model.eval().cpu()
        self.meta = meta
        self.w = meta["window_size"]
        self.lut = make_lookup(meta["event_to_index"])                 # Event ID → 모델 번호 변환표
        # 모델 번호 → Event ID (판정 근거를 사람이 읽을 수 있게 되돌릴 때 사용)
        self.index_to_event = {int(i): int(e) for e, i in meta["event_to_index"].items()}
        self.templates = {int(k): v for k, v in meta["templates"].items()}

    @classmethod
    def load(cls, path) -> "DeepLogDetector":
        """artifacts/models/deeplog.pt 를 불러와 Detector 를 만든다.

        .pt 파일 내용: {"state_dict": 모델 가중치, "meta": 설정·임계치·어휘}
        weights_only=False: meta 에 일반 파이썬 dict 가 들어 있어 이 옵션이 필요하다 (우리가 만든 파일만 불러올 것)
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ckpt["meta"]
        # 저장 때와 같은 크기로 빈 모델을 만들고 → 저장된 가중치를 채워 넣는다
        model = DeepLogModel(m["num_classes"], m["embed_dim"], m["hidden_size"], m["num_layers"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, m)

    def _label(self, idx: int) -> dict:
        """모델 번호 → {"event_id", "template"} (판정 근거 표시용)."""
        if idx == EOS:
            return {"event_id": None, "template": "<EOS: 세션 종료>"}
        e = self.index_to_event.get(int(idx))
        return {"event_id": e, "template": self.templates.get(e) if e is not None else None}

    def _encode(self, events: Sequence[int | None], closed: bool) -> np.ndarray:
        """Event ID 목록 → 모델 번호 배열. 세션이 끝났고(closed) 모델이 EOS 로 학습됐으면 끝에 EOS 추가."""
        # None(온라인 파서가 모르는 로그)은 아주 큰 수 10**9 로 바꿔 두면 encode() 가 UNK 로 처리한다
        ev = np.asarray([10**9 if e is None else int(e) for e in events], dtype=np.int64)
        return encode(ev, self.lut, add_eos=closed and self.meta["add_eos"])

    def _judge(self, events: Sequence[int | None], seq: np.ndarray, positions: np.ndarray) -> DetectionResult:
        """seq 의 positions 위치에 있는 이벤트들을 판정하고, 가장 이상한 위치를 기준으로 결과를 만든다.

        Args:
            events   : 원래 Event ID 목록 (모르는 이벤트를 결과에 그대로 보여 주기 위해)
            seq      : 모델 번호 배열 (_encode 결과)
            positions: 판정할 위치들. 세션 판정이면 전체, 스트리밍이면 마지막 하나
        """
        # 1) 판정 대상에 UNK 가 있으면 모델 계산 없이 즉시 이상
        unk_pos = [int(p) for p in positions if seq[p] == UNK]
        if unk_pos:
            unknown = [events[p] for p in unk_pos]
            return DetectionResult(1.0, True, "unknown_event", self.name,
                                   {"unknown_events": unknown,
                                    "templates": [self.templates.get(e) for e in unknown if e is not None]})
        # 2) 각 위치의 직전 w 개를 윈도우로 만든다 (앞에 PAD w 개를 붙여 첫 이벤트도 판정 가능하게)
        #    padded[p : p+w] = 원래 seq 에서 위치 p 바로 앞의 w 개
        padded = np.concatenate([np.full(self.w, PAD, dtype=np.int64), seq])
        X = np.stack([padded[p:p + self.w] for p in positions])
        y = seq[positions]                                   # 각 위치의 실제 이벤트
        with torch.no_grad():
            lp = torch.log_softmax(self.model(torch.as_tensor(X)), -1).double().numpy()
        la = lp[np.arange(len(y)), y]                        # 실제 이벤트의 로그 확률
        prob, pa = np.exp(lp), np.exp(la)                    # 표시용 확률 (0~1)
        rank = (lp > la[:, None]).sum(1)                     # 실제 이벤트의 예측 순위 (0 = 1등)
        # 3) 판정 방식(meta["mode"])에 따라 점수와 임계치를 정한다 (학습 때 val 로 고른 방식)
        if self.meta["mode"] == "topk":
            # 순위 + 0.5 > k  ⇔  순위 ≥ k  (0.5 를 더해 경계에서 정규화 점수가 정확히 0.5 가 되지 않게)
            s, thr = rank + 0.5, float(self.meta["k"])
        else:
            s, thr = -la, float(self.meta["threshold"])      # −log P 와 백분위수 임계치 (최종 설정)
        # 4) 가장 이상한(점수가 가장 큰) 위치 하나를 대표로 결과를 만든다 (= 세션 점수는 최댓값)
        j = int(np.argmax(s))
        score = float(normalize(s[j], thr))                  # 0~1, 임계치 = 0.5
        # 모델이 그 위치에서 "대신 예상한" 상위 후보들 (최소 3개) → 대시보드에 "실제 vs 예상" 으로 표시
        top = np.argsort(-prob[j])[:max(3, int(self.meta.get("k", 3)))]
        details = {"position": int(positions[j]), "actual": {**self._label(y[j]), "prob": float(pa[j]), "rank": int(rank[j])},
                   "topk": [{**self._label(i), "prob": float(prob[j, i])} for i in top]}
        is_anom = bool(s[j] > thr)
        return DetectionResult(score, is_anom, "model" if is_anom else "normal", self.name, details)

    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        """세션 전체의 모든 위치(EOS 포함)를 판정한다."""
        seq = self._encode(events, closed)
        return self._judge(list(events), seq, np.arange(len(seq)))

    def score_event(self, history: Sequence[int | None]) -> DetectionResult:
        """스트리밍: 마지막 이벤트만 판정 (앞 이벤트는 이전 호출에서 이미 판정됨)."""
        seq = self._encode(history, closed=False)
        return self._judge(list(history), seq, np.array([len(seq) - 1]))


# --------------------------------------------------------------------------- 학습·평가

def run(cfg: dict, add_eos: bool | None = None, tag: str = "deeplog", export: bool = True,
        evaluate_test: bool = False) -> dict:
    """evaluate_test=True 는 6단계 최종 평가에서만 사용. val 로 고른 판정 방식·임계치를 test 에 그대로 적용.

    전체 순서: 데이터 준비 → 학습 → val 채점 → 판정 방식·임계치 선택 → (선택) test 평가 → (선택) 모델 저장

    Args:
        cfg          : 설정 (cfg["deeplog"] 의 하이퍼파라미터 사용)
        add_eos      : None 이면 설정값, False 를 주면 EOS 없이 학습 (비교 실험용)
        tag          : 결과 파일 이름 (results/<tag>.json)
        export       : True 면 artifacts/models/deeplog.pt 로 모델 저장 + Detector 재현·속도 검사
        evaluate_test: True 면 test 까지 평가 (6단계 final_eval 에서만 True)
    Returns:
        결과 dict (val 지표, 선택된 설정, 손실 기록 등)
    """
    dc = dict(cfg["deeplog"])            # 원본 설정을 건드리지 않도록 복사본 사용
    if add_eos is not None:
        dc["add_eos"] = add_eos
    seed_everything(cfg["seed"])         # 시드 고정 → 같은 설정이면 언제나 같은 결과
    proc = cfg["paths"]["processed_dir"]
    vocab = load_vocab(proc / "event_vocab.json")
    w, C = dc["window_size"], vocab["num_classes"]

    # ── 데이터 준비 ───────────────────────────────────────────────────────────────────────
    train = pd.read_parquet(proc / "train.parquet")
    val = pd.read_parquet(proc / "val.parquet")
    tr_seqs = encode_sessions(train, vocab, add_eos=dc["add_eos"])
    va_seqs = encode_sessions(val, vocab, add_eos=dc["add_eos"])

    t0 = time.time()
    tr = build_windows(tr_seqs, w, with_next=True)          # 학습 윈도우 (중복 압축)
    print(f"[{tag}] train windows {tr.n_total:,} -> unique {len(tr.X):,}  (eos={dc['add_eos']}, w={w})")

    # ── 학습 ─────────────────────────────────────────────────────────────────────────────
    model = DeepLogModel(C, dc["embed_dim"], dc["hidden_size"], dc["num_layers"])
    n_params = sum(p.numel() for p in model.parameters())   # 학습 가능한 숫자(파라미터) 총 개수
    # 손실: 다음 이벤트 분류의 Cross-Entropy. reduction="none" 으로 샘플별 손실을 받아 가중 평균은 train_weighted 가 한다
    #   lambda 는 이름 없는 짧은 함수: (모델, 입력배치, 정답배치) → 샘플별 손실
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

    # ── 판정 방식 두 가지를 val 로 비교 ──────────────────────────────────────────────────────
    # (a) top-k: 세션 점수 = (가장 나쁜 순위 + 0.5). k 후보별로 F1 계산
    rank_s = session_scores(rank + 0.5, va, unk)                    # (a) top-k
    topk = pd.DataFrame([{"k": k, **prf(y_val, rank_s > k)} for k in dc["topk_grid"]]).sort_values("f1", ascending=False, kind="stable")
    # (b) 백분위수: 세션 점수 = 가장 큰 −log P. 백분위수 후보별로 F1 계산
    nll_s = session_scores(nll, va, unk)                             # (b) percentile
    pct_table, pct_best = search_threshold(nll_s, y_val, cfg["threshold"]["percentile_grid"])
    best_k = topk.iloc[0].to_dict()

    # val F1 이 더 높은 방식을 채택 (최종 설정에서는 백분위수 방식이 선택됨)
    if best_k["f1"] >= pct_best["f1"]:
        mode = {"mode": "topk", "k": int(best_k["k"])}
        chosen = best_k
    else:
        mode = {"mode": "percentile", "percentile": pct_best["percentile"], "threshold": pct_best["threshold"]}
        chosen = pct_best

    # UNK 단독 기여 (해석용) — "UNK 규칙 하나만으로 val 에서 얼마나 잡히는지"
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

    # ── (6단계 전용) test 평가: val 로 이미 정한 mode 를 그대로 적용. test 를 보고 바꾸는 것은 없음 ──
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
        save_test_scores(cfg, tag, test["label"].to_numpy(), te_nll_s, mode)     # 그래프용 점수 저장

    # ── 모델 저장 + 저장본 검증 ─────────────────────────────────────────────────────────────
    if export:
        tpl = pd.read_csv(cfg["paths"]["interim_dir"] / "hdfs_templates.csv")
        # meta: Detector 가 판정에 필요한 모든 정보 (설정 + 선택된 판정 방식 + 어휘 + 템플릿 문장)
        meta = {**dc, **mode, "num_classes": C, "event_to_index": vocab["event_to_index"],
                "templates": dict(zip(tpl["event_id"].astype(int).astype(str), tpl["template"]))}
        out = cfg["paths"]["artifacts_dir"] / "models" / "deeplog.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": meta}, out)
        det = DeepLogDetector.load(out)          # 방금 저장한 파일을 다시 불러와서
        # 저장본 재현 검증 + 지연시간 (val 세션 2,000개, CPU 단일 호출)
        #   ① 저장본 Detector 의 판정이 위의 일괄 계산 판정과 100% 같은지 (agree_with_batch_eval = 1.0 이어야 함)
        #   ② 세션 1개 / 이벤트 1개 판정에 걸리는 시간
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

    # ── 결과 저장·출력 ─────────────────────────────────────────────────────────────────────
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
    """명령줄 진입점 (`make deeplog`). --no-eos 를 주면 EOS 없는 비교 실험만 하고 모델은 저장하지 않는다."""
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
