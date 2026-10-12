"""공통 학습 유틸: 시드 고정, 장치 선택, 가중(고유 윈도우 × 등장 횟수) 학습 루프."""
from __future__ import annotations

import random
import time
from typing import Callable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_weights(counts: np.ndarray, power: float = 1.0) -> np.ndarray:
    """power=1: 원본 데이터 분포 그대로 (전체 윈도우 학습과 동일), power=0: 고유 윈도우 균등."""
    w = np.asarray(counts, dtype=np.float64) ** power
    return w / w.mean()


def train_weighted(
    model: torch.nn.Module,
    tensors: tuple[np.ndarray, ...],
    weights: np.ndarray,
    loss_fn: Callable[..., torch.Tensor],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
    log_every: int = 1,
    lr_schedule: str = "none",
) -> list[float]:
    """loss_fn(model, *batch) -> 샘플별 손실 (B,). 가중 평균으로 학습하고 에폭별 평균 손실을 반환.

    lr_schedule="cosine" 이면 에폭마다 학습률을 lr → 0 으로 줄인다 (후반 loss 튐 방지).
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if lr_schedule == "cosine" else None
    data = [torch.as_tensor(t) for t in tensors]
    w_all = torch.as_tensor(weights, dtype=torch.float32)
    n = len(w_all)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for idx in torch.randperm(n, generator=gen).split(batch_size):
            batch = [d[idx].to(device) for d in data]
            w = w_all[idx].to(device)
            loss = (loss_fn(model, *batch) * w).sum() / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if sched:
            sched.step()
        history.append(total / n)
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep:>3}/{epochs}  loss {history[-1]:.5f}  ({time.time() - t0:.1f}s)")
    return history
