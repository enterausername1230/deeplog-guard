"""공통 학습 유틸: 시드 고정, 장치 선택, 가중(고유 윈도우 × 등장 횟수) 학습 루프.

DeepLog 와 Autoencoder 가 똑같은 학습 절차를 쓰도록 공통 부분을 이 파일에 모았다.
모델마다 다른 것은 "손실을 어떻게 계산하느냐(loss_fn)" 뿐이고, 나머지(배치 나누기, 최적화, 기록)는 같다.

딥러닝 학습 한 바퀴(에폭)가 하는 일
    1) 학습 데이터를 무작위로 섞는다
    2) batch_size 개씩 잘라서(미니배치) 모델에 넣고 손실(틀린 정도)을 계산한다
    3) 손실을 줄이는 방향으로 모델 가중치를 조금 고친다 (역전파 + Adam 최적화)
    4) 데이터를 다 돌면 한 에폭 끝. 이를 epochs 번 반복한다.
"""
from __future__ import annotations

import random
import time
from typing import Callable     # "함수" 타입 표시용 (loss_fn 처럼 함수를 인자로 받을 때)

import numpy as np
import torch                    # PyTorch: 딥러닝 프레임워크


def seed_everything(seed: int) -> None:
    """모든 난수 생성기의 시드를 고정한다 → 같은 시드면 몇 번을 실행해도 같은 결과가 나온다 (재현성).

    파이썬 기본 random, numpy, PyTorch 가 각자 난수를 쓰므로 세 군데를 모두 고정해야 한다.
    (모델 초기 가중치, 데이터 섞는 순서 등이 난수로 정해진다)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(pref: str = "auto") -> torch.device:
    """계산 장치를 고른다.

    Args:
        pref: "cpu" / "cuda" / "mps" 를 직접 지정하거나, "auto" 면 쓸 수 있는 가장 빠른 장치를 자동 선택
              cuda = NVIDIA GPU (Colab, 학교 GPU 서버), mps = Apple 실리콘 맥의 GPU, cpu = 일반 CPU
    참고: 우리 모델은 매우 작아서(파라미터 수만~수십만 개) GPU 로 옮기는 비용이 더 커서 CPU 가 오히려 빠를 수 있다.
          그래서 config.yaml 에서 device: cpu 로 두었다.
    """
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_weights(counts: np.ndarray, power: float = 1.0) -> np.ndarray:
    """power=1: 원본 데이터 분포 그대로 (전체 윈도우 학습과 동일), power=0: 고유 윈도우 균등.

    고유 윈도우의 등장 횟수(counts)를 학습 가중치로 바꾼다.
        power = 1   : 33만 번 나온 윈도우는 1번 나온 윈도우보다 33만 배 중요 (원본 데이터와 같은 비중)
        power = 0.5 : 제곱근 비중 (흔한 윈도우의 영향을 줄임)
        power = 0   : 모든 고유 윈도우를 똑같이 취급
    마지막에 평균이 1 이 되도록 나눈다 → 가중치 크기와 상관없이 학습률(lr)의 의미가 일정하게 유지된다.
    """
    w = np.asarray(counts, dtype=np.float64) ** power
    return w / w.mean()


def train_weighted(
    model: torch.nn.Module,
    tensors: tuple[np.ndarray, ...],
    weights: np.ndarray,
    loss_fn: Callable[..., torch.Tensor],
    *,                                  # 이 뒤의 인자는 반드시 이름=값 형태로 넘겨야 한다 (실수 방지)
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

    Args:
        model      : 학습할 PyTorch 모델
        tensors    : 학습 데이터 배열들. DeepLog 는 (X, y), Autoencoder 는 (X,) — 같은 순서로 배치가 잘린다
        weights    : 샘플별 가중치 (sample_weights 결과)
        loss_fn    : (모델, 배치데이터들…) → 샘플별 손실 텐서 (평균 내지 않은 상태, 모양 (B,))
        epochs     : 전체 데이터를 몇 바퀴 돌지
        batch_size : 한 번에 몇 개씩 학습할지
        lr         : 학습률. 한 번에 가중치를 얼마나 크게 고칠지
        device     : 계산 장치
        seed       : 데이터 섞는 순서를 고정하기 위한 시드
        log_every  : 몇 에폭마다 손실을 출력할지 (0 이면 출력 안 함)
        lr_schedule: "none" = 학습률 고정, "cosine" = 코사인 곡선 모양으로 0 까지 서서히 감소
                     (Autoencoder 는 고정 학습률에서 후반에 손실이 갑자기 튀는 현상이 있어 cosine 사용)
    Returns:
        에폭별 평균 손실 리스트 (학습이 잘 되면 점점 줄어든다. 결과 json 의 loss_history)
    """
    model.to(device)                                            # 모델 가중치를 계산 장치로 옮긴다
    # Adam: 가장 널리 쓰이는 최적화 알고리즘. 가중치마다 학습 속도를 자동 조절한다
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # CosineAnnealingLR: 학습률을 T_max(=epochs) 에폭에 걸쳐 lr → 0 으로 코사인 모양으로 줄인다
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if lr_schedule == "cosine" else None
    data = [torch.as_tensor(t) for t in tensors]               # numpy 배열 → PyTorch 텐서
    w_all = torch.as_tensor(weights, dtype=torch.float32)
    n = len(w_all)                                              # 학습 샘플(고유 윈도우) 수
    # 데이터 섞기 전용 난수 생성기 (시드 고정 → 매번 같은 순서로 섞임)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(1, epochs + 1):
        model.train()                                           # 학습 모드로 전환 (일부 층의 동작이 달라짐)
        t0, total = time.time(), 0.0
        # torch.randperm(n): 0~n-1 을 무작위로 섞은 순서 / .split(batch_size): batch_size 개씩 자르기
        for idx in torch.randperm(n, generator=gen).split(batch_size):
            batch = [d[idx].to(device) for d in data]           # 이번 배치에 해당하는 데이터만 꺼내기
            w = w_all[idx].to(device)
            # 가중 평균 손실: (샘플별 손실 × 가중치)의 합 / 배치 크기
            #   가중치 평균이 1 이므로, 데이터 전체로 보면 원본 윈도우 분포로 학습한 것과 같은 목표가 된다
            loss = (loss_fn(model, *batch) * w).sum() / len(idx)
            opt.zero_grad()      # 이전 배치에서 쌓인 기울기(gradient)를 지운다 (안 지우면 계속 누적됨)
            loss.backward()      # 역전파: 손실을 줄이려면 각 가중치를 어느 방향으로 바꿔야 하는지 계산
            opt.step()           # 계산된 방향으로 가중치를 실제로 조금 수정
            total += loss.item() * len(idx)                     # 에폭 평균 손실 계산용 누적
        if sched:
            sched.step()         # 에폭이 끝날 때마다 학습률을 한 단계 줄인다 (cosine 일 때만)
        history.append(total / n)
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep:>3}/{epochs}  loss {history[-1]:.5f}  ({time.time() - t0:.1f}s)")
    return history
