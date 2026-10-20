"""추론 인터페이스 (백엔드 연동용). 모든 모델은 Detector 를 구현한다.

입력은 원본 Drain Event ID 시퀀스 (OnlineParser.parse_line(...).event_id 를 블록별로 모은 것).
  score_event(history)              : 스트리밍 — history 의 마지막 이벤트까지 보고 판정 (로그 1줄마다 호출)
  score_session(events, closed=True): 세션 전체 판정. closed=True 면 세션 종료(EOS)까지 고려

반환 DetectionResult.score 는 0~1 로 정규화되어 있고 임계치는 항상 0.5 (대시보드 공통 기준선).

현재 구현
  statistical_frequency : 통계 빈도 베이스라인 (모델 완성 전 연동용). 학습 정상 세션의 이벤트별 횟수
                          p 백분위수를 넘거나, 정상에 없던 이벤트가 나오면 이상.
  deeplog               : LSTM 다음 이벤트 예측 (logguard.models.deeplog), artifacts/models/deeplog.pt
  autoencoder           : 윈도우 재구성 오차 (logguard.models.autoencoder), artifacts/models/autoencoder.pt

사용 예 (FastAPI):
    detector = load_detector("statistical_frequency")
    result = detector.score_event(block_events[blk])   # block_events: dict[str, list[int]]
    if result.is_anomaly: ...

내보내기/검증: python -m logguard.models.inference --export statistical_frequency

────────────────────────────────────────────────────────────────────────────────────────────
"인터페이스" 가 왜 필요한가
────────────────────────────────────────────────────────────────────────────────────────────
백엔드(안명근)는 "어떤 모델이 안에서 어떻게 계산하는지" 몰라도 되어야 한다.
그래서 모든 탐지기가 똑같은 모양의 함수(score_event, score_session)와 똑같은 모양의 결과(DetectionResult)를
갖도록 약속했다. 이 약속(추상 클래스 Detector) 덕분에:
    - 모델을 바꿀 때 백엔드 코드는 load_detector("이름") 의 이름만 바꾸면 된다.
    - 모델이 완성되기 전에도 통계 빈도 탐지기로 백엔드·대시보드 개발을 먼저 진행할 수 있었다.

스트리밍(score_event) 사용 흐름
    블록 A 의 로그가 1줄 들어옴 → 블록 A 이력 [5] → score_event([5])
    블록 A 의 로그가 또 들어옴   → 블록 A 이력 [5, 7] → score_event([5, 7])  (마지막 이벤트 7 을 판정)
    …
    블록 A 가 끝났다고 판단되면(타임아웃) → score_session(전체 이력, closed=True) 로 "너무 일찍 끝났는지" 까지 검사
"""
from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod                   # 추상 클래스: "반드시 구현해야 할 함수" 를 강제하는 도구
from dataclasses import asdict, dataclass, field
from typing import Sequence                           # 리스트·튜플 등 "순서 있는 묶음" 타입 표시

import numpy as np
import pandas as pd

from logguard.config import load_config
from logguard.eval.metrics import prf

# 모든 모델 공통 임계치(정규화 점수 기준). scoring.normalize() 가 임계치를 항상 0.5 에 오도록 변환한다.
THRESHOLD = 0.5
# 6단계 최종 평가로 채택한 모델 (test F1 0.977, results/final_test.json). 백엔드는 이 이름을 사용
#   백엔드가 load_detector(RECOMMENDED_DETECTOR) 로 써 두면, 나중에 최종 모델이 바뀌어도
#   이 한 줄만 고치면 되고 백엔드 코드는 그대로 둘 수 있다.
RECOMMENDED_DETECTOR = "autoencoder"


@dataclass
class DetectionResult:
    """모든 탐지기가 돌려주는 판정 결과. to_dict() 로 바꾸면 JSON 으로 바로 응답·저장할 수 있다."""
    score: float                      # 0~1, 0.5 보다 크면 이상 (정확히 0.5 는 정상). 대시보드 그래프 값
    is_anomaly: bool                  # 최종 판정: True = 이상 (Slack 알림 여부 결정에 사용)
    reason: str                       # "unknown_event" | "model" | "normal"
                                      #   unknown_event: 학습 때 못 본 이벤트 등장 (모델 계산 없이 즉시 이상)
                                      #   model        : 모델이 이상이라고 판정
                                      #   normal       : 정상
    model: str                        # 판정한 탐지기 이름 ("autoencoder" 등)
    details: dict = field(default_factory=dict)   # 판정 근거 (탐지기마다 내용이 다름, 대시보드에 표시)

    def to_dict(self) -> dict:
        """dataclass → 일반 dict (FastAPI 응답, SQLite 저장, json.dumps 용)."""
        return asdict(self)


class Detector(ABC):
    """모든 탐지기의 공통 틀(추상 클래스). 이 클래스를 상속하는 탐지기는 score_session() 을 반드시 구현해야 한다.

    ABC 를 상속하고 @abstractmethod 를 붙이면, score_session 을 구현하지 않은 클래스는 객체를 만들 수 없다.
    → 새 모델을 추가할 때 실수로 필수 함수를 빼먹는 일을 막아 준다.
    """
    name: str        # 탐지기 이름 (하위 클래스에서 정함)

    @abstractmethod
    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        """블록(세션)의 이벤트 목록 전체를 보고 판정한다.

        Args:
            events: Drain Event ID 목록. 온라인 파서가 모르는 로그면 None 이 섞여 있을 수 있다 (→ unknown_event).
            closed: True = 세션이 끝났다 (EOS 를 붙여 "이렇게 끝나도 정상인가" 까지 검사)
                    False = 아직 진행 중 (EOS 검사 안 함)
        """
        ...

    def score_event(self, history: Sequence[int | None]) -> DetectionResult:
        """스트리밍용 기본 구현: 지금까지의 이력 전체를 "아직 안 끝난 세션" 으로 판정한다.

        DeepLog·Autoencoder 는 이 함수를 더 빠르게 다시 구현(override)해서 "마지막 이벤트만" 판정한다.
        """
        return self.score_session(history, closed=False)

    def score_close(self, history: Sequence[int | None]) -> DetectionResult:
        """세션이 끝났다고 판단된 순간(타임아웃·재생 종료) "이렇게 끝나도 정상인가" 만 판정한다.

        왜 score_session(closed=True) 와 따로 두나?
            score_session 은 세션 "전체" 이벤트를 받아야 정확하다. 그런데 백엔드는 블록마다
            "최근 window_size 개" 만 DB 에서 꺼내 온다. 앞부분이 잘린 이력을 score_session 에 넣으면
            잘린 앞부분 자리에 PAD 가 채워져, 학습 때 본 적 없는 모양이 되어 오탐이 난다 (10/20 통합 시험에서 확인).
            그래서 DeepLog·Autoencoder 는 이 함수를 다시 구현해 "EOS 위치 하나만" 판정한다.

        기본 구현(통계 빈도 등): 세션 종료 개념이 없는 탐지기는 그냥 score_session(closed=True) 을 쓴다.

        Args:
            history: 그 블록의 최근 이벤트들 (오래된 것 → 최신 순). 최소 window_size 개면 충분하다.
        """
        return self.score_session(history, closed=True)


class StatisticalFrequencyDetector(Detector):
    """통계 빈도 베이스라인을 Detector 형식으로 감싼 것 (baseline/rule_based.py 의 R3 규칙과 같은 판정).

    판정 규칙: 블록 안에서 어떤 이벤트의 등장 횟수가 "정상 학습 세션 기준 임계치" 를 넘으면 이상.
              학습 때 본 적 없는 이벤트가 하나라도 나오면 즉시 이상.
    """
    name = "statistical_frequency"

    def __init__(self, thresholds: dict[int, float], percentile: float, templates: dict[int, str] | None = None):
        """
        Args:
            thresholds: {Event ID: 허용 최대 횟수}. 예: {1: 3.0, 5: 3.0, 13: 0.0}
                        (임계치 0 = 정상 학습에서 거의 안 나오던 이벤트 → 한 번만 나와도 이상)
            percentile: 임계치를 만들 때 쓴 백분위수 (기록용, 예: 99.95)
            templates : {Event ID: 템플릿 문장} — 판정 근거를 사람이 읽을 수 있게 표시하는 데 사용
        """
        self.thresholds = {int(k): float(v) for k, v in thresholds.items()}
        self.percentile = percentile
        self.templates = templates or {}       # None 이면 빈 dict

    @classmethod
    def fit(cls, train_normal: pd.DataFrame, percentile: float, templates: dict[int, str]) -> "StatisticalFrequencyDetector":
        """정상 학습 세션에서 이벤트별 횟수의 p 백분위수를 계산해 탐지기를 만든다."""
        # 함수 안에서 import: baseline ↔ models 사이의 순환 import 를 피하고, 이 함수를 쓸 때만 불러온다
        from logguard.baseline.rule_based import count_matrix
        n_events = max(templates) + 1          # Event ID 최댓값 + 1 = 행렬 열 개수
        # 이벤트(열)별로 정상 세션 횟수 분포의 p 백분위수 → 임계치
        th = np.percentile(count_matrix(train_normal, n_events), percentile, axis=0)
        # 정상 학습에 실제로 나온 이벤트만 임계치 목록에 넣는다 (나머지는 "모르는 이벤트" 로 취급)
        seen = {int(e) for seq in train_normal["events"] for e in np.unique(seq)}
        return cls({e: th[e] for e in sorted(seen)}, percentile, templates)

    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        """블록의 이벤트별 횟수를 세고, 임계치 대비 가장 많이 초과한 이벤트를 기준으로 판정한다.

        closed 인자는 이 탐지기에서는 쓰지 않는다 (횟수는 세션이 끝났는지와 상관없음). 인터페이스 통일용.
        """
        # 이벤트별 등장 횟수. dropna=False: None(모르는 로그)도 개수에 포함시킨다
        counts = pd.Series([e for e in events]).value_counts(dropna=False)
        # worst: 임계치 대비 비율이 가장 큰 이벤트, ratio: 그 비율, unknown: 모르는 이벤트 목록
        worst, ratio, unknown = None, 0.0, []
        for e, c in counts.items():
            # 모르는 이벤트 판별: None 이거나, (pandas 가 None 을 NaN 으로 바꿨을 경우) NaN 이거나, 임계치 목록에 없음
            if e is None or (isinstance(e, float) and np.isnan(e)) or int(e) not in self.thresholds:
                unknown.append(None if e is None or (isinstance(e, float) and np.isnan(e)) else int(e))
                continue
            th = self.thresholds[int(e)]
            # 비율 = 실제 횟수 / 허용 횟수. 1 을 넘으면 허용치 초과. 허용치가 0 이면 무한대(무조건 초과)
            r = np.inf if th == 0 else c / th
            if r > ratio:
                worst, ratio = int(e), r
        if unknown:
            # 학습 때 못 본 이벤트가 있으면 계산할 것 없이 점수 1.0 으로 즉시 이상
            return DetectionResult(1.0, True, "unknown_event", self.name,
                                   {"unknown_events": unknown,
                                    "templates": [self.templates.get(e) for e in unknown if e is not None]})
        is_anom = ratio > 1                    # 어떤 이벤트든 허용 횟수를 넘었으면 이상
        # 정규화 점수: ratio/(ratio+1) → ratio=1(딱 허용치)일 때 0.5. 다른 모델과 같은 0.5 기준선
        score = 1.0 if np.isinf(ratio) else ratio / (ratio + 1)
        details = {}
        if worst is not None:
            # 판정 근거: 어느 이벤트가 몇 번 나왔고 허용치가 얼마였는지
            details = {"event_id": worst, "count": int(counts[worst]), "threshold": self.thresholds[worst],
                       "template": self.templates.get(worst)}
        return DetectionResult(float(score), bool(is_anom), "model" if is_anom else "normal", self.name, details)

    def to_json(self) -> dict:
        """파일 저장용 dict. json 의 키는 문자열만 가능하므로 Event ID 를 str 로 바꾼다."""
        return {"name": self.name, "percentile": self.percentile,
                "thresholds": {str(k): v for k, v in self.thresholds.items()},
                "templates": {str(k): v for k, v in self.templates.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "StatisticalFrequencyDetector":
        """to_json() 으로 저장한 dict 에서 탐지기를 다시 만든다 (키를 다시 int 로)."""
        return cls({int(k): v for k, v in d["thresholds"].items()}, d["percentile"],
                   {int(k): v for k, v in d["templates"].items()})


# json 파일로 저장·복원하는 탐지기 목록 (딥러닝 모델은 .pt 파일이라 아래 load_detector 에서 따로 처리)
REGISTRY = {"statistical_frequency": StatisticalFrequencyDetector}


def load_detector(name: str = "statistical_frequency", cfg: dict | None = None) -> Detector:
    """이름으로 탐지기를 불러온다. 백엔드는 서버 시작 시 이 함수를 한 번 호출해 두고 계속 재사용한다.

    Args:
        name: "statistical_frequency" | "deeplog" | "autoencoder" (권장: RECOMMENDED_DETECTOR)
        cfg : 설정 (생략하면 기본 설정)
    파일 위치:
        statistical_frequency → artifacts/models/statistical_frequency.json
        deeplog / autoencoder → artifacts/models/<name>.pt   (git 에 포함되어 있어 데이터 없이도 사용 가능)
    """
    cfg = cfg or load_config()
    if name in ("deeplog", "autoencoder"):
        # 딥러닝 모델 모듈은 PyTorch 를 불러오므로, 필요할 때만 import (importlib: 이름 문자열로 모듈 import)
        import importlib
        mod = importlib.import_module(f"logguard.models.{name}")
        cls = mod.DeepLogDetector if name == "deeplog" else mod.AutoencoderDetector
        path = cfg["paths"]["artifacts_dir"] / "models" / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} 없음 → python -m logguard.models.{name}")
        return cls.load(path)
    path = cfg["paths"]["artifacts_dir"] / "models" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음 → python -m logguard.models.inference --export {name}")
    return REGISTRY[name].from_json(json.loads(path.read_text()))


def export_statistical(cfg: dict) -> None:
    """통계 빈도 탐지기를 만들어 파일로 저장하고, 저장본이 베이스라인 결과를 그대로 재현하는지 확인한다.

    순서: 베이스라인 결과에서 val 로 고른 백분위수 읽기 → 정상 학습 세션으로 임계치 계산 → json 저장
          → 저장한 파일을 다시 불러와 test 판정 → F1 이 베이스라인(0.8873)과 같은지 눈으로 확인
    """
    proc, interim, res = cfg["paths"]["processed_dir"], cfg["paths"]["interim_dir"], cfg["paths"]["results_dir"]
    best = json.loads((res / "baseline_statistical_frequency.json").read_text())["best_config"]
    tpl = pd.read_csv(interim / "hdfs_templates.csv")
    templates = dict(zip(tpl["event_id"].astype(int), tpl["template"]))
    train = pd.read_parquet(proc / "train.parquet")
    det = StatisticalFrequencyDetector.fit(train[train["label"] == 0], best["percentile"], templates)

    out = cfg["paths"]["artifacts_dir"] / "models" / "statistical_frequency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(det.to_json(), indent=2, ensure_ascii=False))
    print(f"[export] {out}")

    # 검증: 저장한 Detector 로 test 세션을 다시 판정 → 베이스라인 결과와 같아야 함
    det = load_detector("statistical_frequency", cfg)
    test = pd.read_parquet(proc / "test.parquet")
    t0 = time.perf_counter()                         # perf_counter: 짧은 시간 측정에 정확한 시계
    pred = [det.score_session(seq.tolist()).is_anomaly for seq in test["events"]]
    per_call_ms = (time.perf_counter() - t0) / len(test) * 1e3      # 세션 1개 판정에 걸린 평균 ms
    m = prf(test["label"], pred)
    print(f"[verify] test P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}  ({per_call_ms:.3f} ms/session)")


def main() -> None:
    """명령줄 진입점 (`make export-detector`)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--export", choices=list(REGISTRY), required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.export == "statistical_frequency":
        export_statistical(cfg)


if __name__ == "__main__":
    main()
