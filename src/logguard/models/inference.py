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
"""
from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from logguard.config import load_config
from logguard.eval.metrics import prf

THRESHOLD = 0.5
# 6단계 최종 평가로 채택한 모델 (test F1 0.977, results/final_test.json). 백엔드는 이 이름을 사용
RECOMMENDED_DETECTOR = "autoencoder"


@dataclass
class DetectionResult:
    score: float                      # 0~1, 0.5 이상이면 이상
    is_anomaly: bool
    reason: str                       # "unknown_event" | "model" | "normal"
    model: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Detector(ABC):
    name: str

    @abstractmethod
    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        ...

    def score_event(self, history: Sequence[int | None]) -> DetectionResult:
        return self.score_session(history, closed=False)


class StatisticalFrequencyDetector(Detector):
    name = "statistical_frequency"

    def __init__(self, thresholds: dict[int, float], percentile: float, templates: dict[int, str] | None = None):
        self.thresholds = {int(k): float(v) for k, v in thresholds.items()}
        self.percentile = percentile
        self.templates = templates or {}

    @classmethod
    def fit(cls, train_normal: pd.DataFrame, percentile: float, templates: dict[int, str]) -> "StatisticalFrequencyDetector":
        from logguard.baseline.rule_based import count_matrix
        n_events = max(templates) + 1
        th = np.percentile(count_matrix(train_normal, n_events), percentile, axis=0)
        seen = {int(e) for seq in train_normal["events"] for e in np.unique(seq)}
        return cls({e: th[e] for e in sorted(seen)}, percentile, templates)

    def score_session(self, events: Sequence[int | None], closed: bool = True) -> DetectionResult:
        counts = pd.Series([e for e in events]).value_counts(dropna=False)
        worst, ratio, unknown = None, 0.0, []
        for e, c in counts.items():
            if e is None or (isinstance(e, float) and np.isnan(e)) or int(e) not in self.thresholds:
                unknown.append(None if e is None or (isinstance(e, float) and np.isnan(e)) else int(e))
                continue
            th = self.thresholds[int(e)]
            r = np.inf if th == 0 else c / th
            if r > ratio:
                worst, ratio = int(e), r
        if unknown:
            return DetectionResult(1.0, True, "unknown_event", self.name,
                                   {"unknown_events": unknown,
                                    "templates": [self.templates.get(e) for e in unknown if e is not None]})
        is_anom = ratio > 1
        score = 1.0 if np.isinf(ratio) else ratio / (ratio + 1)
        details = {}
        if worst is not None:
            details = {"event_id": worst, "count": int(counts[worst]), "threshold": self.thresholds[worst],
                       "template": self.templates.get(worst)}
        return DetectionResult(float(score), bool(is_anom), "model" if is_anom else "normal", self.name, details)

    def to_json(self) -> dict:
        return {"name": self.name, "percentile": self.percentile,
                "thresholds": {str(k): v for k, v in self.thresholds.items()},
                "templates": {str(k): v for k, v in self.templates.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "StatisticalFrequencyDetector":
        return cls({int(k): v for k, v in d["thresholds"].items()}, d["percentile"],
                   {int(k): v for k, v in d["templates"].items()})


REGISTRY = {"statistical_frequency": StatisticalFrequencyDetector}


def load_detector(name: str = "statistical_frequency", cfg: dict | None = None) -> Detector:
    cfg = cfg or load_config()
    if name in ("deeplog", "autoencoder"):
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
    t0 = time.perf_counter()
    pred = [det.score_session(seq.tolist()).is_anomaly for seq in test["events"]]
    per_call_ms = (time.perf_counter() - t0) / len(test) * 1e3
    m = prf(test["label"], pred)
    print(f"[verify] test P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}  ({per_call_ms:.3f} ms/session)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--export", choices=list(REGISTRY), required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.export == "statistical_frequency":
        export_statistical(cfg)


if __name__ == "__main__":
    main()
