"""공통 설정 로더. 모든 모듈은 여기서 경로와 하이퍼파라미터를 가져온다."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def raw_log_path(cfg: dict) -> Path:
    return cfg["paths"]["raw_dir"] / cfg["dataset"]["log_file"]


def label_path(cfg: dict) -> Path:
    return cfg["paths"]["raw_dir"] / cfg["dataset"]["label_file"]
