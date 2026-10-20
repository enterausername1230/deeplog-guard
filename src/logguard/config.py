"""공통 설정 로더. 모든 모듈은 여기서 경로와 하이퍼파라미터를 가져온다.

왜 이런 파일이 필요한가?
    데이터 경로("data/raw/HDFS_v1"), 학습 횟수(epochs), 윈도우 크기 같은 값을 코드 곳곳에 직접 쓰면
    값을 바꿀 때 여러 파일을 고쳐야 하고 실수도 생긴다.
    그래서 모든 설정값은 configs/config.yaml 한 곳에 모아 두고, 코드에서는 이 파일의
    load_config() 로 읽어서 쓴다. → "설정 바꾸기 = yaml 한 줄 고치기"

사용 예
    from logguard.config import load_config
    cfg = load_config()                       # configs/config.yaml 을 읽음
    cfg["paths"]["processed_dir"]             # → /…/dplngproj/data/processed (절대 경로)
    cfg["autoencoder"]["epochs"]              # → 300
"""
# 파이썬 3.10 미만에서도 `str | Path` 같은 새 타입 표기를 쓸 수 있게 해 주는 설정
from __future__ import annotations

from pathlib import Path   # 운영체제(Mac/Windows)에 상관없이 경로를 다루는 표준 도구
from typing import Any      # "아무 타입이나 올 수 있음" 을 표시하는 타입 힌트

import yaml                 # YAML 설정 파일을 파이썬 dict 로 읽어 주는 라이브러리 (pyyaml)

# ---------------------------------------------------------------------------------------------
# 프로젝트 루트(최상위 폴더) 경로 계산
#   이 파일 위치:  <루트>/src/logguard/config.py
#   .resolve()   : 상대 경로를 절대 경로로 바꿈 (예: /Users/…/dplngproj/src/logguard/config.py)
#   .parents[0]  : <루트>/src/logguard
#   .parents[1]  : <루트>/src
#   .parents[2]  : <루트>                ← 우리가 원하는 프로젝트 최상위 폴더
#   어느 폴더에서 코드를 실행하든 항상 같은 루트를 가리키므로 경로가 꼬이지 않는다.
# ---------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# 기본 설정 파일 위치: <루트>/configs/config.yaml
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """설정 파일(yaml)을 읽어서 파이썬 dict 로 돌려준다.

    Args:
        path: 읽을 설정 파일 경로. 생략하면(None) 기본 파일 configs/config.yaml 을 읽는다.
              실험용으로 다른 설정 파일을 쓰고 싶을 때만 지정한다.

    Returns:
        설정 dict. 특히 cfg["paths"] 안의 모든 경로는 "프로젝트 루트 기준 절대 경로(Path 객체)" 로
        바뀌어 있으므로 바로 파일을 열거나 저장하는 데 쓸 수 있다.
    """
    # `path or DEFAULT_CONFIG` : path 가 None(지정 안 함)이면 기본 설정 파일을 사용
    # encoding="utf-8"         : 설정 파일에 한글 주석이 있으므로 UTF-8 로 읽어야 깨지지 않는다
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        # safe_load : yaml 텍스트 → dict. (load 대신 safe_load 를 쓰면 임의 코드 실행 위험이 없다)
        cfg = yaml.safe_load(f)

    # yaml 에는 "data/raw/HDFS_v1" 처럼 루트 기준 상대 경로 문자열이 적혀 있다.
    # 이를 ROOT / "data/raw/HDFS_v1" → 절대 Path 객체로 바꿔서 다시 넣어 준다.
    # (dict comprehension: {키: 값 for 키, 값 in 원래dict.items()} 형태로 새 dict 를 만든다)
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def raw_log_path(cfg: dict) -> Path:
    """원본 HDFS 로그 파일 경로를 돌려준다.

    예: <루트>/data/raw/HDFS_v1/HDFS.log  (약 1.5GB, 1,117만 줄)
    여러 모듈(parse, drain_tuning, final_eval …)이 같은 경로를 쓰므로 한 곳에서 계산한다.
    """
    return cfg["paths"]["raw_dir"] / cfg["dataset"]["log_file"]


def label_path(cfg: dict) -> Path:
    """정답 라벨 파일 경로를 돌려준다.

    예: <루트>/data/raw/HDFS_v1/preprocessed/anomaly_label.csv
    내용: "BlockId,Label" 형식으로 블록마다 Normal(정상) / Anomaly(이상) 가 적혀 있다.
    """
    return cfg["paths"]["raw_dir"] / cfg["dataset"]["label_file"]
