"""Block ID 세션 단위로 train/val/test를 분할한다 (데이터 누수 방지).

규칙 (계획서 Ⅱ-4):
  - 분할 단위는 Block ID 세션. 같은 블록의 윈도우는 절대 서로 다른 셋에 들어가지 않는다.
  - train : 정상 세션만 사용 (이상 세션은 버리고 개수만 기록)
  - val   : 정상+이상 모두 저장. 임계치는 val 의 정상 세션(label==0) 스코어 백분위수로 산출
  - test  : 정상+이상, 실제 클래스 분포 유지

입력 : data/interim/hdfs_sessions.parquet, hdfs_templates.csv
출력 : data/processed/{train,val,test}.parquet, event_vocab.json, split_summary.json

사용법: python -m logguard.data.split [--strategy random|chrono]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from logguard.config import load_config
from logguard.data.windows import EOS, NUM_SPECIAL, PAD, UNK


def assign_splits(sess: pd.DataFrame, ratios: dict, strategy: str, seed: int) -> pd.Series:
    r_train, r_val = ratios["train"], ratios["val"]
    split = pd.Series("test", index=sess.index)

    if strategy == "chrono":
        order = sess["first_line"].rank(method="first").to_numpy() / len(sess)
        split[order <= r_train] = "train"
        split[(order > r_train) & (order <= r_train + r_val)] = "val"
        return split

    if strategy == "random":
        # 라벨별로 따로 섞어서 각 셋의 이상 비율을 원본과 동일하게 유지 (층화)
        rng = np.random.default_rng(seed)
        for _, idx in sess.groupby("label").groups.items():
            idx = rng.permutation(np.asarray(idx))
            n_tr, n_va = int(len(idx) * r_train), int(len(idx) * r_val)
            split[idx[:n_tr]] = "train"
            split[idx[n_tr:n_tr + n_va]] = "val"
        return split

    raise ValueError(f"unknown strategy: {strategy}")


def build_vocab(train: pd.DataFrame, templates: pd.DataFrame) -> dict:
    """학습셋에 나온 이벤트만 어휘에 포함. 0=PAD, 1=UNK(학습 때 못 본 이벤트), 2=EOS(세션 종료)."""
    seen = sorted({int(e) for seq in train["events"] for e in np.unique(seq)})
    tpl = dict(zip(templates["event_id"], templates["template"]))
    return {
        "pad_index": PAD,
        "unk_index": UNK,
        "eos_index": EOS,
        "event_to_index": {str(e): i + NUM_SPECIAL for i, e in enumerate(seen)},
        "index_to_template": {str(i + NUM_SPECIAL): tpl.get(e, "") for i, e in enumerate(seen)},
        "num_classes": len(seen) + NUM_SPECIAL,
    }


def run(cfg: dict, strategy: str | None = None) -> None:
    interim, out = cfg["paths"]["interim_dir"], cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    scfg = cfg["split"]
    strategy = strategy or scfg["strategy"]

    sess = pd.read_parquet(interim / "hdfs_sessions.parquet")
    templates = pd.read_csv(interim / "hdfs_templates.csv")
    sess["split"] = assign_splits(sess, scfg["ratios"], strategy, cfg["seed"])

    # 누수 검사: 블록 ID 는 정확히 하나의 셋에만 속해야 함
    assert sess["block_id"].is_unique, "중복 block_id 존재"

    summary = {"strategy": strategy, "seed": cfg["seed"], "ratios": scfg["ratios"], "splits": {}}
    for name in ("train", "val", "test"):
        part = sess[sess["split"] == name].drop(columns="split")
        dropped = 0
        if name == "train" and scfg["train_normal_only"]:
            dropped = int(part["label"].sum())
            part = part[part["label"] == 0]
        part = part.reset_index(drop=True)
        part.to_parquet(out / f"{name}.parquet", index=False)
        summary["splits"][name] = {
            "sessions": len(part),
            "normal": int((part["label"] == 0).sum()),
            "anomaly": int(part["label"].sum()),
            "anomaly_ratio": round(float(part["label"].mean()), 5) if len(part) else 0.0,
            "anomaly_dropped": dropped,
            "events_total": int(part["length"].sum()),
        }
        if name == "train":
            vocab = build_vocab(part, templates)

    test = pd.read_parquet(out / "test.parquet")
    known = {int(k) for k in vocab["event_to_index"]}
    unseen = sorted({int(e) for seq in test["events"] for e in np.unique(seq)} - known)
    summary["vocab_size"] = vocab["num_classes"]
    summary["test_unseen_events"] = unseen

    (out / "event_vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False))
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--strategy", choices=["random", "chrono"], default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.strategy)


if __name__ == "__main__":
    main()
