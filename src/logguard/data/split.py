"""Block ID 세션 단위로 train/val/test를 분할한다 (데이터 누수 방지).

규칙 (계획서 Ⅱ-4):
  - 분할 단위는 Block ID 세션. 같은 블록의 윈도우는 절대 서로 다른 셋에 들어가지 않는다.
  - train : 정상 세션만 사용 (이상 세션은 버리고 개수만 기록)
  - val   : 정상+이상 모두 저장. 임계치는 val 의 정상 세션(label==0) 스코어 백분위수로 산출
  - test  : 정상+이상, 실제 클래스 분포 유지

입력 : data/interim/hdfs_sessions.parquet, hdfs_templates.csv
출력 : data/processed/{train,val,test}.parquet, event_vocab.json, split_summary.json

사용법: python -m logguard.data.split [--strategy random|chrono]

────────────────────────────────────────────────────────────────────────────────────────────
세 데이터셋의 역할 (시험에 비유)
────────────────────────────────────────────────────────────────────────────────────────────
  train (60%) : 교과서. 모델이 "정상 패턴" 을 배우는 데이터. 이상 세션은 빼고 정상만 남긴다.
                (현실에서는 어떤 이상이 올지 모르므로, 정상만 보고 "정상과 다르면 이상" 을 배우는 방식)
  val   (20%) : 모의고사. 임계치·하이퍼파라미터를 고르는 데 쓴다. 정상과 이상이 섞여 있다.
  test  (20%) : 수능. 모든 선택이 끝난 뒤 "딱 한 번" 최종 실력을 재는 데 쓴다.
                test 를 보고 설정을 바꾸면 수능 문제를 미리 보고 공부한 셈이 되어 점수가 부풀려진다.

"데이터 누수(leakage)" 란?
  같은 블록에서 나온 윈도우가 train 과 test 에 동시에 들어가면, 모델이 test 문제를 이미 본 셈이 된다.
  그래서 윈도우가 아니라 "블록(세션)" 단위로 먼저 나누고, 나중에 각 셋 안에서만 윈도우를 만든다.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from logguard.config import load_config
from logguard.data.windows import EOS, NUM_SPECIAL, PAD, UNK


def assign_splits(sess: pd.DataFrame, ratios: dict, strategy: str, seed: int) -> pd.Series:
    """각 세션(행)에 "train" / "val" / "test" 중 하나의 이름표를 붙인다.

    Args:
        sess    : 세션 테이블 (block_id, first_line, label 열 필요)
        ratios  : {"train": 0.6, "val": 0.2, "test": 0.2}  (test 는 나머지 전부)
        strategy: "random" = 라벨 비율을 유지하며 무작위로 나눔 (기본값)
                  "chrono" = 블록이 처음 등장한 시간 순서대로 앞 60% / 다음 20% / 마지막 20%
        seed    : 무작위 섞기의 씨앗. 같은 값이면 언제 실행해도 똑같이 나뉜다 (재현성)

    Returns:
        세션마다 "train"/"val"/"test" 문자열이 들어 있는 Series (sess 와 같은 순서)
    """
    r_train, r_val = ratios["train"], ratios["val"]
    # 일단 전부 "test" 로 채워 두고, train/val 에 해당하는 것만 덮어쓴다 (나머지는 자동으로 test)
    split = pd.Series("test", index=sess.index)

    if strategy == "chrono":
        # 시간순 분할: 처음 등장한 줄 번호(first_line)로 순위를 매기고 0~1 사이 비율로 바꾼다.
        # rank(method="first"): 같은 값이 있어도 먼저 나온 쪽에 앞 순위 → 순위가 겹치지 않는다
        order = sess["first_line"].rank(method="first").to_numpy() / len(sess)
        split[order <= r_train] = "train"                                   # 앞쪽 60%
        split[(order > r_train) & (order <= r_train + r_val)] = "val"       # 그다음 20%
        return split

    if strategy == "random":
        # 라벨별로 따로 섞어서 각 셋의 이상 비율을 원본과 동일하게 유지 (층화)
        #   예: 원본 이상 비율 2.93% → train/val/test 모두 이상 비율이 2.93% 가 된다.
        #   그냥 전체를 섞으면 우연히 한쪽에 이상이 몰릴 수 있다.
        rng = np.random.default_rng(seed)       # 시드를 고정한 난수 생성기
        # groupby("label").groups : {0: 정상 세션들의 인덱스, 1: 이상 세션들의 인덱스}
        for _, idx in sess.groupby("label").groups.items():
            idx = rng.permutation(np.asarray(idx))                    # 같은 라벨끼리 무작위로 섞기
            n_tr, n_va = int(len(idx) * r_train), int(len(idx) * r_val)
            split[idx[:n_tr]] = "train"                               # 앞 60% → train
            split[idx[n_tr:n_tr + n_va]] = "val"                      # 다음 20% → val (나머지는 test 그대로)
        return split

    # 오타 등으로 알 수 없는 방식이 들어오면 조용히 넘어가지 않고 바로 오류를 낸다
    raise ValueError(f"unknown strategy: {strategy}")


def build_vocab(train: pd.DataFrame, templates: pd.DataFrame) -> dict:
    """학습셋에 나온 이벤트만 어휘에 포함. 0=PAD, 1=UNK(학습 때 못 본 이벤트), 2=EOS(세션 종료).

    왜 "학습셋에 나온 이벤트만" 넣나?
        모델은 train 의 정상 세션만 보고 배운다. train 에 한 번도 안 나온 이벤트는 모델 입장에서
        "처음 보는 것" 이므로 UNK 로 통일하고, 실제 판정에서는 UNK 가 나오면 즉시 이상으로 처리한다.
        (EDA: 정상 세션에는 없고 이상 세션에만 나오는 이벤트가 33종 → 이것만으로 이상의 약 23% 를 잡는다)

    Returns:
        event_vocab.json 으로 저장될 dict (windows.load_vocab 설명 참고)
    """
    # train 의 모든 세션에서 나온 Event ID 를 중복 없이 모아 정렬 (set comprehension 사용)
    seen = sorted({int(e) for seq in train["events"] for e in np.unique(seq)})
    tpl = dict(zip(templates["event_id"], templates["template"]))     # Event ID → 템플릿 문장
    return {
        "pad_index": PAD,
        "unk_index": UNK,
        "eos_index": EOS,
        # Event ID 를 정렬된 순서대로 3, 4, 5, … 로 번호 매김 (0~2 는 특수 토큰 자리)
        # json 키는 문자열만 가능하므로 str(e)
        "event_to_index": {str(e): i + NUM_SPECIAL for i, e in enumerate(seen)},
        # 반대 방향(모델 번호 → 템플릿 문장): 결과를 사람이 읽을 때 사용
        "index_to_template": {str(i + NUM_SPECIAL): tpl.get(e, "") for i, e in enumerate(seen)},
        # 모델 출력층 크기 = 특수 토큰 3개 + 학습에 나온 이벤트 수
        "num_classes": len(seen) + NUM_SPECIAL,
    }


def run(cfg: dict, strategy: str | None = None) -> None:
    """세션 테이블을 읽어 train/val/test 로 나누고, 어휘와 요약 정보를 저장한다."""
    interim, out = cfg["paths"]["interim_dir"], cfg["paths"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    scfg = cfg["split"]
    # 명령줄에서 --strategy 를 주면 그걸, 아니면 config.yaml 의 값을 쓴다
    strategy = strategy or scfg["strategy"]

    sess = pd.read_parquet(interim / "hdfs_sessions.parquet")
    templates = pd.read_csv(interim / "hdfs_templates.csv")
    sess["split"] = assign_splits(sess, scfg["ratios"], strategy, cfg["seed"])

    # 누수 검사: 블록 ID 는 정확히 하나의 셋에만 속해야 함
    #   세션 테이블에서 블록 ID 가 유일하면, 각 블록은 이름표를 딱 하나만 받으므로 두 셋에 동시에 들어갈 수 없다.
    #   assert 조건이 거짓이면 프로그램이 즉시 멈추고 뒤의 메시지를 보여 준다.
    assert sess["block_id"].is_unique, "중복 block_id 존재"

    summary = {"strategy": strategy, "seed": cfg["seed"], "ratios": scfg["ratios"], "splits": {}}
    for name in ("train", "val", "test"):
        # 이 셋에 해당하는 세션만 골라내고, 이름표 열("split")은 저장할 필요가 없으니 지운다
        part = sess[sess["split"] == name].drop(columns="split")
        dropped = 0
        if name == "train" and scfg["train_normal_only"]:
            # train 은 정상만 남긴다. 버린 이상 세션 수는 기록해 둔다 (보고서·검증용)
            dropped = int(part["label"].sum())
            part = part[part["label"] == 0]
        part = part.reset_index(drop=True)                 # 행 번호를 0 부터 다시 매김
        part.to_parquet(out / f"{name}.parquet", index=False)
        summary["splits"][name] = {
            "sessions": len(part),
            "normal": int((part["label"] == 0).sum()),
            "anomaly": int(part["label"].sum()),
            "anomaly_ratio": round(float(part["label"].mean()), 5) if len(part) else 0.0,
            "anomaly_dropped": dropped,
            "events_total": int(part["length"].sum()),     # 이 셋의 전체 이벤트(로그 줄) 수
        }
        if name == "train":
            # 어휘는 반드시 "train(정상)" 만 보고 만든다. val/test 를 보고 만들면 누수가 된다.
            vocab = build_vocab(part, templates)

    # 참고 정보: test 에는 나오지만 학습 어휘에 없는(=UNK 가 될) 이벤트 목록
    test = pd.read_parquet(out / "test.parquet")
    known = {int(k) for k in vocab["event_to_index"]}
    # 집합의 빼기(-): test 에 나온 이벤트 집합 - 어휘에 있는 이벤트 집합
    unseen = sorted({int(e) for seq in test["events"] for e in np.unique(seq)} - known)
    summary["vocab_size"] = vocab["num_classes"]
    summary["test_unseen_events"] = unseen

    (out / "event_vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False))
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    """명령줄 진입점. 예: python -m logguard.data.split --strategy chrono"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--strategy", choices=["random", "chrono"], default=None)   # 두 값 외에는 거부
    args = ap.parse_args()
    run(load_config(args.config), args.strategy)


if __name__ == "__main__":
    main()
