"""시연용 샘플 재생 준비와 결과 채점 (2026-11-20, 옥상민).

왜 필요한가?
    빠른 시연(`./start.sh`, 앞 2만 줄)은 세션 도중에 재생을 멈추므로 "세션 종료 판정" 을 보여 줄 수 없다
    (진행 중인 정상 세션까지 '끊긴 세션' 으로 판정되기 때문). 전체 재생(`./start.sh full`)은 약 1시간이 걸린다.
    그래서 test 블록 일부를 무작위로 골라 **그 블록들의 로그만 처음부터 끝까지** 원래 순서대로 보낸다.
      - 이상 비율이 실제와 같다 (무작위 추출, 약 2.9%)
      - 모든 세션이 끝까지 들어오므로 재생 후 세션 종료(EOS) 판정이 정확하다
      - 블록 1,000개 ≈ 로그 약 1만 9천 줄 ≈ 30초
    주의: 앞부분에서 끝나는 블록만 고르면 안 된다. HDFS 정상 블록은 삭제 로그가 몇 시간 뒤에 찍혀서
          일찍 끝나는 블록은 대부분 이상이다 (앞 50만 줄 안에서 끝나는 블록의 이상 비율 35%).

사용 (start.sh demo 가 자동으로 부른다)
    python -m logguard.api.demo prepare --blocks 1000   # 샘플 로그 파일 만들기 (처음 한 번, 원본 전체를 훑어 약 10초)
    python -m logguard.api.demo score   --blocks 1000   # 재생·종료 판정이 끝난 DB 로 이 샘플의 정확도 계산
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from logguard.api.replay import iter_logs
from logguard.api.settings import load_settings
from logguard.config import ROOT, load_config, raw_log_path
from logguard.eval.metrics import prf

DEMO_DIR = ROOT / "artifacts" / "demo"          # git 제외 폴더


def paths(n_blocks: int, seed: int) -> tuple[Path, Path]:
    """샘플 로그 파일과 샘플 블록 목록(정답 포함) 파일 경로"""
    tag = f"{n_blocks}_s{seed}"
    return DEMO_DIR / f"demo_lines_{tag}.txt", DEMO_DIR / f"demo_blocks_{tag}.csv"


def prepare(n_blocks: int, seed: int) -> Path:
    """test 블록 n_blocks 개를 무작위로 골라 그 블록들의 로그를 원래 순서대로 파일에 저장한다."""
    lines_path, blocks_path = paths(n_blocks, seed)
    if lines_path.exists() and blocks_path.exists():
        print(f"[demo] 샘플 파일 있음: {lines_path.relative_to(ROOT)}", flush=True)
        return lines_path
    cfg = load_config()
    test = pd.read_parquet(cfg["paths"]["processed_dir"] / "test.parquet", columns=["block_id", "label"])
    sample = test.sample(n=min(n_blocks, len(test)), random_state=seed)
    print(f"[demo] test 블록 {len(sample):,}개 선택 (이상 {int(sample['label'].sum())}개). "
          f"원본에서 로그를 뽑는 중 (처음 한 번만, 약 10초)...", flush=True)
    # 재생 모듈과 같은 규칙: 줄 안의 블록이 모두 선택된 블록인 줄만
    lines = list(iter_logs(raw_log_path(cfg), set(sample["block_id"])))
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    lines_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sample.to_csv(blocks_path, index=False)
    print(f"[demo] 로그 {len(lines):,}줄 저장: {lines_path.relative_to(ROOT)}", flush=True)
    return lines_path


def score(n_blocks: int, seed: int) -> dict:
    """DB 의 판정 결과로 샘플 블록의 정확도를 계산해 출력한다 (실시간만 / 세션 종료 판정 포함)."""
    _, blocks_path = paths(n_blocks, seed)
    sample = pd.read_csv(blocks_path)
    db = sqlite3.connect(load_settings()["database"])
    q = lambda sql: {b for (b,) in db.execute(sql)}
    realtime = q("SELECT DISTINCT block_id FROM logs WHERE status = 'anomaly'")
    closed = q("SELECT block_id FROM session_results WHERE is_anomaly = 1")
    y = sample["label"].to_numpy()
    rt = prf(y, sample["block_id"].isin(realtime))
    final = prf(y, sample["block_id"].isin(realtime | closed))
    only_close = int((sample["block_id"].isin(closed - realtime) & (sample["label"] == 1)).sum())
    print(f"[demo] 시연 샘플 세션 {len(sample):,}개 (실제 이상 {int(y.sum())}개) 채점", flush=True)
    print(f"   실시간 판정만        : 이상 {rt['tp']}개 탐지 · 오탐 {rt['fp']} · 놓침 {rt['fn']} → F1 {rt['f1']:.3f}")
    print(f"   + 세션 종료 판정     : 이상 {final['tp']}개 탐지 · 오탐 {final['fp']} · 놓침 {final['fn']} → F1 {final['f1']:.3f}"
          f"  (종료 판정으로만 잡은 이상 {only_close}개)")
    return {"realtime": rt, "final": final, "close_only_tp": only_close}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "score"])
    ap.add_argument("--blocks", type=int, default=1000, help="무작위로 고를 test 블록 수 (기본 1000 ≈ 로그 1만 9천 줄)")
    ap.add_argument("--seed", type=int, default=42, help="같은 seed 면 매번 같은 블록 (리허설과 본 시연이 같아진다)")
    args = ap.parse_args()
    if args.action == "prepare":
        print(prepare(args.blocks, args.seed))         # 마지막 줄 = 샘플 파일 경로 (start.sh 가 읽는다)
    else:
        score(args.blocks, args.seed)


if __name__ == "__main__":
    main()
