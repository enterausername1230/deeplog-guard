"""Locust 부하 시험 — 실제 HDFS test 로그를 원래 순서대로 보낸다.

원작성: 안명근 (같은 로그 한 줄 반복) / 2026-11-20 개선: 옥상민 (실제 로그 순환)

왜 바꿨나?
    예전에는 모든 요청이 "같은 블록의 같은 한 줄" 이었다. 그러면 Drain 매칭 캐시가 항상 맞고, DB 에서 꺼내는
    블록 이력도 늘 같은 블록이라 실제보다 빠르게 나올 수 있다 (계획서 목표 ④ 를 말하기에 근거가 약함).
    이제는 로그 재생 모듈(replay)과 같은 기준으로 고른 test 블록 로그를 원래 순서대로 돌아가며 보낸다.

동작
    1) 처음 실행할 때 원본 HDFS.log 에서 "모든 블록이 test 소속인 줄" 을 앞에서부터 N 줄 뽑아
       artifacts/locust/test_lines_N.txt 에 저장한다 (다음부터는 이 파일을 바로 읽는다, git 제외 폴더).
    2) 가상 사용자들이 공용 순번을 하나씩 가져가며 그 줄을 보낸다 → 전체적으로 원본 순서대로 흘러간다.
       (동시에 여러 요청이 나가므로 같은 블록 안에서 순서가 아주 조금 바뀔 수 있다 → 처리량 측정용이며,
        탐지 정확도는 replay 로 따로 확인한다)
    3) 요청마다 sent_at(보낸 시각)을 같이 보내 Slack 까지의 E2E 지연도 DB 에서 계산할 수 있다.

실행 (프로젝트 루트, 서버가 켜진 상태에서)
    .venv/bin/locust -f src/logguard/api/locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 50 -r 10 -t 70s --csv artifacts/locust/run
    50명 × 초당 10건 = 목표 500 logs/s. 상승 구간(약 5초) 이후 60초 이상을 본다.

환경변수 (명령 앞에 붙인다)
    LOCUST_LINES=100000    미리 뽑아 둘 줄 수 (기본 10만. 500건/초로 200초 분량, 넘으면 처음부터 다시 돈다)
    LOCUST_MODE=single     예전 방식(같은 한 줄 반복)으로 비교 측정
"""
import itertools
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from locust import HttpUser, constant_throughput, task

ROOT = Path(__file__).resolve().parents[3]          # 프로젝트 루트 (src/logguard/api/ 의 세 단계 위)
sys.path.insert(0, str(ROOT / "src"))               # locust 가 어디서 실행되든 이 저장소 코드를 쓰도록

N_LINES = int(os.environ.get("LOCUST_LINES", "100000"))
MODE = os.environ.get("LOCUST_MODE", "real")
CACHE = ROOT / "artifacts" / "locust" / f"test_lines_{N_LINES}.txt"
# 예전 방식에서 쓰던 한 줄 (LOCUST_MODE=single)
SINGLE_LINE = ("081109 203615 148 INFO dfs.DataNode$PacketResponder: "
               "PacketResponder 1 for block blk_38865049064139660 terminating")


def load_lines() -> list[str]:
    """보낼 로그 목록을 준비한다. 캐시 파일이 있으면 읽고, 없으면 원본에서 뽑아 만든다."""
    if MODE == "single":
        return [SINGLE_LINE]
    if CACHE.exists():
        return CACHE.read_text(encoding="utf-8").splitlines()

    # 재생 모듈과 똑같은 필터: 줄 안의 블록이 모두 test 블록인 줄만 (replay --test-sessions 와 같은 기준)
    import pandas as pd
    from logguard.api.replay import iter_logs
    from logguard.config import load_config, raw_log_path

    cfg = load_config()
    test_blocks = set(pd.read_parquet(cfg["paths"]["processed_dir"] / "test.parquet", columns=["block_id"])["block_id"])
    print(f"[locust] 원본에서 test 로그 {N_LINES:,}줄을 뽑는 중 (처음 한 번만)...", flush=True)
    lines = list(itertools.islice(iter_logs(raw_log_path(cfg), test_blocks), N_LINES))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[locust] {len(lines):,}줄 저장: {CACHE.relative_to(ROOT)}", flush=True)
    return lines


LINES = load_lines()
_ORDER = itertools.count()        # 모든 가상 사용자가 함께 쓰는 순번 (0, 1, 2, ...) → 원본 순서대로 나눠 보낸다


class LogUser(HttpUser):
    # 사용자 1명당 초당 10건. 50명이면 목표 500 logs/s (서버 응답이 느리면 실제로는 그보다 적게 나간다)
    wait_time = constant_throughput(10)

    @task
    def submit_log(self):
        raw = LINES[next(_ORDER) % len(LINES)]
        self.client.post("/logs", name="/logs", json={
            "raw_log": raw,
            "sent_at": datetime.now(timezone.utc).isoformat(),   # E2E 지연 계산용 (alerts.completed_at − logs.sent_at)
        })
