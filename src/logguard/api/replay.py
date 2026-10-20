"""Replay HDFS logs in order. No automatic retry: POST is not idempotent.

로그 재생 모듈 (원작성: 안명근 / 2026-10-20 sent_at·세션 종료 추가: 옥상민)
    공개 데이터셋 HDFS.log 는 이미 다 쌓인 파일이다. 이 모듈은 그 줄들을 원래 순서대로, 정해진 속도로
    서버(POST /logs)에 하나씩 보내 "실시간으로 로그가 들어오는 상황" 을 흉내 낸다.

    - 요청마다 sent_at(보낸 시각, UTC)을 함께 보낸다 → E2E 지연(전송 → Slack 완료) 계산에 사용
    - --close-sessions : 재생이 끝나면 POST /sessions/close 를 호출해 열린 세션을 한꺼번에 종료 판정
                          (중간에 끊긴 이상 세션까지 잡아 최종 F1 0.977 을 재현하려면 필요)

예) test 블록만 초당 500줄로 재생하고 끝나면 세션 종료 판정:
    PYTHONPATH=src .venv/bin/python -m logguard.api.replay data/raw/HDFS_v1/HDFS.log \
        --test-sessions data/processed/test.parquet --rate 500 --close-sessions
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import httpx
from .inference import BLOCK
from .settings import load_settings


def iter_logs(path, allowed_blocks=None):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            raw = line.rstrip('\r\n')
            if not raw:
                continue
            blocks = set(BLOCK.findall(raw))
            # A multi-block line is safe only when ALL its blocks belong to test.
            if allowed_blocks is not None and (not blocks or not blocks <= allowed_blocks):
                continue
            yield raw


def replay(path, url, rate, timeout, limit=None, allowed_blocks=None, close_url=None):
    """로그를 순서대로 보내고 보낸 줄 수를 돌려준다.

    Args:
        rate      : 초당 최대 전송 줄 수 (서버 응답이 느리면 실제 속도는 더 낮다)
        close_url : 주면 재생이 끝난 뒤 이 주소(POST /sessions/close)를 호출해 세션 종료 판정을 실행
    """
    if rate <= 0 or timeout <= 0 or (limit is not None and limit <= 0):
        raise ValueError('rate, timeout and limit must be positive')
    sent = 0
    with httpx.Client(timeout=timeout) as client:
        next_send = time.monotonic()
        for raw in iter_logs(path, allowed_blocks):
            time.sleep(max(0, next_send - time.monotonic()))
            # sent_at: 보낸 시각(UTC ISO). 서버가 DB 에 저장 → E2E 지연 = Slack 완료 시각 − sent_at
            response = client.post(url, json={'raw_log': raw, 'sent_at': datetime.now(timezone.utc).isoformat()})
            response.raise_for_status()
            sent += 1
            if limit is not None and sent >= limit:
                break
            next_send = max(next_send + 1 / rate, time.monotonic())
        if close_url is not None:
            # 열린 세션 전부 종료 판정 (블록 수만큼 모델 호출 → 오래 걸릴 수 있어 timeout 을 넉넉히)
            response = client.post(close_url, timeout=max(timeout, 600))
            response.raise_for_status()
            print(f"Closed sessions: {response.json().get('closed')}")
    return sent


def main():
    cfg = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--url', default=cfg['replay_url'])
    parser.add_argument('--rate', type=float, default=cfg['replay_rate'])
    parser.add_argument('--timeout', type=float, default=cfg['replay_timeout_seconds'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--test-sessions', type=Path, help='test.parquet; include only lines whose blocks are all test blocks')
    parser.add_argument('--close-sessions', action='store_true',
                        help='재생이 끝나면 POST /sessions/close 로 열린 세션을 종료 판정 (최종 세션 판정)')
    args = parser.parse_args()
    allowed = None
    if args.test_sessions:
        import pandas as pd
        allowed = set(pd.read_parquet(args.test_sessions, columns=['block_id'])['block_id'])
    # 수신 주소 .../logs → 종료 판정 주소 .../sessions/close
    close_url = args.url.rsplit('/logs', 1)[0] + '/sessions/close' if args.close_sessions else None
    count = replay(args.file, args.url, args.rate, args.timeout, args.limit, allowed, close_url)
    print(f'Sent {count} log lines')


if __name__ == '__main__':
    main()
