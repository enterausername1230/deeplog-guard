"""Replay HDFS logs in order. No automatic retry: POST is not idempotent."""
import argparse
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


def replay(path, url, rate, timeout, limit=None, allowed_blocks=None):
    if rate <= 0 or timeout <= 0 or (limit is not None and limit <= 0):
        raise ValueError('rate, timeout and limit must be positive')
    sent = 0
    with httpx.Client(timeout=timeout) as client:
        next_send = time.monotonic()
        for raw in iter_logs(path, allowed_blocks):
            time.sleep(max(0, next_send - time.monotonic()))
            response = client.post(url, json={'raw_log': raw})
            response.raise_for_status()
            sent += 1
            if limit is not None and sent >= limit:
                break
            next_send = max(next_send + 1 / rate, time.monotonic())
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
    args = parser.parse_args()
    allowed = None
    if args.test_sessions:
        import pandas as pd
        allowed = set(pd.read_parquet(args.test_sessions, columns=['block_id'])['block_id'])
    count = replay(args.file, args.url, args.rate, args.timeout, args.limit, allowed)
    print(f'Sent {count} log lines')


if __name__ == '__main__':
    main()
