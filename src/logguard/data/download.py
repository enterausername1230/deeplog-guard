"""LogPAI Loghub HDFS_v1 데이터셋 다운로드 및 압축 해제.

사용법: python -m logguard.data.download
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile

from tqdm import tqdm

from logguard.config import ROOT, label_path, load_config, raw_log_path


def download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 받기")
    args = ap.parse_args()

    cfg = load_config(args.config)
    zip_path = cfg["paths"]["download_dir"] / cfg["dataset"]["zip_name"]
    raw_dir = cfg["paths"]["raw_dir"]

    if args.force or not zip_path.exists():
        print(f"[download] {cfg['dataset']['url']}")
        download(cfg["dataset"]["url"], zip_path)
    else:
        print(f"[download] 이미 존재: {zip_path}")

    if args.force or not raw_log_path(cfg).exists():
        print(f"[unzip] -> {raw_dir}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    for p in (raw_log_path(cfg), label_path(cfg)):
        if not p.exists():
            raise FileNotFoundError(f"필수 파일 없음: {p}")
        print(f"[ok] {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
