"""LogPAI Loghub HDFS_v1 데이터셋 다운로드 및 압축 해제.

이 파일은 무슨 일을 하나?
    프로젝트에서 쓰는 공개 데이터셋(HDFS_v1)을 인터넷(Zenodo)에서 내려받고 압축을 푼다.
    데이터가 커서(zip 187MB, 압축 해제 시 약 1.8GB) git 에는 올리지 않으므로,
    팀원은 각자 이 스크립트(또는 `make download`)로 같은 데이터를 받는다.

받은 뒤 폴더 모습
    data/raw/_download/HDFS_v1.zip                        ← 원본 zip (다시 받지 않도록 보관)
    data/raw/HDFS_v1/HDFS.log                             ← 서버 로그 1,117만 줄 (1.5GB)
    data/raw/HDFS_v1/preprocessed/anomaly_label.csv       ← 블록별 정상/이상 정답
    data/raw/HDFS_v1/preprocessed/HDFS.log_templates.csv  ← Loghub 공식 템플릿 29개 (drain_tuning 에서 사용)
    data/raw/HDFS_v1/preprocessed/Event_traces.csv        ← 블록별 공식 이벤트 시퀀스 (drain_tuning 에서 사용)

사용법: python -m logguard.data.download
"""
from __future__ import annotations

import argparse          # 명령줄 옵션(--force 등)을 읽는 표준 라이브러리
import urllib.request    # 파이썬 기본 HTTP 다운로드 도구 (추가 설치 불필요)
import zipfile           # zip 파일 압축 해제

from tqdm import tqdm    # 터미널에 진행률 막대(████ 45%)를 그려 주는 라이브러리

from logguard.config import ROOT, label_path, load_config, raw_log_path


def download(url: str, dest) -> None:
    """url 의 파일을 dest 경로에 저장한다. 진행률을 보여 주고, 중간에 끊겨도 깨진 파일이 남지 않게 한다.

    Args:
        url : 내려받을 주소 (configs/config.yaml 의 dataset.url)
        dest: 저장할 경로 (Path 객체). 예: data/raw/_download/HDFS_v1.zip
    """
    # 저장할 폴더가 없으면 만든다. parents=True: 중간 폴더까지 전부, exist_ok=True: 이미 있어도 오류 아님
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 바로 dest 에 쓰지 않고 "HDFS_v1.zip.part" 라는 임시 파일에 먼저 쓴다.
    # 다운로드가 중간에 끊기면 .part 파일만 남고, 완성본(.zip)은 생기지 않는다.
    # → 다음 실행 때 "zip 이 이미 있다" 고 착각해서 깨진 파일을 쓰는 사고를 막는다.
    tmp = dest.with_suffix(dest.suffix + ".part")

    # urlopen : 서버에 연결해 응답 스트림을 연다 / open(tmp, "wb") : 임시 파일을 바이너리 쓰기 모드로 연다
    # with 문을 쓰면 블록이 끝날 때(오류가 나도) 연결과 파일이 자동으로 닫힌다.
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        # 서버가 알려 주는 전체 파일 크기(바이트). 진행률 막대의 100% 기준으로 쓴다. 없으면 0.
        total = int(resp.headers.get("Content-Length", 0))
        # unit="B", unit_scale=True : 바이트를 KB/MB 단위로 보기 좋게 표시
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            # 1MB(1 << 20 = 2^20 바이트)씩 끊어서 읽는다. 187MB 를 한 번에 메모리에 올리지 않기 위해서다.
            # `while chunk := …` : 읽은 값을 chunk 에 넣고, 비어 있지 않은 동안 반복 (파이썬 3.8+ "바다코끼리 연산자")
            while chunk := resp.read(1 << 20):
                f.write(chunk)          # 받은 조각을 파일에 이어 쓰기
                bar.update(len(chunk))  # 진행률 막대를 받은 만큼 전진

    # 끝까지 무사히 받았을 때만 .part → .zip 으로 이름을 바꿔 "완성본" 으로 만든다.
    tmp.rename(dest)


def main() -> None:
    """명령줄 진입점: 필요하면 다운로드 → 필요하면 압축 해제 → 필수 파일 존재 확인."""
    # ---- 명령줄 옵션 정의 -------------------------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__)   # --help 를 치면 이 파일 맨 위 설명이 나온다
    ap.add_argument("--config", default=None)          # 다른 설정 파일을 쓰고 싶을 때만 지정
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 받기")  # 지정하면 True
    args = ap.parse_args()

    # ---- 경로 준비 -------------------------------------------------------------------------
    cfg = load_config(args.config)
    zip_path = cfg["paths"]["download_dir"] / cfg["dataset"]["zip_name"]   # data/raw/_download/HDFS_v1.zip
    raw_dir = cfg["paths"]["raw_dir"]                                       # data/raw/HDFS_v1

    # ---- 1) 다운로드: zip 이 없을 때만 (또는 --force) ------------------------------------
    # 187MB 를 매번 다시 받지 않도록, 이미 받은 zip 이 있으면 건너뛴다.
    if args.force or not zip_path.exists():
        print(f"[download] {cfg['dataset']['url']}")
        download(cfg["dataset"]["url"], zip_path)
    else:
        print(f"[download] 이미 존재: {zip_path}")

    # ---- 2) 압축 해제: HDFS.log 가 없을 때만 (또는 --force) ------------------------------
    if args.force or not raw_log_path(cfg).exists():
        print(f"[unzip] -> {raw_dir}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)   # zip 안의 모든 파일을 raw_dir 아래에 풀기

    # ---- 3) 검증: 이후 단계에 꼭 필요한 두 파일이 실제로 있는지 확인 ---------------------
    # 없으면 여기서 바로 오류를 내서, 나중 단계(parse)에서 알아보기 어려운 오류가 나는 것을 막는다.
    for p in (raw_log_path(cfg), label_path(cfg)):
        if not p.exists():
            raise FileNotFoundError(f"필수 파일 없음: {p}")
        # relative_to(ROOT): 긴 절대 경로 대신 "data/raw/…" 처럼 짧게 출력 / st_size: 파일 크기(바이트)
        print(f"[ok] {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")


# 이 파일을 `python -m logguard.data.download` 로 "직접 실행" 했을 때만 main() 을 부른다.
# 다른 파일에서 `import` 할 때는 실행되지 않는다 (파이썬 관용구).
if __name__ == "__main__":
    main()
