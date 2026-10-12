PY ?= .venv/bin/python

.PHONY: setup download parse split baseline data test clean-data

setup:            ## 가상환경 + 의존성 설치 (uv 필요)
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e .

download:         ## HDFS_v1 다운로드·압축 해제 (~187MB zip, ~1.5GB 해제)
	$(PY) -m logguard.data.download

parse:            ## Drain3 파싱 → data/interim
	$(PY) -m logguard.data.parse

split:            ## Block ID 세션 분할 → data/processed
	$(PY) -m logguard.data.split

baseline:         ## 베이스라인 2종 → results/baseline_*.json
	$(PY) -m logguard.baseline.rule_based

data: download parse split   ## 데이터 파이프라인 전체

test:
	$(PY) -m pytest -q

clean-data:       ## 파생 데이터 삭제 (원본은 유지)
	rm -rf data/interim/* data/processed/* artifacts/drain3
