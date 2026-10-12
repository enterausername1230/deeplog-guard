PY ?= .venv/bin/python

.PHONY: setup download parse split baseline drain-eval export-detector deeplog autoencoder seed-study tune data test clean-data

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

drain-eval:       ## Drain 검증(공식 정답 대비)·sim_th/depth 튜닝 → results/drain_*.{csv,json}
	$(PY) -m logguard.data.drain_tuning

export-detector:  ## 백엔드용 Detector 내보내기 + test 재현 검증 → artifacts/models/
	$(PY) -m logguard.models.inference --export statistical_frequency

deeplog:          ## DeepLog 학습·val 평가 + EOS 제거 비교 → results/deeplog*.json, artifacts/models/deeplog.pt
	$(PY) -m logguard.models.deeplog
	$(PY) -m logguard.models.deeplog --no-eos

autoencoder:      ## Autoencoder 학습·val 평가 + EOS 제거 비교 → results/autoencoder*.json, artifacts/models/autoencoder.pt
	$(PY) -m logguard.models.autoencoder
	$(PY) -m logguard.models.autoencoder --no-eos

seed-study:       ## 시드 42·43·44 val 편차 → results/seed_study_*.csv
	$(PY) -m logguard.models.seed_study --model deeplog
	$(PY) -m logguard.models.seed_study --model autoencoder

tune:             ## 5단계 튜닝 (조합 × 시드 3개, 약 45분) → results/tuning_*.csv
	$(PY) -m logguard.models.tune --model deeplog
	$(PY) -m logguard.models.tune --model autoencoder

data: download parse split   ## 데이터 파이프라인 전체

test:
	$(PY) -m pytest -q

clean-data:       ## 파생 데이터 삭제 (원본은 유지)
	rm -rf data/interim/* data/processed/* artifacts/drain3
