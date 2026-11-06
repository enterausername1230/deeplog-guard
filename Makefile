# =============================================================================================
# Makefile — 자주 쓰는 명령을 짧은 이름으로 묶어 둔 파일
#
# 사용법: 터미널에서 프로젝트 폴더로 이동한 뒤  make <이름>
#   예) make setup   → 가상환경 만들고 패키지 설치
#       make data    → 데이터 다운로드 + Drain 파싱 + 분할 (처음 한 번)
#       make test    → 단위 테스트 실행
#
# 처음 받은 팀원이 할 일 (순서대로)
#   1. make setup        (uv 가 없으면 먼저: brew install uv)
#   2. make data         (다운로드 약 7분 + 파싱 약 1~2분)
#   3. make test         (15개 테스트가 모두 passed 면 준비 완료)
#
# 각 줄의 "## …" 은 그 명령이 무엇을 하는지에 대한 설명이다.
# 실험 재현 순서: data → baseline → drain-eval → deeplog/autoencoder → tune → final-eval → figures
# =============================================================================================

# 사용할 파이썬 경로 (프로젝트 가상환경). `make test PY=python3` 처럼 바꿔 쓸 수도 있다 (?= : 지정 안 했을 때만 기본값)
PY ?= .venv/bin/python

# .PHONY: 아래 이름들은 "파일 이름이 아니라 명령 이름" 이라는 표시 (같은 이름의 파일이 있어도 항상 실행됨)
.PHONY: setup download parse split baseline drain-eval export-detector deeplog autoencoder seed-study tune final-eval figures data test clean-data

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

final-eval:       ## 6단계 test 1회 평가 (시드 3개, 약 20분) → results/final_test.{json,csv}
	$(PY) -m logguard.eval.final_eval

figures:          ## 보고서 그래프 → docs/figures/*.png (final-eval 이후)
	$(PY) -m logguard.eval.figures

data: download parse split   ## 데이터 파이프라인 전체

test:
	$(PY) -m pytest -q

clean-data:       ## 파생 데이터 삭제 (원본은 유지)
	rm -rf data/interim/* data/processed/* artifacts/drain3

.PHONY: dashboard api replay
dashboard:       ## Streamlit 모니터링 대시보드
	$(PY) -m streamlit run src/logguard/dashboard/app.py

api:              ## FastAPI 백엔드 (configs/api.yaml 의 model 로 실시간 판정, 단일 워커)
	$(PY) -m uvicorn logguard.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1

replay:           ## test 블록 로그를 초당 RATE 줄로 재생하고 끝나면 세션 종료 판정 (예: make replay RATE=500)
	PYTHONPATH=src $(PY) -m logguard.api.replay data/raw/HDFS_v1/HDFS.log \
		--test-sessions data/processed/test.parquet --rate $(or $(RATE),100) --close-sessions

.PHONY: slack slack-preview
slack:             ## .env Webhook으로 미알림 이상 세션 전송 (별도 프로세스)
	PYTHONPATH=src $(PY) -m logguard.dashboard.notifier

slack-preview:     ## 외부 전송·DB 변경 없이 알림 대상 확인
	PYTHONPATH=src $(PY) -m logguard.dashboard.notifier --preview
