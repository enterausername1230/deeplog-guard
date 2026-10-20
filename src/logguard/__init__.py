"""7조 DeepLog-Guard: AI 기반 실시간 서버 로그 이상 탐지 및 알림 시스템.

이 파일은 무엇인가?
    `logguard` 라는 파이썬 패키지(=폴더 묶음)의 "입구" 파일이다.
    폴더 안에 `__init__.py` 가 있으면 파이썬은 그 폴더를 패키지로 인식하고,
    `import logguard.data.parse` 처럼 점(.)으로 하위 모듈을 불러올 수 있게 된다.
    이 파일 자체에는 실행되는 코드가 없고, 패키지 전체를 소개하는 설명만 들어 있다.

패키지 전체 구조 (데이터가 흘러가는 순서대로)
    logguard/
    ├── config.py          ① 모든 코드가 공통으로 쓰는 설정(경로, 하이퍼파라미터) 불러오기
    ├── data/              ② 원본 로그 → 모델이 먹을 수 있는 숫자 시퀀스로 바꾸는 단계
    │   ├── download.py        HDFS 데이터셋 내려받기
    │   ├── parse.py           Drain3 로 로그 한 줄 한 줄을 "이벤트 번호(Event ID)" 로 바꾸기
    │   ├── drain_tuning.py    Drain 파싱이 공식 정답과 얼마나 맞는지 검증·튜닝
    │   ├── split.py           블록(세션) 단위로 train / val / test 나누기
    │   ├── windows.py         특수 토큰(PAD/UNK/EOS), 이벤트 번호 → 모델 입력 번호 변환
    │   └── online.py          실시간 서버용: 로그 1줄 → Event ID (백엔드가 사용)
    ├── baseline/          ③ 딥러닝과 비교할 "규칙 기반" 탐지기 2종
    ├── models/            ④ 딥러닝 모델 (DeepLog, Autoencoder) 과 학습·평가·추론 도구
    ├── eval/              ⑤ 평가 지표(Precision/Recall/F1), 최종 평가, 보고서 그래프
    ├── api/               ⑥ (안명근 담당) FastAPI 실시간 추론 서버 — 예정
    └── dashboard/         ⑦ (이동준 담당) Streamlit 대시보드 + Slack 알림 — 예정

처음 보는 사람이 읽는 순서 추천
    config.py → data/parse.py → data/split.py → baseline/rule_based.py
    → models/dataset.py → models/autoencoder.py → models/inference.py
"""
