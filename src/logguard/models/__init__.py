"""이상 탐지 모델 (담당: 옥상민).

이 폴더는 무엇인가?
    딥러닝 이상 탐지 모델 2종(DeepLog, Autoencoder)과, 두 모델이 함께 쓰는 학습·평가·추론 도구가 들어 있다.

모듈 지도 (역할별)
    [데이터 준비]
      dataset.py     : 세션(이벤트 시퀀스) → 길이 w 의 "윈도우" 로 자르고, 똑같은 윈도우는 하나로 합친다
    [모델]
      deeplog.py     : LSTM 으로 "다음 이벤트" 를 예측. 실제 다음 이벤트가 예상 밖이면 이상
      autoencoder.py : 윈도우를 압축했다가 복원. 복원이 잘 안 되면(=처음 보는 패턴) 이상  ← 최종 채택
    [공통 도구]
      train_utils.py : 시드 고정, CPU/GPU 선택, 가중치 학습 루프
      scoring.py     : 윈도우 점수 → 세션 점수, 임계치 고르기, 0~1 정규화, 결과 저장
      inference.py   : 백엔드(FastAPI)가 쓰는 공통 인터페이스 Detector 와 load_detector()
    [실험 관리]
      seed_study.py  : 랜덤 시드를 바꿔 가며 성능이 얼마나 흔들리는지 측정
      tune.py        : 하이퍼파라미터 조합 × 시드 그리드 탐색 (5단계)

모든 모델이 지키는 공통 규칙 (docs/plan_week4-6.md)
    1. 학습은 train 의 "정상" 세션만 사용한다 (이상이 뭔지 모르는 상태에서 정상 패턴만 배운다)
    2. 임계치는 val 의 정상 세션 점수 백분위수로 정한다 (logguard.eval.metrics.percentile_threshold)
    3. 세션 점수 = 세션 안 윈도우 점수 중 최댓값. 학습 때 못 본 이벤트(UNK)가 있으면 즉시 이상
    4. test 는 맨 마지막(6단계)에 단 한 번만 평가한다 (eval/final_eval.py)
    5. 결과는 results/<모델이름>.json 으로 저장해 베이스라인과 같은 형식으로 비교한다
"""
