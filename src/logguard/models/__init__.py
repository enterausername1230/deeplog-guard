"""이상 탐지 모델 (담당: 옥상민).

예정 모듈
  autoencoder.py : 윈도우 재구성 오차 기반 (logguard.data.windows.sliding_windows 사용)
  deeplog.py     : LSTM 다음 이벤트 top-k 예측 (logguard.data.windows.next_event_windows 사용)
공통 규칙: 학습은 train(정상), 임계치는 val 정상 세션 스코어 백분위수(logguard.eval.metrics.percentile_threshold),
최종 평가는 test 세션 단위 prf(). 결과는 results/<model>.json 으로 저장해 baseline 과 비교.
"""
