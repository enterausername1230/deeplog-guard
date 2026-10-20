"""평가 지표 (Precision / Recall / F1, 임계치 산출).

이 폴더는 무엇인가?
    "모델이 얼마나 잘 맞히는가" 를 숫자로 계산하고, 최종 결과를 정리하는 코드가 들어 있다.

들어 있는 모듈
    metrics.py    : Precision / Recall / F1 계산, 백분위수 임계치 계산 (모든 모델이 공통으로 사용)
    final_eval.py : 6단계 최종 평가. test 데이터로 딱 한 번 평가하고 계획서 목표 달성 여부를 정리
    figures.py    : 보고서용 그래프(F1 비교, PR 곡선, 점수 분포, EOS 효과) 그리기

용어 빠른 설명 (이상 = "양성", 정상 = "음성" 으로 본다)
    TP (True Positive)  : 진짜 이상을 이상이라고 맞힘
    FP (False Positive) : 정상인데 이상이라고 잘못 경보 (오탐)
    FN (False Negative) : 이상인데 정상이라고 놓침 (미탐)
    TN (True Negative)  : 정상을 정상이라고 맞힘
    Precision = TP / (TP + FP) : "이상이라고 한 것 중 진짜 이상 비율" → 높을수록 헛경보가 적다
    Recall    = TP / (TP + FN) : "진짜 이상 중 잡아낸 비율"          → 높을수록 놓치는 게 적다
    F1        = 둘의 조화평균  : 두 지표를 하나로 합친 종합 점수 (이 프로젝트의 주 지표)
"""
