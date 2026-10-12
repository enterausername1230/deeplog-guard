"""Rule-Based Baseline (계획서 목표 ②의 비교 기준).

이 폴더는 무엇인가?
    딥러닝 모델이 "정말 쓸모가 있는지" 판단하려면 비교 대상이 필요하다.
    이 폴더에는 딥러닝을 쓰지 않고 사람이 정한 규칙(키워드, 길이, 빈도)만으로 이상을 잡는
    단순한 탐지기(베이스라인)가 들어 있다.

들어 있는 모듈
    rule_based.py : 베이스라인 2종을 한 번에 만들고 평가한다.
        - rule_based            : 에러 키워드 + 세션 길이 규칙 (계획서 목표 ② "+10%p" 의 비교 대상)
        - statistical_frequency : 이벤트별 등장 횟수 임계치 (통계 기법, 참고용이지만 꽤 강력함)

결과 파일
    results/baseline_rule_based.json, results/baseline_statistical_frequency.json
"""
