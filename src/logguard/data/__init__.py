"""데이터 수집·파싱·분할·윈도우 생성 (담당: 옥상민).

이 폴더는 무엇인가?
    서버 로그(사람이 읽는 문장)를 딥러닝 모델이 이해할 수 있는 "숫자 시퀀스" 로 바꾸는 모든 단계를 담당한다.

    원본 로그 한 줄 예시:
        081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: ...
    → Drain 파싱 후:  Event ID = 1  ("Receiving block <BLK> src: <IP> dest: <IP>" 라는 템플릿 번호)
    → 블록별로 모으면: blk_-1608999687919862906 의 이벤트 시퀀스 = [1, 2, 1, 1, 3, 4, ...]
    → 모델 입력 번호로 바꾸면: [3, 4, 3, 3, 5, 6, ...] (+ 끝에 EOS)

실행 순서 (Makefile 의 `make data` 가 이 순서로 실행)
    1. download.py : HDFS 데이터셋(약 187MB zip) 다운로드·압축 해제
    2. parse.py    : Drain3 로 1,117만 줄을 Event ID 로 바꾸고 블록(세션) 단위로 묶기
    3. split.py    : 블록 단위로 train / val / test 분할 + 모델용 어휘(vocab) 만들기

보조 모듈
    windows.py      : 특수 토큰(PAD, UNK, EOS) 정의, Event ID → 모델 입력 번호 변환 함수
    drain_tuning.py : Drain 설정이 공식 정답과 잘 맞는지 검증하고 sim_th·depth 를 튜닝
    online.py       : 실시간 서버(FastAPI)에서 로그 1줄을 즉시 Event ID 로 바꾸는 파서
"""
