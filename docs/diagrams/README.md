# 보고서용 다이어그램

작성: 옥상민 / 2026-11-27 (코드 기준 main `a42d0e4`). 최종 보고서(작성: 이동준)에 넣을 그림이다.
모든 수치와 이름은 코드·결과 파일에서 가져왔다. 각 `.py` 파일 맨 위에 근거 파일을 적어 두었다.

| 파일 | 내용 | 보고서 위치 |
|---|---|---|
| `1_system_architecture.png` | 시스템 구성도: 로그 발생원 → 백엔드 → SQLite → Slack 알림 · 대시보드 | 시스템 설계 |
| `2_erd.png` | DB 구조(ERD): `logs` · `session_results` · `alerts` 의 칸 전체와 관계 | DB 설계 |
| `3_usecase.png` | 유스케이스: 액터 4종(로그 수집기 · 개발자 · 운영자 · Slack), 유스케이스 24개 | 요구사항 분석 |
| `4_pipeline.png` | 학습 · 평가 파이프라인 12단계 (데이터 준비 → 학습 → test 1회 평가 → 배포) | 모델 개발 과정 |
| `5_model.png` | Autoencoder 구조(440→64→32→64→440)와 판정 흐름 (UNK · 임계치 6.78 · EOS) | 모델 설계 |

## 다시 그리기
코드나 수치가 바뀌면 그림 파일을 고치지 말고 `.py` 를 고친 뒤 다시 실행한다.

```bash
cd docs/diagrams
for f in 1_system_architecture.py 2_erd.py 3_usecase.py 4_pipeline.py 5_model.py; do ../../.venv/bin/python $f; done
```

- `_draw.py`: 공통 도구 (색 · 박스 · 화살표). 다섯 그림이 같은 색을 같은 의미로 쓴다
  (회색 = 로그 발생원, 파랑 = 백엔드, 노랑 = 저장소, 보라 = Slack 알림, 청록 = 대시보드, 초록 = 모델 · 학습)
- 1 · 3 · 4 · 5번은 matplotlib 으로 좌표를 직접 지정해 그린다. 2번(ERD)은 Graphviz(`dot`)로 그린다
  (`brew install graphviz` 필요, `2_erd.dot` 은 실행 때 자동으로 생기는 중간 파일)
- 글꼴: macOS 기본 `Apple SD Gothic Neo`. 다른 OS 에서는 `_draw.py` 의 `FONT` 를 바꾼다
