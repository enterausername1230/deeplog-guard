#!/usr/bin/env bash
# =============================================================================================
# start.sh — DeepLog-Guard 전체 시스템을 한 번에 실행하는 스크립트
#
#   켜지는 것: ① 백엔드 FastAPI 서버 (Autoencoder 실시간 판정, http://127.0.0.1:8000)
#             ② Streamlit 대시보드     (http://localhost:8501, 브라우저 자동으로 열림)
#             ③ 로그 재생              (test 블록 로그를 서버로 보냄)
#   Ctrl+C 한 번이면 위 프로그램이 모두 함께 꺼진다.
#
# 사용법 (프로젝트 폴더에서)
#   ./start.sh            빠른 시연: 로그 2만 줄 재생 (약 1분). 끝나도 서버·대시보드는 계속 켜져 있음
#   ./start.sh full       전체 test 재생 (223만 줄, 약 1시간) + 끝나면 세션 종료 판정 → 최종 F1 0.977 재현
#   ./start.sh serve      재생 없이 서버·대시보드만 (http://127.0.0.1:8000/docs 에서 직접 로그를 보내 볼 때)
#
# 옵션 (환경변수, 명령 앞에 붙인다)
#   LIMIT=50000 ./start.sh     빠른 시연에서 보낼 줄 수 (기본 20000)
#   RATE=300 ./start.sh        초당 전송 줄 수 상한 (기본 1000, 서버 처리 한계가 약 660줄/초)
#   KEEP_DB=1 ./start.sh       이전 실행의 DB 를 지우지 않고 이어서 쓰기 (기본: 매번 새 DB)
#   NO_BROWSER=1 ./start.sh    브라우저 자동 열기 끄기
#
# 실행 기록(로그 파일): artifacts/run_logs/{api,dashboard,replay}.log
# =============================================================================================

# -e: 명령이 실패하면 즉시 중단 / -u: 정의 안 된 변수를 쓰면 오류 / -o pipefail: 파이프 중간 실패도 실패로 처리
set -euo pipefail

# ── 0. 기본 설정 ─────────────────────────────────────────────────────────────────────────────
# 이 스크립트가 있는 폴더(= 프로젝트 루트)로 이동. 어느 위치에서 실행해도 같은 결과가 나오게 한다
cd "$(dirname "$0")"

MODE="${1:-quick}"                       # 첫 번째 인자. 없으면 quick
LIMIT="${LIMIT:-20000}"                  # ${변수:-기본값}: 변수가 비어 있으면 기본값 사용
RATE="${RATE:-1000}"
PY=".venv/bin/python"                    # 프로젝트 가상환경의 파이썬 (make setup 으로 생성)
API_PORT=8000
DASH_PORT=8501
DB="artifacts/api/logguard.sqlite"       # configs/api.yaml 의 database 와 같은 파일
LOG_DIR="artifacts/run_logs"
PIDS=()                                  # 백그라운드로 띄운 프로그램 번호(PID) 목록 → 종료 시 한꺼번에 끈다

# 화면 출력용 작은 함수들
say()  { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }      # 파란색 안내
warn() { printf '\033[1;33m[주의]\033[0m %s\n' "$*"; }       # 노란색 주의
die()  { printf '\033[1;31m[오류]\033[0m %s\n' "$*"; exit 1; }  # 빨간색 오류 후 종료

case "$MODE" in
  quick|full|serve) ;;
  -h|--help|help) sed -n '2,24p' "$0"; exit 0 ;;           # 위 설명 부분을 그대로 보여 줌
  *) die "알 수 없는 모드: $MODE  (quick / full / serve 중 하나)" ;;
esac

# ── 1. 종료 처리: Ctrl+C 를 누르거나 오류로 끝나도 띄운 프로그램을 모두 끈다 ─────────────────────
cleanup() {
  trap - EXIT INT TERM                   # 정리 도중 다시 불리지 않도록 해제
  echo
  say "종료 중..."
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  say "모두 종료했습니다. (DB: $DB, 실행 기록: $LOG_DIR/)"
}
trap cleanup EXIT INT TERM

# ── 2. 사전 확인 ─────────────────────────────────────────────────────────────────────────────
[ -x "$PY" ] || die "가상환경이 없습니다. 먼저 실행: make setup"
[ -f artifacts/models/autoencoder.pt ] || die "모델 파일이 없습니다: artifacts/models/autoencoder.pt"
if [ "$MODE" != "serve" ]; then
  [ -f data/raw/HDFS_v1/HDFS.log ] && [ -f data/processed/test.parquet ] \
    || die "재생할 데이터가 없습니다. 먼저 실행: make data  (또는 ./start.sh serve 로 서버만)"
fi
# 포트가 이미 쓰이고 있으면(예: 이전 실행이 남아 있음) 알려 주고 멈춘다
for port in "$API_PORT" "$DASH_PORT"; do
  if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    die "포트 $port 이(가) 이미 사용 중입니다. 이전 실행을 끄거나: lsof -iTCP:$port -sTCP:LISTEN"
  fi
done

mkdir -p "$LOG_DIR" "$(dirname "$DB")"

# ── 3. DB 준비: 기본은 매번 새 DB (이전 재생 결과와 섞이면 세션 판정이 오염됨) ─────────────────────
if [ "${KEEP_DB:-0}" = "1" ]; then
  say "기존 DB 를 이어서 사용합니다: $DB"
else
  rm -f "$DB" "$DB-wal" "$DB-shm"
  say "새 DB 로 시작합니다: $DB"
fi

# ── 4. 백엔드 서버 실행 (백그라운드) ───────────────────────────────────────────────────────────
say "① 백엔드 서버 시작 (Autoencoder 로딩)..."
# --no-access-log: 요청마다 남는 접속 기록을 끈다 (전체 재생이면 약 140MB). 오류·시작 기록은 그대로 남는다
"$PY" -m uvicorn logguard.api.main:app --app-dir src --host 127.0.0.1 --port "$API_PORT" --workers 1 \
  --no-access-log > "$LOG_DIR/api.log" 2>&1 &
PIDS+=($!)                               # $! = 방금 백그라운드로 띄운 프로그램의 번호

# /health 가 응답할 때까지 최대 60초 기다린다 (모델 로딩에 몇 초 걸림)
for _ in $(seq 1 60); do
  if curl -s -m 1 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then break; fi
  # 서버 프로그램이 이미 죽었으면(설정 오류 등) 기다리지 않고 기록을 보여 준다
  kill -0 "${PIDS[0]}" 2>/dev/null || { tail -20 "$LOG_DIR/api.log"; die "백엔드 서버가 시작하지 못했습니다 (위 기록 참고)"; }
  sleep 1
done
HEALTH="$(curl -s -m 2 "http://127.0.0.1:$API_PORT/health" || true)"
[ -n "$HEALTH" ] || { tail -20 "$LOG_DIR/api.log"; die "백엔드 서버 응답 없음 (위 기록 참고)"; }
say "   준비 완료: $HEALTH"
say "   API 화면: http://127.0.0.1:$API_PORT/docs"

# ── 5. 대시보드 실행 (백그라운드) ─────────────────────────────────────────────────────────────
say "② 대시보드 시작..."
# --server.headless true: 첫 실행 때 묻는 이메일 입력을 건너뛴다 (브라우저는 아래에서 직접 연다)
"$PY" -m streamlit run src/logguard/dashboard/app.py --server.port "$DASH_PORT" --server.headless true \
  > "$LOG_DIR/dashboard.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do
  curl -s -m 1 "http://localhost:$DASH_PORT" >/dev/null 2>&1 && break
  sleep 1
done
say "   대시보드: http://localhost:$DASH_PORT  →  사이드바에서 데이터 소스를 'SQLite' 로 선택하세요"
if [ "${NO_BROWSER:-0}" != "1" ] && command -v open >/dev/null 2>&1; then
  open "http://localhost:$DASH_PORT"     # macOS: 기본 브라우저로 열기
fi

# ── 6. 로그 재생 ─────────────────────────────────────────────────────────────────────────────
# 서버 상태(/stats)를 몇 초마다 한 줄씩 보여 주는 함수
show_stats() {
  # (f-string 안에 따옴표를 넣지 않도록 값을 먼저 변수에 꺼낸다 — Python 3.11 은 f-string 안의 \" 를 허용하지 않음)
  curl -s -m 2 "http://127.0.0.1:$API_PORT/stats" | "$PY" -c '
import json, sys
s = json.load(sys.stdin); b = s["by_status"]
total, normal, anomaly = s["total_records"], b.get("normal", 0), b.get("anomaly", 0)
closed, closed_anom = s["sessions_closed"], s["sessions_closed_anomaly"]
print(f"   저장 {total:>9,}행 | 정상 {normal:>9,} | 이상 {anomaly:>6,} | 종료 판정 세션 {closed:,} (이상 {closed_anom:,})")' || true
}

if [ "$MODE" = "serve" ]; then
  say "③ 재생 없이 대기합니다. http://127.0.0.1:$API_PORT/docs 의 POST /logs 로 직접 로그를 보내 보세요."
else
  REPLAY_ARGS=(data/raw/HDFS_v1/HDFS.log --url "http://127.0.0.1:$API_PORT/logs"
               --test-sessions data/processed/test.parquet --rate "$RATE")
  if [ "$MODE" = "quick" ]; then
    # 빠른 시연에서는 세션 종료 판정을 하지 않는다: 중간에서 끊었으므로 진행 중인 정상 세션까지
    # "끊긴 세션(이상)" 으로 판정되기 때문 (끝까지 재생하는 full 모드에서만 의미가 있음)
    REPLAY_ARGS+=(--limit "$LIMIT")
    say "③ 빠른 시연: test 로그 ${LIMIT}줄을 초당 최대 ${RATE}줄로 재생합니다."
  else
    REPLAY_ARGS+=(--close-sessions)
    say "③ 전체 test 재생: 약 223만 줄 (초당 최대 ${RATE}줄, 약 1시간). 끝나면 세션 종료 판정까지 합니다."
  fi
  PYTHONPATH=src "$PY" -m logguard.api.replay "${REPLAY_ARGS[@]}" > "$LOG_DIR/replay.log" 2>&1 &
  REPLAY_PID=$!
  PIDS+=("$REPLAY_PID")
  # 재생이 끝날 때까지 5초마다 진행 상황 출력
  while kill -0 "$REPLAY_PID" 2>/dev/null; do
    show_stats
    sleep 5
  done
  if wait "$REPLAY_PID"; then
    say "재생 완료: $(tail -2 "$LOG_DIR/replay.log" | tr '\n' ' ')"
  else
    warn "재생 중 오류가 났습니다 (아래 기록). 서버·대시보드는 계속 켜 둡니다."
    tail -5 "$LOG_DIR/replay.log"
  fi
  show_stats
fi

say "서버와 대시보드는 계속 켜져 있습니다. 끝내려면 Ctrl+C"
# 백엔드 서버가 살아 있는 동안 기다린다 (Ctrl+C → cleanup 이 모두 종료)
wait "${PIDS[0]}"
