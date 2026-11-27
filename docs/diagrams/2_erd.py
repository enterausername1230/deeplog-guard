"""2. ERD (DB 다이어그램) — src/logguard/api/schema.sql 의 테이블 3개를 그대로 옮긴 그림.

작성: 옥상민 / 2026-11-27 (코드 기준 main a42d0e4)
근거: src/logguard/api/schema.sql (칸·제약·인덱스), api/db.py(조회 방식), dashboard/notifier.py(alerts 쓰기)
    - 실제 SQL 에 FOREIGN KEY 제약은 없다 → 관계선은 "코드가 이 칸으로 연결해서 조회한다" 는 논리적 관계
그리기: Graphviz(dot) 로 .dot 파일을 만든 뒤 PNG 로 변환 (표 모양 노드는 Graphviz 가 깔끔하다)
실행: .venv/bin/python docs/diagrams/2_erd.py  → 2_erd.dot, 2_erd.png
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
FONT = "Apple SD Gothic Neo"

# (칸 이름, 타입, 키 표시, 설명)  키: PK 기본키, UQ 유일, IX 인덱스, FK 논리적 참조
LOGS = [
    ("id", "INTEGER", "PK", "저장 순서 번호 (자동 증가)"),
    ("timestamp", "TEXT", "", "서버 수신 시각 (UTC)"),
    ("sent_at", "TEXT", "", "재생 모듈 전송 시각 (E2E 지연용)"),
    ("log_date", "TEXT", "", "원본 로그 날짜 (YYMMDD)"),
    ("log_time", "TEXT", "", "원본 로그 시각 (HHMMSS)"),
    ("level", "TEXT", "", "INFO / WARN …"),
    ("component", "TEXT", "", "로그 발생 위치 (dfs.DataNode…)"),
    ("block_id", "TEXT", "IX", "블록 ID (없으면 NULL)"),
    ("event_id", "INTEGER", "", "Drain 템플릿 번호 (처음 보는 형식 NULL)"),
    ("event_index", "INTEGER", "", "모델 입력 번호 (UNK = 1)"),
    ("model_name", "TEXT", "", "판정 모델 (autoencoder)"),
    ("anomaly_score", "REAL", "", "0~1 정규화 이상 점수"),
    ("threshold", "REAL", "", "판정 기준선 (0.5)"),
    ("is_anomaly", "INTEGER", "", "1 이상 / 0 정상"),
    ("reason", "TEXT", "", "unknown_event / model / normal"),
    ("details", "TEXT", "", "판정 근거 JSON"),
    ("latency_ms", "REAL", "", "수신 → 판정 완료 지연"),
    ("status", "TEXT", "IX", "pending_model / no_block / normal / anomaly"),
    ("raw_log", "TEXT", "", "원본 로그 한 줄"),
    ("alerted", "INTEGER", "", "알림 보냈으면 1"),
]
SESSIONS = [
    ("block_id", "TEXT", "PK", "블록 ID"),
    ("model_name", "TEXT", "PK", "판정 모델"),
    ("closed_at", "TEXT", "", "종료 판정 시각 (UTC)"),
    ("close_reason", "TEXT", "", "idle 자동 / manual 수동"),
    ("last_log_id", "INTEGER", "FK", "그 블록의 마지막 logs.id"),
    ("anomaly_score", "REAL", "", "EOS 판정 점수 (0~1)"),
    ("threshold", "REAL", "", "기준선 (0.5)"),
    ("is_anomaly", "INTEGER", "", "1 = 중간에 끊긴 세션"),
    ("reason", "TEXT", "", "판정 이유"),
    ("details", "TEXT", "", "판정 근거 JSON"),
]
ALERTS = [
    ("id", "INTEGER", "PK", "알림 번호 (자동 증가)"),
    ("timestamp", "TEXT", "IX", "알림 생성 시각 (UTC)"),
    ("block_id", "TEXT", "UQ", "블록 ID"),
    ("model_name", "TEXT", "UQ", "판정 모델"),
    ("log_id", "INTEGER", "FK", "원인 logs.id (세션 종료 원인이면 NULL)"),
    ("status", "TEXT", "", "sent / failed / pending"),
    ("message", "TEXT", "", "보낸 메시지 본문"),
    ("completed_at", "TEXT", "", "Slack 요청 완료 시각 (E2E 계산)"),
    ("error", "TEXT", "", "실패 이유"),
]
KEY_STYLE = {"PK": ("#FDE68A", "PK"), "FK": ("#DBEAFE", "FK"), "UQ": ("#EDE9FE", "UQ"), "IX": ("#F1F5F9", "IX"), "": ("white", "")}


def table(name, desc, rows, head_color, writer):
    """Graphviz HTML 표 하나 (칸마다 port 를 달아 관계선을 칸에 직접 연결)."""
    out = [f'<<table border="1" cellborder="0" cellspacing="0" cellpadding="5" color="#334155">',
           f'<tr><td colspan="4" bgcolor="{head_color}" align="left"><font point-size="15"><b>{name}</b></font>'
           f'<br/><font point-size="10" color="#334155">{desc}</font></td></tr>',
           f'<tr><td colspan="4" bgcolor="#F8FAFC" align="left"><font point-size="9.5" color="#475569">{writer}</font></td></tr>']
    for col, typ, key, note in rows:
        bg, label = KEY_STYLE[key]
        name = f"<b>{col}</b>" if key == "PK" else col
        port = col.split(" ")[0]
        # 키 표시가 없는 칸에 빈 <b></b> 를 넣으면 Graphviz 가 표 전체를 못 읽으므로 공백만 넣는다
        key_cell = f'<font point-size="9"><b>{label}</b></font>' if label else " "
        out.append(f'<tr><td bgcolor="{bg}" width="26" port="{port}_k">{key_cell}</td>'
                   f'<td align="left" port="{port}">{name}</td>'
                   f'<td align="left"><font color="#64748B" point-size="9.5">{typ}</font></td>'
                   f'<td align="left" port="{port}_r"><font color="#475569" point-size="9.5">{note}</font></td></tr>')
    out.append("</table>>")
    return "\n".join(out)


dot = f'''// 2. ERD — 자동 생성 파일 (2_erd.py 를 고치고 다시 실행)
digraph erd {{
  graph [rankdir=LR, fontname="{FONT}", fontsize=20, labelloc=t, nodesep=0.6, ranksep=1.1, pad=0.3,
         label=<<b>DeepLog-Guard DB 구조 (ERD)</b><br/><font point-size="12" color="#475569">SQLite 파일 1개 · 테이블 3개 · 쓰기: 백엔드·알림 작업자 / 읽기: 대시보드 (읽기 전용)</font><br/> >];
  node [shape=plain, fontname="{FONT}", fontsize=11];
  edge [fontname="{FONT}", fontsize=10.5, color="#334155", penwidth=1.4];

  session_results [label={table("session_results", "세션(블록) 종료 판정 결과 · 블록·모델당 1행",
                                SESSIONS, "#FEF3C7", "쓰기: 백엔드 (POST /sessions/close, 장시간 무소식 블록 자동) · 재판정 시 같은 행 갱신")}];
  logs [label={table("logs", "로그 1줄 × 블록 1개 = 1행 · 실시간 판정 결과",
                     LOGS, "#DBEAFE", "쓰기: 백엔드 (POST /logs) · 인덱스: (block_id, id), (status, id)")}];
  alerts [label={table("alerts", "Slack 알림 1건 = 1행 · UNIQUE(block_id, model_name)",
                       ALERTS, "#EDE9FE", "쓰기: Slack 알림 작업자 (notifier.py) · 인덱스: timestamp")}];

  // 관계 (FOREIGN KEY 제약 없이 코드가 이 칸으로 연결)
  session_results:last_log_id_r:e -> logs:id_k:w [dir=both, arrowtail=crow, arrowhead=tee, color="#2563EB", fontcolor="#1D4ED8",
      label="  last_log_id → logs.id\\n  N : 1  (블록의 마지막 로그)  "];
  session_results:block_id_r:e -> logs:block_id_k:w [dir=both, arrowtail=teeodot, arrowhead=crow, style=dashed, color="#D97706", fontcolor="#B45309",
      label="  block_id + model_name\\n  0..1 : N  (블록의 로그들 ↔ 종료 판정)  "];
  logs:id_r:e -> alerts:log_id_k:w [dir=both, arrowtail=teeodot, arrowhead=crowodot, color="#7C3AED", fontcolor="#6D28D9",
      label="  alerts.log_id → logs.id\\n  0..1 : 0..1  (NULL = 세션 종료 판정이 원인)  "];
  logs:block_id_r:e -> alerts:block_id_k:w [dir=both, arrowtail=crow, arrowhead=teeodot, style=dashed, color="#7C3AED", fontcolor="#6D28D9",
      label="  block_id + model_name\\n  블록·모델당 알림 최대 1건  "];

  legend [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" color="#94A3B8">
    <tr><td colspan="2" align="left"><b>표시</b></td></tr>
    <tr><td bgcolor="#FDE68A"><b>PK</b></td><td align="left">기본키</td></tr>
    <tr><td bgcolor="#DBEAFE"><b>FK</b></td><td align="left">다른 테이블의 id 를 가리킴 (논리적 참조, SQL 제약 없음)</td></tr>
    <tr><td bgcolor="#EDE9FE"><b>UQ</b></td><td align="left">함께 유일 (같은 블록 알림 중복 방지)</td></tr>
    <tr><td bgcolor="#F1F5F9"><b>IX</b></td><td align="left">조회용 인덱스</td></tr>
    <tr><td colspan="2" align="left"><font point-size="9.5" color="#475569">실선: id 참조 · 점선: block_id + model_name 으로 묶어 조회<br/>최종 세션 판정 = logs 에 이상 행이 하나라도 있음 OR session_results.is_anomaly = 1</font></td></tr>
  </table>>];
  {{ rank=same; session_results; legend; }}
  session_results -> legend [style=invis];
}}
'''
(HERE / "2_erd.dot").write_text(dot, encoding="utf-8")
subprocess.run(["dot", "-Tpng", "-Gdpi=200", str(HERE / "2_erd.dot"), "-o", str(HERE / "2_erd.png")], check=True)
print("saved", HERE / "2_erd.png")
