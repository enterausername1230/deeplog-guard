"""SQLite storage shared with read-only dashboard consumers.

이 파일은 무엇인가? (원작성: 안명근 / 2026-10-20 통합: 옥상민)
    백엔드가 판정 결과를 SQLite 파일 하나에 저장하고 꺼내는 코드를 모은 곳이다.
    대시보드는 같은 파일을 "읽기 전용" 으로 열어 화면에 보여 준다.

테이블 구조는 이 코드 안에 직접 쓰지 않고 같은 폴더의 schema.sql 한 곳에서만 정의한다.
    → 백엔드(쓰기)와 대시보드(읽기, 테스트)가 같은 파일을 보므로 칸 이름이 어긋날 일이 없다.
    → 칸 설명은 schema.sql 과 docs/db_schema.md 참고.

SQLite 란?
    서버 프로그램 없이 파일 하나(.sqlite)로 동작하는 가벼운 데이터베이스. 파이썬에 기본 내장(sqlite3)되어 있다.
"""
from contextlib import contextmanager
import sqlite3
from pathlib import Path

# 테이블 정의 파일 (이 파일과 같은 폴더). 대시보드 테스트도 이 경로를 읽는다
SCHEMA_PATH = Path(__file__).with_name('schema.sql')


class Database:
    """SQLite 파일 하나를 감싼 저장소. 요청마다 connect() 로 연결을 열고 닫는다."""

    def __init__(self, path: Path, timeout: float = 10):
        """
        Args:
            path   : DB 파일 경로 (configs/api.yaml 의 database, 기본 artifacts/api/logguard.sqlite)
            timeout: 다른 연결이 쓰는 중일 때 기다리는 최대 초 (대시보드가 읽는 중이어도 오류 대신 잠깐 대기)
        """
        self.path, self.timeout = path, timeout

    @contextmanager
    def connect(self):
        """`with db.connect() as con:` 형태로 쓰는 연결.

        블록이 오류 없이 끝나면 자동 commit(저장 확정), 오류가 나면 자동 rollback(전부 취소) 된다.
        → 한 요청에서 여러 블록 행을 저장하다 중간에 실패하면 아무것도 저장되지 않는다 (하나의 트랜잭션).
        """
        con = sqlite3.connect(self.path, timeout=self.timeout)
        con.row_factory = sqlite3.Row          # 결과 행을 row["칸이름"] 처럼 이름으로 꺼낼 수 있게
        try:
            with con:                          # sqlite3 연결의 with: 성공 시 commit, 예외 시 rollback
                yield con
        finally:
            con.close()

    def initialize(self):
        """서버 시작 시 1회: 폴더를 만들고 schema.sql 을 실행해 테이블이 없으면 만든다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            # WAL 모드: 백엔드가 쓰는 동안에도 대시보드가 동시에 읽을 수 있게 해 주는 SQLite 설정
            con.execute('PRAGMA journal_mode=WAL')
            con.executescript(SCHEMA_PATH.read_text(encoding='utf-8'))

    # ------------------------------------------------------------------ 실시간 판정용

    @staticmethod
    def history(con, block_id, limit):
        """블록의 최근 limit 개 event_index 를 "오래된 것 → 최신" 순서로 돌려준다 (모델 입력 만들기용).

        DB 에서 읽기 때문에 서버를 껐다 켜도 블록별 이력이 이어진다.
        limit = configs/api.yaml 의 window_size (모델 윈도우 크기 20 이상이어야 오프라인 평가와 같은 판정이 나온다)
        """
        rows = con.execute(
            'SELECT event_index FROM logs WHERE block_id = ? ORDER BY id DESC LIMIT ?',
            (block_id, limit),
        ).fetchall()
        return [row[0] for row in reversed(rows)]      # DESC 로 최신부터 가져왔으니 뒤집어서 시간순으로

    @staticmethod
    def insert(con, record):
        """dict 한 개를 logs 에 한 행으로 저장하고, 저장된 행(자동 부여된 id 포함)을 돌려준다.

        record 의 키 = 칸 이름. 키에 없는 칸은 schema.sql 의 기본값(대부분 NULL)이 들어간다.
        """
        columns = ', '.join(record)
        placeholders = ', '.join('?' for _ in record)  # 값은 ? 자리표시자로 넘긴다 (SQL 인젝션 방지)
        cursor = con.execute(f'INSERT INTO logs ({columns}) VALUES ({placeholders})', tuple(record.values()))
        return dict(con.execute('SELECT * FROM logs WHERE id = ?', (cursor.lastrowid,)).fetchone())

    # ------------------------------------------------------------------ 세션 종료 판정용 (2026-10-20 추가)

    @staticmethod
    def sessions_to_close(con, model_name, max_last_id=None):
        """아직 종료 판정을 받지 않은 블록 목록 [(block_id, 마지막 logs.id), ...] 을 돌려준다.

        Args:
            model_name : 이 모델로 아직 종료 판정이 안 된 블록만 고른다
            max_last_id: 주면 "마지막 로그 id 가 이 값 이하인 블록" 만 = 그 뒤로 오래 소식이 없는 블록 (idle 종료)
                         None 이면 열린 블록 전부 (재생이 끝났을 때 수동 종료)
        """
        sql = ('SELECT block_id, MAX(id) AS last_id FROM logs '
               'WHERE block_id IS NOT NULL AND block_id NOT IN '
               '(SELECT block_id FROM session_results WHERE model_name = ?) '
               'GROUP BY block_id')
        args = [model_name]
        if max_last_id is not None:
            sql += ' HAVING MAX(id) <= ?'          # HAVING: GROUP BY 로 묶은 결과에 거는 조건
            args.append(max_last_id)
        return [(row['block_id'], row['last_id']) for row in con.execute(sql, args)]

    @staticmethod
    def insert_session_result(con, row):
        """session_results 에 종료 판정 1행 저장 (같은 블록·모델이 이미 있으면 무시 → 중복 판정 방지)."""
        columns = ', '.join(row)
        placeholders = ', '.join('?' for _ in row)
        con.execute(f'INSERT OR IGNORE INTO session_results ({columns}) VALUES ({placeholders})', tuple(row.values()))

    @staticmethod
    def max_id(con):
        """지금까지 저장된 가장 큰 logs.id (= 서버가 받은 행 수). 세션 idle 판정의 "현재 시각" 역할."""
        return con.execute('SELECT COALESCE(MAX(id), 0) FROM logs').fetchone()[0]

    # ------------------------------------------------------------------ 조회 API 용

    def list_logs(self, after_id=0, limit=100, block_id=None, status=None):
        """GET /logs: id 가 after_id 보다 큰 행을 오름차순으로 최대 limit 개 (블록·상태로 거르기 가능)."""
        filters, args = ['id > ?'], [after_id]
        for key, value in [('block_id', block_id), ('status', status)]:
            if value is not None:
                filters.append(f'{key} = ?')
                args.append(value)
        args.append(limit)
        with self.connect() as con:
            return [dict(row) for row in con.execute(
                f"SELECT * FROM logs WHERE {' AND '.join(filters)} ORDER BY id LIMIT ?", args
            )]

    def list_sessions(self, limit=100, anomalies_only=False):
        """GET /sessions: 종료 판정이 끝난 세션 목록 (최근 종료 순)."""
        sql = 'SELECT * FROM session_results'
        if anomalies_only:
            sql += ' WHERE is_anomaly = 1'
        with self.connect() as con:
            return [dict(row) for row in con.execute(sql + ' ORDER BY closed_at DESC LIMIT ?', (limit,))]

    def stats(self):
        """GET /stats: 상태별 행 수 + 종료 판정된 세션 수."""
        with self.connect() as con:
            counts = dict(con.execute('SELECT status, COUNT(*) FROM logs GROUP BY status').fetchall())
            closed = dict(con.execute(
                'SELECT is_anomaly, COUNT(*) FROM session_results GROUP BY is_anomaly').fetchall())
        return {'total_records': sum(counts.values()), 'by_status': counts,
                'sessions_closed': sum(closed.values()), 'sessions_closed_anomaly': closed.get(1, 0)}
