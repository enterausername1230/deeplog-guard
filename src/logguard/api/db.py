"""SQLite storage shared with read-only dashboard consumers."""
from contextlib import contextmanager
import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: Path, timeout: float = 10):
        self.path, self.timeout = path, timeout

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=self.timeout)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.execute('PRAGMA journal_mode=WAL')
            con.executescript('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    log_date TEXT NOT NULL,
                    log_time TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    block_id TEXT,
                    event_id INTEGER,
                    event_index INTEGER NOT NULL,
                    anomaly_score REAL,
                    is_anomaly INTEGER,
                    status TEXT NOT NULL,
                    raw_log TEXT NOT NULL,
                    alerted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS logs_block_id_id ON logs(block_id, id);
                CREATE INDEX IF NOT EXISTS logs_status_id ON logs(status, id);
            ''')

    @staticmethod
    def history(con, block_id, limit):
        rows = con.execute(
            'SELECT event_index FROM logs WHERE block_id = ? ORDER BY id DESC LIMIT ?',
            (block_id, limit),
        ).fetchall()
        return [row[0] for row in reversed(rows)]

    @staticmethod
    def insert(con, record):
        columns = ', '.join(record)
        placeholders = ', '.join('?' for _ in record)
        cursor = con.execute(f'INSERT INTO logs ({columns}) VALUES ({placeholders})', tuple(record.values()))
        return dict(con.execute('SELECT * FROM logs WHERE id = ?', (cursor.lastrowid,)).fetchone())

    def list_logs(self, after_id=0, limit=100, block_id=None, status=None):
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

    def stats(self):
        with self.connect() as con:
            counts = dict(con.execute('SELECT status, COUNT(*) FROM logs GROUP BY status').fetchall())
        return {'total_records': sum(counts.values()), 'by_status': counts}
