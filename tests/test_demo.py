"""시연 모드(start.sh demo) 채점 로직 검사 (2026-11-20 추가, 옥상민).

logguard.api.demo.score 는 재생이 끝난 DB 에서
  실시간 판정 = logs 에 이상 행이 있는 블록, 최종 = 실시간 OR session_results 이상
으로 샘플 블록의 정확도를 계산한다. 원본 데이터 없이 작은 가짜 DB 로 확인한다.
"""
import sqlite3

import pandas as pd

from logguard.api import demo
from logguard.config import ROOT


def test_score_counts_realtime_and_session_close(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "demo.sqlite"
    con = sqlite3.connect(db_path)
    con.executescript((ROOT / "src/logguard/api/schema.sql").read_text(encoding="utf-8"))
    # 블록 a: 실시간 이상(정답 이상) / b: 종료 판정으로만 이상(정답 이상) / c: 정상(정답 정상) / d: 실시간 오탐(정답 정상)
    for i, (block, status) in enumerate([("a", "anomaly"), ("b", "normal"), ("c", "normal"), ("d", "anomaly")]):
        con.execute("INSERT INTO logs (timestamp, log_date, log_time, level, component, block_id, event_index, status, raw_log)"
                    " VALUES ('t', '081109', '000000', 'INFO', 'x', ?, 3, ?, 'raw')", (block, status))
    con.execute("INSERT INTO session_results VALUES ('b', 'autoencoder', 't', 'manual', 2, 0.9, 0.5, 1, 'model', NULL)")
    con.commit()
    con.close()
    blocks = tmp_path / "blocks.csv"
    pd.DataFrame({"block_id": list("abcd"), "label": [1, 1, 0, 0]}).to_csv(blocks, index=False)
    monkeypatch.setattr(demo, "paths", lambda n, seed: (tmp_path / "lines.txt", blocks))
    monkeypatch.setattr(demo, "load_settings", lambda: {"database": db_path})

    result = demo.score(4, 42)
    assert (result["realtime"]["tp"], result["realtime"]["fp"], result["realtime"]["fn"]) == (1, 1, 1)
    assert (result["final"]["tp"], result["final"]["fp"], result["final"]["fn"]) == (2, 1, 0)
    assert result["close_only_tp"] == 1
    assert "세션 종료 판정" in capsys.readouterr().out
