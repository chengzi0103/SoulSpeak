"""Quickly inspect STO tasks stored in DuckDB."""
from __future__ import annotations

import duckdb
import pandas as pd

DB_PATH = "/Users/hougarden/project/github/SoulSpeak/data/sto_tasks.duckdb"

con = duckdb.connect(DB_PATH)

print("=== Tasks ===")
tasks_df: pd.DataFrame = con.execute(
    """
    SELECT id, type, status, created_at, updated_at, executed_at
    FROM tasks
    ORDER BY created_at DESC
    """
).df()
print(tasks_df.to_string(index=False))

print("\n=== Task Logs ===")
logs_df: pd.DataFrame = con.execute(
    """
    SELECT task_id, event, message, timestamp, details
    FROM task_logs
    ORDER BY timestamp
    """
).df()
print(logs_df.to_string(index=False))

con.close()
print(logs_df)
