"""DuckDB-backed TaskStore implementation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import duckdb

from soul_speak.sto.models import Task, TaskLog, TaskStatus


class DuckDBTaskStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            base = Path(__file__).resolve().parents[2] / "data"
            base.mkdir(parents=True, exist_ok=True)
            db_path = base / "sto_tasks.duckdb"
        self.db_path = Path(db_path)
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                type TEXT,
                status TEXT,
                manual_required BOOLEAN,
                payload TEXT,
                scheduled_for TIMESTAMP,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                executed_at TIMESTAMP,
                result TEXT,
                error TEXT,
                attempts INTEGER,
                confirmation_note TEXT
            )
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS task_logs (
                task_id TEXT,
                event TEXT,
                message TEXT,
                timestamp TIMESTAMP
            )
            """
        )

    # ---- helpers ----
    @staticmethod
    def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    @staticmethod
    def _str_to_dt(value: Optional[str]) -> Optional[datetime]:
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _status_from_str(value: str) -> TaskStatus:
        return TaskStatus(value)

    @staticmethod
    def _json_dump(data: Optional[dict]) -> Optional[str]:
        return json.dumps(data, ensure_ascii=False) if data is not None else None

    @staticmethod
    def _json_load(text: Optional[str]) -> Optional[dict]:
        return json.loads(text) if text else None

    # ---- CRUD ----
    def create_task(self, task: Task) -> None:
        self.con.execute(
            """
            INSERT INTO tasks (
                id, type, status, manual_required, payload, scheduled_for,
                created_at, updated_at, executed_at, result, error, attempts, confirmation_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.id,
                task.type,
                task.status.value,
                task.manual_required,
                self._json_dump(task.payload),
                self._dt_to_str(task.scheduled_for),
                self._dt_to_str(task.created_at),
                self._dt_to_str(task.updated_at),
                self._dt_to_str(task.executed_at),
                self._json_dump(task.result),
                self._json_dump(task.error),
                task.attempts,
                task.confirmation_note,
            ),
        )
        self.append_log(TaskLog(task_id=task.id, event="created", message=json.dumps(task.payload, ensure_ascii=False)))

    def get_task(self, task_id: str) -> Optional[Task]:
        cur = self.con.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(cur) if cur else None

    def list_tasks(self) -> List[Task]:
        rows = self.con.execute("SELECT * FROM tasks").fetchall()
        return [self._row_to_task(row) for row in rows]

    def update_task(self, task: Task) -> None:
        task.updated_at = datetime.utcnow()
        self.con.execute(
            """
            UPDATE tasks SET
                type = ?,
                status = ?,
                manual_required = ?,
                payload = ?,
                scheduled_for = ?,
                created_at = ?,
                updated_at = ?,
                executed_at = ?,
                result = ?,
                error = ?,
                attempts = ?,
                confirmation_note = ?
            WHERE id = ?
            """,
            (
                task.type,
                task.status.value,
                task.manual_required,
                self._json_dump(task.payload),
                self._dt_to_str(task.scheduled_for),
                self._dt_to_str(task.created_at),
                self._dt_to_str(task.updated_at),
                self._dt_to_str(task.executed_at),
                self._json_dump(task.result),
                self._json_dump(task.error),
                task.attempts,
                task.confirmation_note,
                task.id,
            ),
        )

    def append_log(self, log: TaskLog) -> None:
        self.con.execute(
            "INSERT INTO task_logs (task_id, event, message, timestamp) VALUES (?, ?, ?, ?)",
            (log.task_id, log.event, log.message, self._dt_to_str(log.timestamp)),
        )

    def list_logs(self, task_id: str) -> List[TaskLog]:
        rows = self.con.execute(
            "SELECT task_id, event, message, timestamp FROM task_logs WHERE task_id = ? ORDER BY timestamp",
            (task_id,),
        ).fetchall()
        return [TaskLog(task_id=row[0], event=row[1], message=row[2], timestamp=self._str_to_dt(row[3]) or datetime.utcnow()) for row in rows]

    # ---- internal ----
    def _row_to_task(self, row) -> Task:
        return Task(
            id=row[0],
            type=row[1],
            status=self._status_from_str(row[2]),
            manual_required=bool(row[3]),
            payload=self._json_load(row[4]) or {},
            scheduled_for=self._str_to_dt(row[5]),
            created_at=self._str_to_dt(row[6]) or datetime.utcnow(),
            updated_at=self._str_to_dt(row[7]) or datetime.utcnow(),
            executed_at=self._str_to_dt(row[8]),
            result=self._json_load(row[9]),
            error=self._json_load(row[10]),
            attempts=row[11] or 0,
            confirmation_note=row[12],
        )


__all__ = ["DuckDBTaskStore"]
