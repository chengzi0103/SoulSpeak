"""In-memory TaskStore for testing and prototyping."""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

from soul_speak.sto.models import Task, TaskLog, TaskStatus


class MemoryTaskStore:
    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._logs: List[TaskLog] = []
        self._lock = threading.Lock()

    def create_task(self, task: Task) -> None:
        with self._lock:
            if task.id in self._tasks:
                raise ValueError(f"Task {task.id} already exists")
            self._tasks[task.id] = task
        self._logs.append(
            TaskLog(
                task_id=task.id,
                event="created",
                message="task created",
                details={"payload": task.payload},
            )
        )

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        with self._lock:
            return list(self._tasks.values())

    def update_task(self, task: Task) -> None:
        with self._lock:
            if task.id not in self._tasks:
                raise KeyError(f"Task {task.id} not found")
            self._tasks[task.id] = task
            self._logs.append(
                TaskLog(
                    task_id=task.id,
                    event="updated",
                    message=f"status -> {task.status.value}",
                )
            )

    def append_log(self, log: TaskLog) -> None:
        with self._lock:
            self._logs.append(log)

    def list_logs(self, task_id: str) -> List[TaskLog]:
        with self._lock:
            return [log for log in self._logs if log.task_id == task_id]


__all__ = ["MemoryTaskStore"]
