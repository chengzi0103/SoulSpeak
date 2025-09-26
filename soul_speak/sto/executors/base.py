"""Executor interfaces for SoulTask Orchestrator."""
from __future__ import annotations

import abc
from typing import Optional

from soul_speak.sto.models import Task, TaskLog, TaskStatus
from soul_speak.sto.store.interface import TaskStoreProtocol


class Executor(abc.ABC):
    """Base executor interface."""

    @abc.abstractmethod
    def can_handle(self, task: Task) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def execute(self, task: Task, store: TaskStoreProtocol) -> None:
        raise NotImplementedError

    async def _mark_running(self, task: Task, store: TaskStoreProtocol) -> None:
        task.status = TaskStatus.RUNNING
        store.update_task(task)
        store.append_log(TaskLog(task_id=task.id, event="running", message="task started"))

    async def _finish_success(self, task: Task, store: TaskStoreProtocol, result: Optional[str] = None) -> None:
        task.status = TaskStatus.SUCCESS
        if result:
            task.result = {"message": result}
        store.update_task(task)
        store.append_log(TaskLog(task_id=task.id, event="success", message=result or "ok"))

    async def _finish_failed(self, task: Task, store: TaskStoreProtocol, error: str) -> None:
        task.status = TaskStatus.FAILED
        task.error = {"error": error}
        store.update_task(task)
        store.append_log(TaskLog(task_id=task.id, event="failed", message=error))
