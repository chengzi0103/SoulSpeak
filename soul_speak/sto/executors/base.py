"""Executor interfaces for SoulTask Orchestrator."""
from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Dict, Optional

from attrs import define

from soul_speak.sto.models import Task, TaskLog, TaskStatus
from soul_speak.sto.store.interface import TaskStoreProtocol


@define(init=False)
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
        task.executed_at = datetime.utcnow()
        store.update_task(task)
        self._log_event(store, task, "running", "task started")

    async def _finish_success(self, task: Task, store: TaskStoreProtocol, result: Optional[str] = None) -> None:
        task.status = TaskStatus.SUCCESS
        if result:
            if task.result is None:
                task.result = {"message": result}
            else:
                task.result.setdefault("message", result)
        store.update_task(task)
        self._log_event(store, task, "success", result or "ok")

    async def _finish_failed(self, task: Task, store: TaskStoreProtocol, error: str) -> None:
        task.status = TaskStatus.FAILED
        task.error = {"error": error}
        store.update_task(task)
        self._log_event(store, task, "failed", error)

    def _log_event(
        self,
        store: TaskStoreProtocol,
        task: Task,
        event: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        store.append_log(
            TaskLog(
                task_id=task.id,
                event=event,
                message=message,
                details=details,
            )
        )


__all__ = ["Executor"]
