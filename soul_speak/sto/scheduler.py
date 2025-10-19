"""Async task scheduler for SoulTask Orchestrator."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Iterable, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from attrs import define, field

from soul_speak.sto.executors.base import Executor
from soul_speak.sto.models import Task, TaskLog, TaskStatus
from soul_speak.sto.store.interface import TaskStoreProtocol


@define(slots=False)
class TaskScheduler:
    """Periodically polls the TaskStore and dispatches due tasks to executors."""

    store: TaskStoreProtocol
    executors: Iterable[Executor]
    poll_interval: float = 1.0
    max_concurrent: int = 4
    enable_background: bool = True
    scheduler: Optional[AsyncIOScheduler] = None
    _semaphore: asyncio.Semaphore = field(init=False)
    _scheduler: Optional[AsyncIOScheduler] = field(init=False)
    _job_id: Optional[str] = field(init=False, default=None)
    _running: bool = field(init=False, default=False)

    def __attrs_post_init__(self) -> None:
        self.executors = tuple(self.executors)
        self.poll_interval = max(0.1, float(self.poll_interval))
        self._semaphore = asyncio.Semaphore(max(1, int(self.max_concurrent)))
        self._scheduler = self.scheduler if self.enable_background else None
        self._job_id = None
        self._running = False

        if self.enable_background and self._scheduler is None:
            self._scheduler = AsyncIOScheduler(timezone="UTC")

    def start(self) -> None:
        """Start background polling using APScheduler."""
        if not self.enable_background:
            raise RuntimeError("Background scheduling is disabled for this TaskScheduler instance")
        if self._scheduler is None:
            raise RuntimeError("AsyncIOScheduler is not configured")
        if self._running:
            return
        self._job_id = "sto-dispatch"
        self._scheduler.add_job(
            self.dispatch_due_tasks,
            trigger="interval",
            seconds=self.poll_interval,
            id=self._job_id,
            name="sto-dispatch",
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        self._running = True

    def stop(self, wait: bool = True) -> None:
        """Stop background polling."""
        if not self.enable_background or self._scheduler is None:
            return
        if self._job_id and self._scheduler.get_job(self._job_id):
            self._scheduler.remove_job(self._job_id)
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
        self._job_id = None
        self._running = False

    async def run_once(self) -> None:
        """Execute one polling cycle synchronously (useful for tests)."""
        await self.dispatch_due_tasks()

    async def dispatch_due_tasks(self) -> None:
        """Find due tasks and dispatch them to matching executors."""
        try:
            tasks = self.store.list_tasks()
        except Exception as exc:  # pragma: no cover - defensive fallback
            # Without access to a logger, best effort is to swallow the error.
            return

        now = datetime.utcnow()
        pending: List[Task] = [
            task
            for task in tasks
            if self._is_task_due(task, now)
        ]
        if not pending:
            return

        await asyncio.gather(*(self._execute_task(task) for task in pending))

    def _is_task_due(self, task: Task, now: datetime) -> bool:
        if task.manual_required:
            return False
        if task.status is not TaskStatus.PENDING:
            return False
        if task.scheduled_for is None:
            return True
        try:
            return task.scheduled_for <= now
        except TypeError:
            # Fallback for timezone-aware datetimes mixed with naive now
            return task.scheduled_for <= datetime.fromtimestamp(now.timestamp(), tz=task.scheduled_for.tzinfo)

    async def _execute_task(self, task: Task) -> None:
        executor = self._select_executor(task)
        if executor is None:
            self._record_failure(task, "no executor registered for task type")
            return

        async with self._semaphore:
            task.attempts += 1
            self.store.update_task(task)
            self._log(
                task,
                "scheduler_dispatch",
                f"dispatching task to {executor.__class__.__name__}",
                {"executor": executor.__class__.__name__},
            )
            try:
                await executor.execute(task, self.store)
            except Exception as exc:  # pylint: disable=broad-except
                self._log(
                    task,
                    "scheduler_error",
                    "executor raised exception",
                    {"error": str(exc)},
                )
                task.status = TaskStatus.FAILED
                task.error = {"error": f"executor raised exception: {exc}"}
                self.store.update_task(task)
                return

            self._handle_post_execution(task.id)

    def _select_executor(self, task: Task) -> Optional[Executor]:
        for executor in self.executors:
            try:
                if executor.can_handle(task):
                    return executor
            except Exception:  # pragma: no cover - executor misbehaviour
                continue
        return None

    def _record_failure(self, task: Task, message: str) -> None:
        self._log(
            task,
            "scheduler_skipped",
            message,
            {"task_type": task.type},
        )
        task.status = TaskStatus.FAILED
        task.error = {"error": message}
        self.store.update_task(task)

    def _log(self, task: Task, event: str, message: str, details: Optional[dict] = None) -> None:
        try:
            self.store.append_log(
                TaskLog(
                    task_id=task.id,
                    event=event,
                    message=message,
                    details=details,
                )
            )
        except Exception:  # pragma: no cover - logging failure should not break scheduler
            pass

    def _handle_post_execution(self, task_id: str) -> None:
        updated = self.store.get_task(task_id)
        if updated is None:
            return

        interval = self._extract_interval(updated)
        if interval is None or updated.status is not TaskStatus.SUCCESS:
            return

        next_time = datetime.utcnow() + interval
        updated.status = TaskStatus.PENDING
        updated.scheduled_for = next_time
        updated.error = None
        updated.result = None
        updated.attempts = 0
        self.store.update_task(updated)
        self._log(
            updated,
            "scheduler_rescheduled",
            "task rescheduled for next interval",
            {
                "next_run_at": next_time.isoformat(),
                "interval_seconds": interval.total_seconds(),
            },
        )

    def _extract_interval(self, task: Task) -> Optional[timedelta]:
        payload = task.payload or {}
        interval_value = payload.get("interval_seconds")
        if interval_value is None:
            return None
        try:
            seconds = float(interval_value)
        except (TypeError, ValueError):
            return None
        if seconds <= 0:
            return None
        return timedelta(seconds=seconds)


__all__ = ["TaskScheduler"]
