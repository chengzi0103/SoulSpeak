"""Reminder executor using local playback (placeholder)."""
from __future__ import annotations

from typing import Dict

from attrs import define, field

from soul_speak.sto.executors.base import Executor
from soul_speak.sto.models import Task
from soul_speak.sto.store.interface import TaskStoreProtocol


@define(slots=True)
class ReminderExecutor(Executor):
    voices: Dict[str, str] = field(factory=dict)
    supported_types: set[str] = field(factory=lambda: {"reminder"})

    def can_handle(self, task: Task) -> bool:
        return task.type in self.supported_types

    async def execute(self, task: Task, store: TaskStoreProtocol) -> None:
        await self._mark_running(task, store)
        payload = task.payload
        message = payload.get("message", "提醒时间到了")
        # Placeholder: real implementation would call play_sentences or another TTS pipeline
        print(f"[Reminder] {message}")
        await self._finish_success(task, store, result=message)
