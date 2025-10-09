"""Smoke test for ReminderExecutor."""
import asyncio
from datetime import datetime

from soul_speak.sto.executors.reminder import ReminderExecutor
from soul_speak.sto.models import Task, TaskStatus
from soul_speak.sto.store.memory import MemoryTaskStore


async def main() -> None:
    store = MemoryTaskStore()
    task = Task(
        id="reminder-2025-01-08-2100",
        type="reminder",
        payload={"message": "晚上9点喝水", "tts_voice": "emilia"},
        status=TaskStatus.PENDING,
        scheduled_for=datetime.now(),
    )
    store.create_task(task)
    executor = ReminderExecutor()
    await executor.execute(task, store)
    updated = store.get_task(task.id)
    print(updated.status, updated.result)


if __name__ == "__main__":
    asyncio.run(main())
