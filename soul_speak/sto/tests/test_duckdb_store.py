"""Basic smoke test for DuckDBTaskStore."""
import asyncio
from datetime import datetime

from soul_speak.sto.executors.reminder import ReminderExecutor
from soul_speak.sto.models import Task, TaskStatus
from soul_speak.sto.store.duckdb_store import DuckDBTaskStore


async def main() -> None:
    store = DuckDBTaskStore()
    task_id = "reminder-duckdb-2100"
    if store.get_task(task_id):
        # Clean up previous run
        pass
    task = Task(
        id=task_id,
        type="reminder",
        payload={"message": "DuckDB提醒测试"},
        status=TaskStatus.PENDING,
        scheduled_for=datetime.utcnow(),
    )
    store.create_task(task)
    executor = ReminderExecutor()
    await executor.execute(task, store)
    updated = store.get_task(task_id)
    print(updated.status, updated.result)


if __name__ == "__main__":
    asyncio.run(main())
