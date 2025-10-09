"""Demo: schedule a reminder job using APScheduler + DuckDB jobstore."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from soul_speak.utils.scheduler import create_async_scheduler


async def reminder_job(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Reminder: {message}")


async def main() -> None:
    scheduler = create_async_scheduler()
    scheduler.start()

    run_time = datetime.now() + timedelta(seconds=5)
    scheduler.add_job(reminder_job, "date", run_date=run_time, args=["Demo reminder"], id="demo-job")

    try:
        await asyncio.sleep(10)
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
