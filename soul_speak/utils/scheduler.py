"""Shared APScheduler utilities for STO reminder scheduling."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore


DEFAULT_DB_PATH = Path("/Users/hougarden/project/github/SoulSpeak/data/apscheduler_jobs.duckdb")


def _build_jobstores(db_path: Path) -> Dict[str, SQLAlchemyJobStore]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"duckdb:///{db_path}"
    return {"default": SQLAlchemyJobStore(url=url)}


def create_async_scheduler(db_path: Optional[Path] = None) -> AsyncIOScheduler:
    jobstores = _build_jobstores(db_path or DEFAULT_DB_PATH)
    scheduler = AsyncIOScheduler(jobstores=jobstores)
    return scheduler


def create_background_scheduler(db_path: Optional[Path] = None) -> BackgroundScheduler:
    jobstores = _build_jobstores(db_path or DEFAULT_DB_PATH)
    scheduler = BackgroundScheduler(jobstores=jobstores)
    return scheduler


__all__ = [
    "create_async_scheduler",
    "create_background_scheduler",
]
