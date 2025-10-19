"""Data models for SoulTask Orchestrator."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, Optional

from attrs import define, field


class TaskStatus(str, enum.Enum):
    PENDING_MANUAL = "pending_manual"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@define(slots=True)
class Task:
    id: str
    type: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    manual_required: bool = False
    scheduled_for: Optional[datetime] = None
    created_at: datetime = field(factory=datetime.utcnow)
    updated_at: datetime = field(factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    attempts: int = 0
    confirmation_note: Optional[str] = None


@define(slots=True)
class TaskLog:
    task_id: str
    event: str
    message: str
    timestamp: datetime = field(factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None


__all__ = [
    "Task",
    "TaskLog",
    "TaskStatus",
]
