"""Data models for SoulTask Orchestrator."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


class TaskStatus(str, enum.Enum):
    PENDING_MANUAL = "pending_manual"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    type: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    manual_required: bool = False
    scheduled_for: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    attempts: int = 0
    confirmation_note: Optional[str] = None


@dataclass
class TaskLog:
    task_id: str
    event: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None


__all__ = [
    "Task",
    "TaskLog",
    "TaskStatus",
]
