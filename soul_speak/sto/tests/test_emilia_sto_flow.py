"""End-to-end flow: emulate Emilia scheduling an agent plan via STO tool and execute it."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from soul_speak.llm.tools.toolkit import sto_schedule_agent_plan, sto_schedule_task
from soul_speak.sto.models import TaskStatus
from soul_speak.sto.runtime import STOSchedulerService
from soul_speak.sto.store.duckdb_store import DuckDBTaskStore


def test_emilia_agent_plan_with_sandbox(tmp_path: Path) -> None:
    """Simulate Emilia receiving a request, scheduling an agent plan, and STO executing it."""

    db_path = tmp_path / "sto_tasks.duckdb"
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    safe_script = scripts_dir / "safe_task.py"
    safe_script.write_text(
        "import datetime\nprint(f\"[safe_task] ran at {datetime.datetime.utcnow().isoformat()}Z\")\n",
        encoding="utf-8",
    )

    runner_path = Path(__file__).resolve().parents[2] / "scripts" / "sandbox_runner.py"

    config = {
        "store": {"kind": "duckdb", "path": str(db_path)},
        "executors": {
            "host": {"enabled": False},
            "sandbox": {
                "enabled": True,
                "command": [sys.executable, str(runner_path)],
                "allowed_paths": [str(scripts_dir)],
                "default_timeout": 30,
            },
            "agent": {
                "enabled": True,
                "supported_types": ["agent_plan"],
                "allowed_commands": [sys.executable, "python", "python3"],
            },
            "llm": {"enabled": False},
            "reminder": {"enabled": False},
            "system_usage": {"enabled": False},
        },
    }

    service = STOSchedulerService(config)

    async def scenario() -> None:
        steps = json.dumps(
            [
                {"type": "note", "message": "准备执行安全脚本"},
                {
                    "type": "command",
                    "command": [sys.executable, str(safe_script)],
                    "timeout": 15,
                },
            ]
        )

        result = await sto_schedule_agent_plan(
            task_id="agent-plan-demo",
            steps=steps,
            summary="执行安全脚本",
            delay_seconds=0.0,
            db_path=str(db_path),
        )

        assert result["task_type"] == "agent_plan"

        store = DuckDBTaskStore(db_path)
        try:
            await service.scheduler.run_once()
            task = store.get_task("agent-plan-demo")
            assert task is not None
            assert task.status is TaskStatus.SUCCESS
            assert task.result is not None
            assert "summary" in task.result
        finally:
            store.con.close()

    try:
        asyncio.run(scenario())
    finally:
        service.stop()


def test_emilia_host_command_flow(tmp_path: Path) -> None:
    """Emilia schedules a host_command task via MCP tool and STO executes it."""

    db_path = tmp_path / "sto_tasks.duckdb"

    service = STOSchedulerService(
        {
            "store": {"kind": "duckdb", "path": str(db_path)},
            "executors": {
                "host": {
                    "enabled": True,
                    "allowed_commands": [sys.executable],
                },
                "sandbox": {"enabled": False},
                "agent": {"enabled": False},
                "llm": {"enabled": False},
                "reminder": {"enabled": False},
                "system_usage": {"enabled": False},
            },
        }
    )

    async def scenario() -> None:
        payload = json.dumps(
            {
                "command": [sys.executable, "-c", "print('host ok')"],
                "timeout": 10,
            }
        )

        result = await sto_schedule_task(
            task_id="host-demo",
            task_type="host_command",
            payload=payload,
            db_path=str(db_path),
        )

        assert result["task_type"] == "host_command"

        store = DuckDBTaskStore(db_path)
        try:
            await service.scheduler.run_once()
            task = store.get_task("host-demo")
            assert task is not None
            assert task.status is TaskStatus.SUCCESS
            assert task.result is not None
            assert task.result.get("returncode") == 0
        finally:
            store.con.close()

    try:
        asyncio.run(scenario())
    finally:
        service.stop()


def test_emilia_system_usage_flow(tmp_path: Path) -> None:
    """Emilia schedules a system_usage task and STO collects metrics."""

    db_path = tmp_path / "sto_tasks.duckdb"

    service = STOSchedulerService(
        {
            "store": {"kind": "duckdb", "path": str(db_path)},
            "executors": {
                "host": {"enabled": False},
                "sandbox": {"enabled": False},
                "agent": {"enabled": False},
                "llm": {"enabled": False},
                "reminder": {"enabled": False},
                "system_usage": {"enabled": True, "sample_interval": 0.05},
            },
        }
    )

    async def scenario() -> None:
        result = await sto_schedule_task(
            task_id="system-usage-demo",
            task_type="system_usage",
            payload=json.dumps({}),
            db_path=str(db_path),
        )
        assert result["task_type"] == "system_usage"

        store = DuckDBTaskStore(db_path)
        try:
            await service.scheduler.run_once()
            task = store.get_task("system-usage-demo")
            assert task is not None
            assert task.status is TaskStatus.SUCCESS
            assert task.result is not None
            assert "metrics" in task.result
        finally:
            store.con.close()

    try:
        asyncio.run(scenario())
    finally:
        service.stop()
