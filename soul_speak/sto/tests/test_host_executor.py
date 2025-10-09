"""Tests for HostExecutor behaviour."""
import asyncio
import sys

import pytest

from soul_speak.sto.executors.host_executor import HostExecutor
from soul_speak.sto.models import Task, TaskStatus
from soul_speak.sto.store.memory import MemoryTaskStore


def test_host_executor_runs_python_command() -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        task = Task(
            id="host-command-echo",
            type="host_command",
            payload={
                "command": [sys.executable, "-c", "print('host executor works')"],
                "timeout": 5,
            },
        )
        store.create_task(task)

        executor = HostExecutor(allowed_commands=[sys.executable])
        await executor.execute(task, store)

        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.SUCCESS
        assert updated.result is not None
        assert updated.result.get("stdout", "").strip() == "host executor works"

    asyncio.run(scenario())


def test_host_executor_blocks_disallowed_command() -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        task = Task(
            id="host-command-blocked",
            type="host_command",
            payload={
                "command": [sys.executable, "-c", "print('should not run')"],
            },
        )
        store.create_task(task)

        executor = HostExecutor(allowed_commands=["/usr/bin/python3"])  # intentionally different path
        await executor.execute(task, store)

        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.FAILED
        assert updated.error is not None
        assert "not allowed" in updated.error.get("error", "")

    asyncio.run(scenario())


def test_host_executor_simulates_brew_install(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        task = Task(
            id="host-command-brew",
            type="host_command",
            payload={
                "command": ["brew", "install", "ffmpeg"],
            },
        )
        store.create_task(task)

        executor = HostExecutor(allowed_commands=["brew"])

        async def fake_run_command(command, *, shell, cwd, env, timeout):  # type: ignore[override]
            assert command == ("brew", "install", "ffmpeg")
            return ("ffmpeg installed", "", 0)

        monkeypatch.setattr(executor, "_run_command", fake_run_command)
        await executor.execute(task, store)

        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.SUCCESS
        assert updated.result is not None
        assert updated.result.get("stdout") == "ffmpeg installed"

    asyncio.run(scenario())
