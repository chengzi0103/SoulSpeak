"""Tests for the TaskScheduler."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from attrs import define, field

from soul_speak.sto.executors.base import Executor
from soul_speak.sto.executors.sandbox import SandboxExecutor
from soul_speak.sto.executors.agent_executor import AgentExecutor
from soul_speak.sto.models import Task, TaskStatus
from soul_speak.sto.runtime import STOSchedulerService
from soul_speak.sto.scheduler import TaskScheduler
from soul_speak.sto.store.memory import MemoryTaskStore


@define(slots=True)
class DummySuccessExecutor(Executor):
    supported_types: set[str] = field(factory=lambda: {"dummy"})

    def can_handle(self, task: Task) -> bool:  # type: ignore[override]
        return task.type in self.supported_types

    async def execute(self, task: Task, store):  # type: ignore[override]
        await self._mark_running(task, store)
        await self._finish_success(task, store, result="done")


@define(slots=True)
class DummyFailExecutor(Executor):
    supported_types: set[str] = field(factory=lambda: {"fail"})

    def can_handle(self, task: Task) -> bool:  # type: ignore[override]
        return task.type in self.supported_types

    async def execute(self, task: Task, store):  # type: ignore[override]
        await self._mark_running(task, store)
        raise RuntimeError("boom")


def run(coro):
    return asyncio.run(coro)


def test_sandbox_executor_runs_allowed_script(tmp_path) -> None:
    script = tmp_path / "sandbox_ok.py"
    script.write_text("print('sandbox ok')\n", encoding="utf-8")

    runner = Path(__file__).resolve().parents[3] / "scripts" / "sandbox_runner.py"
    executor = SandboxExecutor(
        sandbox_cmd=(sys.executable, str(runner)),
        allowed_paths=(tmp_path,),
        supported_types={"sandbox_command"},
        stdout_log_limit=200,
        stderr_log_limit=200,
    )

    store = MemoryTaskStore()
    task = Task(
        id="sandbox-1",
        type="sandbox_command",
        payload={"path": str(script)},
    )
    store.create_task(task)

    run(executor.execute(task, store))

    updated = store.get_task("sandbox-1")
    assert updated is not None
    assert updated.status is TaskStatus.SUCCESS
    assert updated.result is not None
    assert "sandbox ok" in (updated.result.get("stdout") or "")


def test_sandbox_executor_blocks_unauthorised_path(tmp_path) -> None:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    external_script = tmp_path / "outside.py"
    external_script.write_text("print('no')\n", encoding="utf-8")

    runner = Path(__file__).resolve().parents[3] / "scripts" / "sandbox_runner.py"
    executor = SandboxExecutor(
        sandbox_cmd=(sys.executable, str(runner)),
        allowed_paths=(allowed_dir,),
        supported_types={"sandbox_command"},
    )

    store = MemoryTaskStore()
    task = Task(
        id="sandbox-unauthorised",
        type="sandbox_command",
        payload={"path": str(external_script)},
    )
    store.create_task(task)

    run(executor.execute(task, store))

    updated = store.get_task("sandbox-unauthorised")
    assert updated is not None
    assert updated.status is TaskStatus.FAILED
    assert updated.error is not None
    assert "outside allowed" in updated.error.get("error", "")


def test_agent_executor_runs_plan() -> None:
    executor = AgentExecutor(
        supported_types={"agent_plan"},
        max_steps=5,
        allowed_commands={sys.executable},
    )
    store = MemoryTaskStore()
    task = Task(
        id="agent-1",
        type="agent_plan",
        payload={
            "plan": [
                {"type": "note", "message": "准备"},
                {"type": "sleep", "seconds": 0.01},
                {
                    "type": "command",
                    "command": [sys.executable, "-c", "print('done')"],
                    "timeout": 5,
                },
            ],
            "summary": "完成三步计划",
        },
    )
    store.create_task(task)

    run(executor.execute(task, store))

    updated = store.get_task("agent-1")
    assert updated is not None
    assert updated.status is TaskStatus.SUCCESS
    assert updated.result is not None
    assert updated.result.get("summary") == "完成三步计划"


def test_agent_executor_rejects_invalid_plan() -> None:
    executor = AgentExecutor(supported_types={"agent_plan"})
    store = MemoryTaskStore()
    task = Task(
        id="agent-invalid",
        type="agent_plan",
        payload={"plan": "oops"},
    )
    store.create_task(task)

    run(executor.execute(task, store))

    updated = store.get_task("agent-invalid")
    assert updated is not None
    assert updated.status is TaskStatus.FAILED
    assert updated.error is not None
    assert "step" in updated.error.get("error", "")


def test_scheduler_executes_due_task() -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        task = Task(
            id="dummy-1",
            type="dummy",
            payload={},
            scheduled_for=datetime.utcnow() - timedelta(seconds=1),
        )
        store.create_task(task)
        scheduler = TaskScheduler(
            store,
            executors=[DummySuccessExecutor()],
            poll_interval=0.1,
            enable_background=False,
        )
        await scheduler.run_once()
        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.SUCCESS
        assert updated.result is not None
        assert updated.result.get("message") == "done"

    run(scenario())


def test_scheduler_marks_failure_on_executor_exception() -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        task = Task(
            id="fail-1",
            type="fail",
            payload={},
            scheduled_for=datetime.utcnow() - timedelta(seconds=1),
        )
        store.create_task(task)
        scheduler = TaskScheduler(
            store,
            executors=[DummyFailExecutor()],
            poll_interval=0.1,
            enable_background=False,
        )
        await scheduler.run_once()
        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.FAILED
        assert updated.error is not None
        assert "executor raised exception" in updated.error.get("error", "")

    run(scenario())


def test_scheduler_reschedules_interval_task() -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        initial_time = datetime.utcnow() - timedelta(seconds=5)
        task = Task(
            id="interval-1",
            type="dummy",
            payload={"interval_seconds": 0.5},
            scheduled_for=initial_time,
        )
        store.create_task(task)
        scheduler = TaskScheduler(
            store,
            executors=[DummySuccessExecutor()],
            poll_interval=0.1,
            enable_background=False,
        )
        before_run = datetime.utcnow()
        await scheduler.run_once()
        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.PENDING
        assert updated.scheduled_for is not None
        assert updated.scheduled_for > before_run
        assert updated.result is None
        assert updated.error is None
        assert updated.attempts == 0

    run(scenario())


def test_scheduler_skips_future_tasks() -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        task = Task(
            id="future-1",
            type="dummy",
            payload={},
            scheduled_for=datetime.utcnow() + timedelta(hours=1),
        )
        store.create_task(task)
        scheduler = TaskScheduler(
            store,
            executors=[DummySuccessExecutor()],
            poll_interval=0.1,
            enable_background=False,
        )
        await scheduler.run_once()
        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.PENDING
        assert updated.result is None

    run(scenario())


def test_scheduler_marks_failed_when_no_executor() -> None:
    async def scenario() -> None:
        store = MemoryTaskStore()
        task = Task(
            id="no-exec",
            type="unknown",
            payload={},
            scheduled_for=datetime.utcnow() - timedelta(seconds=1),
        )
        store.create_task(task)
        scheduler = TaskScheduler(
            store,
            executors=[DummySuccessExecutor()],
            poll_interval=0.1,
            enable_background=False,
        )
        await scheduler.run_once()
        updated = store.get_task(task.id)
        assert updated is not None
        assert updated.status is TaskStatus.FAILED
        assert updated.error is not None
        assert "no executor" in updated.error.get("error", "")

    run(scenario())


def test_scheduler_service_builds_llm_executor(monkeypatch) -> None:
    created_kwargs = {}

    class DummyLLMExecutor(Executor):
        def __init__(self, **kwargs):
            created_kwargs.update(kwargs)
            supported = kwargs.get("supported_types") or []
            self.supported_types = set(supported)

        def can_handle(self, task: Task) -> bool:  # type: ignore[override]
            return task.type in self.supported_types or not self.supported_types

        async def execute(self, task: Task, store):  # type: ignore[override]
            await self._mark_running(task, store)
            await self._finish_success(task, store)

    monkeypatch.setattr("soul_speak.sto.runtime.LLMExecutor", DummyLLMExecutor)

    service = STOSchedulerService(
        {
            "executors": {
                "host": {"enabled": True, "allowed_commands": [sys.executable]},
                "reminder": {"enabled": False},
                "system_usage": {"enabled": False},
                "llm": {
                    "enabled": True,
                    "retry_limit": 2,
                    "supported_types": ["qa"],
                },
            }
        }
    )

    assert any(isinstance(executor, DummyLLMExecutor) for executor in service.executors)
    assert created_kwargs.get("retry_limit") == 2
    assert created_kwargs.get("supported_types") == ["qa"]
    tool_registry = created_kwargs.get("tool_registry")
    assert tool_registry is not None
    assert "host_command" in tool_registry
    assert tool_registry["host_command"].__class__.__name__ == "HostExecutor"
    assert "host_command" in service.tool_registry


def test_scheduler_service_executes_llm_task(monkeypatch) -> None:
    executed = {}

    class DummyLLMExecutor(Executor):
        def __init__(self, **kwargs):
            self.supported_types = {"plan_and_execute"}
            executed["init_kwargs"] = kwargs

        def can_handle(self, task: Task) -> bool:  # type: ignore[override]
            return task.type in self.supported_types

        async def execute(self, task: Task, store):  # type: ignore[override]
            executed["task_id"] = task.id
            await self._mark_running(task, store)
            await self._finish_success(task, store, result="ok")

    monkeypatch.setattr("soul_speak.sto.runtime.LLMExecutor", DummyLLMExecutor)

    service = STOSchedulerService(
        {
            "store": {"kind": "memory"},
            "executors": {
                "host": {"enabled": False},
                "reminder": {"enabled": False},
                "system_usage": {"enabled": False},
                "llm": {
                    "enabled": True,
                    "supported_types": ["plan_and_execute"],
                },
            },
        }
    )

    task = Task(
        id="llm-1",
        type="plan_and_execute",
        payload={"prompt": "测试调度 LLM 执行"},
        scheduled_for=datetime.utcnow() - timedelta(seconds=1),
    )
    service.store.create_task(task)

    asyncio.run(service.scheduler.run_once())

    updated = service.store.get_task("llm-1")
    assert updated is not None
    assert updated.status is TaskStatus.SUCCESS
    assert executed.get("task_id") == "llm-1"


def test_scheduler_service_builds_sandbox_executor(tmp_path) -> None:
    service = STOSchedulerService(
        {
            "store": {"kind": "memory"},
            "executors": {
                "host": {"enabled": False},
                "reminder": {"enabled": False},
                "system_usage": {"enabled": False},
                "sandbox": {
                    "enabled": True,
                    "command": [sys.executable],
                    "allowed_paths": [str(tmp_path)],
                },
            },
        }
    )

    assert any(isinstance(executor, SandboxExecutor) for executor in service.executors)


def test_scheduler_service_builds_agent_executor() -> None:
    service = STOSchedulerService(
        {
            "store": {"kind": "memory"},
            "executors": {
                "host": {"enabled": False},
                "reminder": {"enabled": False},
                "system_usage": {"enabled": False},
                "agent": {
                    "enabled": True,
                    "supported_types": ["agent_plan"],
                    "max_steps": 10,
                    "allowed_commands": [sys.executable],
                },
            },
        }
    )

    assert any(isinstance(executor, AgentExecutor) for executor in service.executors)


def test_scheduler_service_executes_agent_plan(monkeypatch) -> None:
    service = STOSchedulerService(
        {
            "store": {"kind": "memory"},
            "executors": {
                "host": {"enabled": False},
                "reminder": {"enabled": False},
                "system_usage": {"enabled": False},
                "agent": {
                    "enabled": True,
                    "supported_types": ["agent_plan"],
                    "allowed_commands": [sys.executable],
                },
            },
        }
    )

    task = Task(
        id="agent-plan-run",
        type="agent_plan",
        payload={
            "plan": [
                {"type": "note", "message": "start"},
                {"type": "command", "command": [sys.executable, "-c", "print('agent ok')"]},
            ],
            "summary": "agent ok",
        },
        scheduled_for=datetime.utcnow() - timedelta(seconds=1),
    )
    service.store.create_task(task)

    asyncio.run(service.scheduler.run_once())

    updated = service.store.get_task("agent-plan-run")
    assert updated is not None
    assert updated.status is TaskStatus.SUCCESS
    assert updated.result is not None
    assert updated.result.get("summary") == "agent ok"
