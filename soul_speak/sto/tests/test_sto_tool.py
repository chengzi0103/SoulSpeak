"""Validate the sto_schedule_task FastMCP tool."""
from __future__ import annotations

import asyncio
import json
from typing import Set

import pytest

from soul_speak.llm.tools.toolkit import (
    TOOL_REGISTRY,
    sto_schedule_task,
    sto_schedule_agent_plan,
    sto_list_tasks,
    sto_task_detail,
)
from soul_speak.sto.store.duckdb_store import DuckDBTaskStore


def run(coro):
    return asyncio.run(coro)


def test_tool_is_registered() -> None:
    names: Set[str] = {tool.name for tool in TOOL_REGISTRY}
    assert "sto_schedule_task" in names
    assert "sto_schedule_agent_plan" in names
    assert "sto_list_tasks" in names
    assert "sto_task_detail" in names


def test_tool_creates_duckdb_task(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "sto_tasks.duckdb"

    def _factory(*args, **kwargs) -> DuckDBTaskStore:
        return DuckDBTaskStore(db_path=db_path)

    monkeypatch.setattr("soul_speak.llm.tools.toolkit.DuckDBTaskStore", _factory)

    payload = json.dumps({"message": "test schedule"})
    result = run(
        sto_schedule_task(
            task_id="tool-reminder-1",
            task_type="reminder",
            payload=payload,
            delay_seconds=0.0,
            interval_seconds=60,
            db_path=str(db_path),
        )
    )

    assert result["task_id"] == "tool-reminder-1"
    assert result["action"] == "created"
    assert result["interval_seconds"] == 60

    store = DuckDBTaskStore(db_path=db_path)
    try:
        task = store.get_task("tool-reminder-1")
        assert task is not None
        assert task.payload["message"] == "test schedule"
        assert task.payload["interval_seconds"] == 60
    finally:
        store.con.close()


def test_agent_plan_tool_creates_task(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "sto_tasks.duckdb"

    def _factory(*args, **kwargs) -> DuckDBTaskStore:
        return DuckDBTaskStore(db_path=db_path)

    monkeypatch.setattr("soul_speak.llm.tools.toolkit.DuckDBTaskStore", _factory)

    steps = json.dumps([
        {"type": "note", "message": "准备"},
        {"type": "command", "command": ["python", "-c", "print('ok')"], "timeout": 10},
    ])

    result = run(
        sto_schedule_agent_plan(
            task_id="agent-plan-1",
            steps=steps,
            summary="agent ok",
            delay_seconds=0.0,
            db_path=str(db_path),
        )
    )

    assert result["task_id"] == "agent-plan-1"
    assert result["task_type"] == "agent_plan"

    store = DuckDBTaskStore(db_path=db_path)
    try:
        task = store.get_task("agent-plan-1")
        assert task is not None
        assert task.type == "agent_plan"
        assert len(task.payload["plan"]) == 2
        assert task.payload["summary"] == "agent ok"
    finally:
        store.con.close()


def test_task_listing_and_detail(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "sto_tasks.duckdb"

    def _factory(*args, **kwargs) -> DuckDBTaskStore:
        return DuckDBTaskStore(db_path=db_path)

    monkeypatch.setattr("soul_speak.llm.tools.toolkit.DuckDBTaskStore", _factory)

    payload = json.dumps({"message": "list demo"})
    run(
        sto_schedule_task(
            task_id="list-demo",
            task_type="reminder",
            payload=payload,
            db_path=str(db_path),
        )
    )

    listing = run(sto_list_tasks(status="pending", db_path=str(db_path)))
    ids = {task["id"] for task in listing["tasks"]}
    assert "list-demo" in ids

    detail = run(sto_task_detail(task_id="list-demo", db_path=str(db_path)))
    assert detail["task"]["id"] == "list-demo"
    assert detail["task"]["status"] == "pending"
    assert detail["logs"][0]["event"] == "created"
