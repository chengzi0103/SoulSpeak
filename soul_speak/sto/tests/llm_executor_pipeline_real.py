"""Pipeline 示例：使用真实 LLM 代理规划并执行“安装 ffmpeg”任务。

运行前请确保：
  1. 已在 conf/llm/gpt.yaml 或对应配置中填好模型参数与 API Key；
  2. 环境变量或 conf 文件可被 openai_native.build_agent() 读取；
  3. 当前用户具有执行 brew 的权限（建议非 root）。

执行方式：
  python soul_speak/sto/tests/llm_executor_pipeline_real.py
"""
from __future__ import annotations

import asyncio
import argparse
import os
import json
from pathlib import Path

from soul_speak.sto.executors.host_executor import HostExecutor
from soul_speak.sto.executors.llm_executor import LLMExecutor
from soul_speak.sto.executors.system_usage import SystemUsageExecutor
from soul_speak.sto.models import Task
from soul_speak.sto.store.memory import MemoryTaskStore
from soul_speak.sto.store.duckdb_store import DuckDBTaskStore
from soul_speak.utils.hydra_config.init import conf

try:
    from soul_speak.llm.openai_native import build_agent
except ImportError as exc:  # pragma: no cover - 如果缺少依赖
    raise RuntimeError("无法加载真实 LLM 代理，请确认依赖已安装。") from exc


async def main(prompt: str, task_id: str, store_kind: str, db_path: str) -> None:
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    if store_kind == "duckdb":
        store = DuckDBTaskStore(Path(db_path))
    else:
        store = MemoryTaskStore()
    actual_task_id = task_id
    if isinstance(store, DuckDBTaskStore):
        suffix = 1
        while store.get_task(actual_task_id) is not None:
            suffix += 1
            actual_task_id = f"{task_id}-{suffix}"

    task = Task(
        id=actual_task_id,
        type="plan_and_execute",
        payload={
            "prompt": prompt,
        },
    )
    store.create_task(task)

    host_executor = HostExecutor(
        allowed_commands={
            "brew",
            "/usr/local/bin/brew",
            "/opt/homebrew/bin/brew",
            "top",
            "/usr/bin/top",
        },
        stdout_log_limit=4000,
        stderr_log_limit=4000,
    )
    system_usage_executor = SystemUsageExecutor()

    if hasattr(conf.llm, "memory"):
        conf.llm.memory.enable = True
    if hasattr(conf.llm, "tools"):
        conf.llm.tools.enable = True

    agent = build_agent()
    executor = LLMExecutor(
        agent=agent,
        tool_registry={
            "host_command": host_executor,
            "system_usage": system_usage_executor,
        },
        config_path=Path("conf/sto/llm_executor.yaml"),
        retry_limit=1,
    )

    try:
        await executor.execute(task, store)
    finally:
        shutdown = getattr(agent, "shutdown", None)
        if callable(shutdown):
            await shutdown()

    updated = store.get_task(task.id)
    if updated:
        print("任务状态:", updated.status.value)
        if updated.result:
            print("规划结果:")
            print(json.dumps(updated.result.get("plan", {}), ensure_ascii=False, indent=2))
            print("步骤执行:")
            for step in updated.result.get("step_results", []):
                print("-", step)
            print("最终摘要:", updated.result.get("reply"))
        if updated.error:
            print("错误:", updated.error.get("error"))

    print("日志事件:")
    for log in store.list_logs(task.id):
        print(f"[{log.event}] {log.message}")
        if log.details:
            print("  details:", log.details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMExecutor planner+executor pipeline")
    parser.add_argument("prompt", help="用户自然语言任务描述，如 '请在本地安装 ffmpeg。'")
    parser.add_argument(
        "--task-id",
        default="task-llm-executor",
        help="自定义任务 ID（默认: task-llm-executor）",
    )
    parser.add_argument(
        "--store",
        choices=["memory", "duckdb"],
        default="memory",
        help="选择任务存储后端，memory 或 duckdb (默认 memory)",
    )
    parser.add_argument(
        "--db-path",
        default="data/sto_tasks.duckdb",
        help="当使用 duckdb 存储时的数据库路径",
    )
    args = parser.parse_args()

    asyncio.run(main(args.prompt, args.task_id, args.store, args.db_path))
