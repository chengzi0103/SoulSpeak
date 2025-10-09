"""示例：LLMExecutor 自行规划并执行“安装 ffmpeg”任务。"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from soul_speak.sto.executors.host_executor import HostExecutor
from soul_speak.sto.executors.llm_executor import LLMExecutor
from soul_speak.sto.models import Task
from soul_speak.sto.store.memory import MemoryTaskStore


class RulePlannerAgent:
    """极简规划代理：根据 natural language prompt 返回固定 JSON 计划."""

    async def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        if "ffmpeg" in prompt_lower and ("install" in prompt_lower or "安装" in prompt_lower or "安裝" in prompt_lower):
            plan = {
                "plan": [
                    {
                        "id": "step-1",
                        "task_type": "host_command",
                        "description": "使用 brew 安装 ffmpeg",
                        "payload": {
                            "command": ["brew", "install", "ffmpeg"],
                            "timeout": kwargs.get("timeout", 600),
                        },
                    }
                ],
                "final_summary": "命令已执行，检查返回结果。",
                "metadata": {"usage": {"total_tokens": 1}},
            }
            return {"text": json.dumps(plan), "metadata": plan["metadata"]}
        raise ValueError("planner 无法识别该任务：" + prompt)


async def main() -> None:
    store = MemoryTaskStore()
    task = Task(
        id="task-install-ffmpeg",
        type="plan_and_execute",
        payload={
            "prompt": "请在本地安装 ffmpeg。",
        },
    )
    store.create_task(task)

    host_executor = HostExecutor(
        allowed_commands={"brew", "/usr/local/bin/brew", "/opt/homebrew/bin/brew"}
    )
    executor = LLMExecutor(
        agent=RulePlannerAgent(),
        tool_registry={"host_command": host_executor},
        retry_limit=0,
    )

    await executor.execute(task, store)

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
    asyncio.run(main())
