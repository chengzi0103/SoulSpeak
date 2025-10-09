"""简单脚本：使用 HostExecutor 测试本地安装 ffmpeg。"""
from __future__ import annotations

import asyncio

from soul_speak.sto.executors.host_executor import HostExecutor
from soul_speak.sto.models import Task
from soul_speak.sto.store.memory import MemoryTaskStore


async def install_ffmpeg() -> None:
    store = MemoryTaskStore()
    task = Task(
        id="host-install-ffmpeg",
        type="host_command",
        payload={
            "command": ["brew", "install", "ffmpeg"],
            "timeout": 900,
        },
    )
    store.create_task(task)

    executor = HostExecutor(allowed_commands={"brew", "/usr/local/bin/brew", "/opt/homebrew/bin/brew"})
    await executor.execute(task, store)

    updated = store.get_task(task.id)
    print("任务状态:", updated.status.value if updated else "未知")
    if updated and updated.result:
        print("命令输出:")
        print((updated.result.get("stdout") or "").strip())
        print((updated.result.get("stderr") or "").strip())
    if updated and updated.error:
        print("错误信息:", updated.error.get("error"))


if __name__ == "__main__":
    asyncio.run(install_ffmpeg())
