"""Executor that reports current system resource usage."""
from __future__ import annotations

import asyncio
import os
from typing import Dict, Optional

try:  # psutil 提供更丰富的指标，如果缺失则使用内置信息
    import psutil  # type: ignore
    from psutil import AccessDenied  # type: ignore
except Exception:  # pragma: no cover - 可选依赖
    psutil = None  # type: ignore
    AccessDenied = Exception  # type: ignore

from soul_speak.sto.executors.base import Executor
from soul_speak.sto.models import Task
from soul_speak.sto.store.interface import TaskStoreProtocol


class SystemUsageExecutor(Executor):
    """Gather CPU/Memory information from the local machine."""

    def __init__(self, sample_interval: float = 0.2) -> None:
        self.sample_interval = sample_interval
        self.supported_types = {"system_usage"}

    def can_handle(self, task: Task) -> bool:
        return task.type in self.supported_types

    async def execute(self, task: Task, store: TaskStoreProtocol) -> None:
        await self._mark_running(task, store)

        try:
            metrics = await asyncio.get_event_loop().run_in_executor(
                None,
                self._collect_metrics,
            )
        except Exception as exc:  # pragma: no cover - 纯防护
            await self._finish_failed(task, store, f"failed to collect system usage: {exc}")
            return

        summary = (
            f"CPU {metrics.get('cpu_percent', 'n/a')}%, "
            f"Memory {metrics.get('memory_percent', 'n/a')}%"
        )
        task.result = {"metrics": metrics, "message": summary}
        await self._finish_success(task, store, result=summary)

    def _collect_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        try:
            if psutil is not None:
                try:
                    metrics["cpu_percent"] = float(psutil.cpu_percent(interval=self.sample_interval))
                except Exception:
                    pass

                try:
                    virtual = psutil.virtual_memory()
                    metrics["memory_percent"] = float(virtual.percent)
                    metrics["memory_used_mb"] = float(virtual.used) / (1024 * 1024)
                    metrics["memory_total_mb"] = float(virtual.total) / (1024 * 1024)
                except AccessDenied:
                    pass

                try:
                    swap = psutil.swap_memory()
                    metrics["swap_percent"] = float(swap.percent)
                    metrics["swap_used_mb"] = float(swap.used) / (1024 * 1024)
                    metrics["swap_total_mb"] = float(swap.total) / (1024 * 1024)
                except AccessDenied:
                    pass

            try:
                load1, load5, load15 = os.getloadavg()
                metrics.update({
                    "load_1m": float(load1),
                    "load_5m": float(load5),
                    "load_15m": float(load15),
                })
            except OSError:
                metrics.setdefault("load_1m", 0.0)
                metrics.setdefault("load_5m", 0.0)
                metrics.setdefault("load_15m", 0.0)
        except Exception:
            pass

        return metrics


__all__ = ["SystemUsageExecutor"]
