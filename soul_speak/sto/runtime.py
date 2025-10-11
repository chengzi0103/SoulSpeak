"""Runtime glue that wires the TaskScheduler into Emilia's process."""
from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf

from soul_speak.sto.executors.host_executor import HostExecutor
from soul_speak.sto.executors.llm_executor import LLMExecutor
from soul_speak.sto.executors.reminder import ReminderExecutor
from soul_speak.sto.executors.sandbox import SandboxExecutor
from soul_speak.sto.executors.system_usage import SystemUsageExecutor
from soul_speak.sto.executors.agent_executor import AgentExecutor
from soul_speak.sto.scheduler import TaskScheduler
from soul_speak.sto.store.duckdb_store import DuckDBTaskStore
from soul_speak.sto.store.interface import TaskStoreProtocol
from soul_speak.sto.store.memory import MemoryTaskStore


def _to_dict(data: Any) -> Dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    try:
        return OmegaConf.to_container(data, resolve=True)  # type: ignore[arg-type]
    except Exception:
        return {}


class STOSchedulerService:
    """Configure and run the STO scheduler with executors defined in Hydra config."""

    def __init__(self, config: Optional[Any] = None) -> None:
        cfg = _to_dict(config)

        self.poll_interval = float(cfg.get("poll_interval", 1.0))
        self.max_concurrent = int(cfg.get("max_concurrent", 4))
        self.auto_start = bool(cfg.get("auto_start", True))

        self.store: TaskStoreProtocol = self._create_store(cfg.get("store"))
        self.executors, self.tool_registry = self._create_executors(cfg.get("executors"))

        self.scheduler = TaskScheduler(
            self.store,
            self.executors,
            poll_interval=self.poll_interval,
            max_concurrent=self.max_concurrent,
        )
        self._started = False

    @classmethod
    def from_global_config(cls) -> "STOSchedulerService":
        try:
            from soul_speak.utils.hydra_config.init import conf  # type: ignore
        except Exception:
            return cls({})
        sto_conf = getattr(conf, "sto", None)
        return cls(sto_conf)

    def start(self) -> None:
        if self._started:
            return
        try:
            self.scheduler.start()
            self._started = True
        except Exception as exc:
            # Avoid crashing the caller; they can continue without scheduler.
            print(f"[STO] Failed to start scheduler: {exc}", file=sys.stderr)

    async def run_once(self) -> None:
        await self.scheduler.run_once()

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.scheduler.stop()
        finally:
            self._started = False
            self._close_store()

    def _create_store(self, cfg: Optional[Dict[str, Any]]) -> TaskStoreProtocol:
        cfg = cfg or {}
        kind = str(cfg.get("kind", "duckdb")).lower()
        if kind == "memory":
            return MemoryTaskStore()
        path_value = cfg.get("path")
        db_path = Path(path_value) if path_value else None
        return DuckDBTaskStore(db_path=db_path)

    def _create_executors(self, cfg: Optional[Dict[str, Any]]) -> Tuple[List, Dict[str, Any]]:
        cfg = cfg or {}
        executors: List = []
        tool_registry: Dict[str, Any] = {}

        host_cfg = _to_dict(cfg.get("host"))
        if host_cfg.get("enabled", True):
            allowed = set(str(item) for item in host_cfg.get("allowed_commands", []) if item)
            allowed.add(sys.executable)
            base_env = host_cfg.get("base_env") or {}
            host_executor = HostExecutor(
                allowed_commands=allowed or None,
                base_env=base_env if base_env else None,
            )
            executors.append(host_executor)
            tool_registry.setdefault("host_command", host_executor)

        sandbox_cfg = _to_dict(cfg.get("sandbox"))
        if sandbox_cfg.get("enabled", False):
            command_value = sandbox_cfg.get("command") or sandbox_cfg.get("runner")
            if command_value:
                sandbox_kwargs: Dict[str, Any] = {
                    "sandbox_cmd": command_value,
                }
                allowed_paths = sandbox_cfg.get("allowed_paths")
                if allowed_paths:
                    sandbox_kwargs["allowed_paths"] = allowed_paths
                base_env = sandbox_cfg.get("base_env")
                if base_env:
                    sandbox_kwargs["base_env"] = base_env
                default_timeout = sandbox_cfg.get("default_timeout")
                if default_timeout is not None:
                    sandbox_kwargs["default_timeout"] = default_timeout
                supported_types = sandbox_cfg.get("supported_types")
                if supported_types:
                    sandbox_kwargs["supported_types"] = supported_types
                stdout_limit = sandbox_cfg.get("stdout_log_limit")
                if stdout_limit is not None:
                    sandbox_kwargs["stdout_log_limit"] = stdout_limit
                stderr_limit = sandbox_cfg.get("stderr_log_limit")
                if stderr_limit is not None:
                    sandbox_kwargs["stderr_log_limit"] = stderr_limit
                try:
                    sandbox_executor = SandboxExecutor(**sandbox_kwargs)
                    executors.append(sandbox_executor)
                    tool_registry.setdefault("sandbox_command", sandbox_executor)
                except Exception as exc:
                    print(f"[STO] Failed to initialise SandboxExecutor: {exc}", file=sys.stderr)
            else:
                print("[STO] Sandbox executor enabled but no command configured", file=sys.stderr)

        reminder_cfg = _to_dict(cfg.get("reminder"))
        if reminder_cfg.get("enabled", True):
            executors.append(ReminderExecutor())

        system_cfg = _to_dict(cfg.get("system_usage"))
        if system_cfg.get("enabled", True):
            interval = float(system_cfg.get("sample_interval", 0.2))
            system_executor = SystemUsageExecutor(sample_interval=interval)
            executors.append(system_executor)
            tool_registry.setdefault("system_usage", system_executor)

        llm_cfg = _to_dict(cfg.get("llm"))
        if llm_cfg.get("enabled", False):
            llm_kwargs: Dict[str, Any] = {}
            supported_types = llm_cfg.get("supported_types")
            if supported_types:
                llm_kwargs["supported_types"] = [
                    str(item) for item in supported_types if str(item)
                ]
            config_path_value = llm_cfg.get("config_path")
            if config_path_value:
                try:
                    llm_kwargs["config_path"] = Path(str(config_path_value)).expanduser()
                except Exception:
                    pass
            template_cfg = _to_dict(llm_cfg.get("template_config"))
            if template_cfg:
                llm_kwargs["template_config"] = template_cfg
            retry_limit_value = llm_cfg.get("retry_limit")
            if retry_limit_value is not None:
                try:
                    llm_kwargs["retry_limit"] = int(retry_limit_value)
                except (TypeError, ValueError):
                    pass
            output_filter = llm_cfg.get("output_filter")
            if output_filter is not None:
                llm_kwargs["output_filter"] = output_filter
            tool_registry_cfg = _to_dict(llm_cfg.get("tool_registry"))
            if tool_registry_cfg:
                resolved_registry: Dict[str, Any] = dict(tool_registry)
                for key, value in tool_registry_cfg.items():
                    if isinstance(value, str) and value in tool_registry:
                        resolved_registry[key] = tool_registry[value]
                    else:
                        resolved_registry[key] = value
                llm_kwargs["tool_registry"] = resolved_registry
            else:
                llm_kwargs["tool_registry"] = dict(tool_registry)
            try:
                executors.append(LLMExecutor(**llm_kwargs))
            except Exception as exc:
                print(f"[STO] Failed to initialise LLMExecutor: {exc}", file=sys.stderr)

        agent_cfg = _to_dict(cfg.get("agent"))
        if agent_cfg.get("enabled", False):
            agent_kwargs: Dict[str, Any] = {}
            supported_types = agent_cfg.get("supported_types")
            if supported_types:
                agent_kwargs["supported_types"] = supported_types
            max_steps = agent_cfg.get("max_steps")
            if max_steps is not None:
                agent_kwargs["max_steps"] = max_steps
            default_timeout = agent_cfg.get("default_command_timeout")
            if default_timeout is not None:
                agent_kwargs["default_command_timeout"] = default_timeout
            try:
                agent_executor = AgentExecutor(**agent_kwargs)
                executors.append(agent_executor)
                tool_registry.setdefault("agent_plan", agent_executor)
            except Exception as exc:
                print(f"[STO] Failed to initialise AgentExecutor: {exc}", file=sys.stderr)

        if not executors:
            # Fallback to at least a reminder executor, so scheduler has work to do.
            executors.append(ReminderExecutor())
        return executors, tool_registry

    def _close_store(self) -> None:
        connection = getattr(self.store, "con", None)
        with suppress(Exception):
            if connection is not None:
                connection.close()


__all__ = ["STOSchedulerService"]
