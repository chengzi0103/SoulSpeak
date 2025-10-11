"""Executor that runs approved scripts inside a sandbox entrypoint."""
from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from attrs import define, field

from soul_speak.sto.executors.base import Executor
from soul_speak.sto.models import Task, TaskLog
from soul_speak.sto.store.interface import TaskStoreProtocol


def _ensure_sequence(value: Sequence[str] | str) -> Sequence[str]:
    if isinstance(value, str):
        return tuple(part for part in shlex.split(value) if part)
    return tuple(str(part) for part in value if str(part))


@define
class SandboxExecutor(Executor):
    """Execute pre-approved scripts via a sandbox runner (Docker, gVisor, etc.).

    Configuration expectations:
    * ``sandbox_cmd``: base command/entrypoint executed for every task
    * ``allowed_paths``: directories whose scripts may be executed
    * ``supported_types``: task types handled (default ``{"sandbox_command"}``)

    Task payload requirements:
    * ``path``: script path (must live under one of the allowed directories)
    * ``args``: optional list/tuple of arguments passed to the script
    * ``env``: optional environment variable overrides
    * ``timeout``: optional per-task timeout (seconds)
    """

    sandbox_cmd: Sequence[str] = field(converter=_ensure_sequence)
    allowed_paths: Sequence[Path] = field(
        factory=tuple,
        converter=lambda items: tuple(Path(p).expanduser().resolve(strict=False) for p in items),
    )
    supported_types: Iterable[str] = field(factory=lambda: {"sandbox_command"})
    base_env: Optional[Mapping[str, str]] = None
    default_timeout: Optional[float] = None
    stdout_log_limit: int = 4000
    stderr_log_limit: int = 4000
    env_var_name: str = "SANDBOX_ALLOWED_ROOTS"

    def __attrs_post_init__(self) -> None:
        if not self.sandbox_cmd:
            raise ValueError("SandboxExecutor requires a non-empty sandbox_cmd")
        self.supported_types = set(self.supported_types)
        normalized: list[Path] = []
        for path in self.allowed_paths:
            resolved = path.resolve()
            if not resolved.exists():
                resolved.parent.mkdir(parents=True, exist_ok=True)
            normalized.append(resolved if resolved.is_dir() else resolved.parent)
        self.allowed_paths = tuple(normalized)
        self._allowed_env_value = os.pathsep.join(str(path) for path in self.allowed_paths)

    def can_handle(self, task: Task) -> bool:  # type: ignore[override]
        return task.type in self.supported_types

    async def execute(self, task: Task, store: TaskStoreProtocol) -> None:  # type: ignore[override]
        await self._mark_running(task, store)

        payload = task.payload or {}
        path_value = payload.get("path")
        if not path_value:
            await self._finish_failed(task, store, "missing script path for sandbox execution")
            return

        script_path = Path(str(path_value)).expanduser().resolve()
        if not script_path.exists():
            await self._finish_failed(task, store, f"script not found: {script_path}")
            return

        if self.allowed_paths and not self._is_allowed(script_path):
            await self._finish_failed(task, store, f"script {script_path} is outside allowed sandbox paths")
            return

        args_value = payload.get("args") or []
        if isinstance(args_value, str):
            args = shlex.split(args_value)
        elif isinstance(args_value, Sequence):
            args = [str(item) for item in args_value]
        else:
            await self._finish_failed(task, store, "args must be a sequence or string")
            return

        timeout_value = payload.get("timeout", self.default_timeout)
        try:
            timeout = float(timeout_value) if timeout_value is not None else None
            if timeout is not None and timeout <= 0:
                raise ValueError
        except (TypeError, ValueError):
            await self._finish_failed(task, store, "timeout must be a positive number")
            return

        env_overrides = payload.get("env")
        if env_overrides is not None and not isinstance(env_overrides, Mapping):
            await self._finish_failed(task, store, "env must be a mapping of environment overrides")
            return

        extra_env: dict[str, str] = {}
        if self.allowed_paths:
            extra_env[self.env_var_name] = self._allowed_env_value

        if env_overrides:
            extra_env.update({str(k): str(v) for k, v in env_overrides.items()})

        command = list(self.sandbox_cmd) + [str(script_path)] + list(args)

        self._log_event(
            store,
            task,
            "sandbox_requested",
            "sandbox execution requested",
            {
                "command": command,
                "timeout": timeout,
                "script": str(script_path),
            },
        )

        try:
            stdout, stderr, returncode = await self._run_command(
                command,
                env=extra_env,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            await self._finish_failed(task, store, f"sandbox command timed out after {timeout}s")
            return
        except FileNotFoundError as exc:
            await self._finish_failed(task, store, f"sandbox runner unavailable: {exc}")
            return
        except Exception as exc:  # pylint: disable=broad-except
            await self._finish_failed(task, store, f"sandbox execution failed: {exc}")
            return

        self._append_log(store, task, "stdout", stdout, self.stdout_log_limit)
        self._append_log(store, task, "stderr", stderr, self.stderr_log_limit)

        task.result = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

        if returncode == 0:
            summary = stdout.strip().splitlines()[0] if stdout.strip() else "sandbox exited with 0"
            await self._finish_success(task, store, result=summary)
        else:
            await self._finish_failed(task, store, f"sandbox exited with code {returncode}")

    def _is_allowed(self, target: Path) -> bool:
        for base in self.allowed_paths:
            try:
                target.relative_to(base)
                return True
            except ValueError:
                continue
        return False

    async def _run_command(
        self,
        command: Sequence[str],
        *,
        env: Optional[Mapping[str, str]],
        timeout: Optional[float],
    ) -> tuple[str, str, int]:
        env_vars = {**os.environ}
        if self.base_env:
            env_vars.update({str(k): str(v) for k, v in self.base_env.items()})
        if env:
            env_vars.update({str(k): str(v) for k, v in env.items()})

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=PIPE,
            stderr=PIPE,
            env=env_vars,
        )

        try:
            if timeout is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.kill()
            with contextlib.suppress(Exception):
                await process.communicate()
            raise

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        return stdout_text, stderr_text, process.returncode

    def _append_log(
        self,
        store: TaskStoreProtocol,
        task: Task,
        event: str,
        message: str,
        limit: int,
    ) -> None:
        if not message:
            return
        snippet = message[-limit:]
        store.append_log(TaskLog(task_id=task.id, event=event, message=snippet))


__all__ = ["SandboxExecutor"]
