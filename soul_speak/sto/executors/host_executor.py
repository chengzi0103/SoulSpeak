"""Executor that runs whitelisted host OS commands."""
from __future__ import annotations

import contextlib
import asyncio
import os
import shlex
from asyncio.subprocess import PIPE
from typing import Iterable, Mapping, Optional, Sequence, Union

from soul_speak.sto.executors.base import Executor
from soul_speak.sto.models import Task, TaskLog
from soul_speak.sto.store.interface import TaskStoreProtocol


CommandType = Union[str, Sequence[str]]


class HostExecutor(Executor):
    """Executor that executes commands on the host operating system.

    The executor expects tasks to provide a payload with at least a
    ``command`` key. The value may be a string (executed through the shell)
    or a sequence of arguments passed directly to ``exec``.

    Optional payload keys:

    * ``shell`` – force shell invocation when ``command`` is a sequence
    * ``cwd`` – working directory for the process
    * ``env`` – mapping of environment variable overrides
    * ``timeout`` – execution timeout in seconds (float/int)

    To reduce the risk of arbitrary execution, callers can provide
    ``allowed_commands`` so only selected entrypoints are accepted.
    """

    def __init__(
        self,
        *,
        supported_types: Optional[Iterable[str]] = None,
        allowed_commands: Optional[Iterable[str]] = None,
        base_env: Optional[Mapping[str, str]] = None,
        stdout_log_limit: int = 4000,
        stderr_log_limit: int = 4000,
    ) -> None:
        self.supported_types = set(supported_types or {"host_command"})
        self.allowed_commands = set(allowed_commands or []) or None
        self.base_env = dict(base_env) if base_env else None
        self.stdout_log_limit = stdout_log_limit
        self.stderr_log_limit = stderr_log_limit

    def can_handle(self, task: Task) -> bool:
        return task.type in self.supported_types

    async def execute(self, task: Task, store: TaskStoreProtocol) -> None:
        await self._mark_running(task, store)

        payload = task.payload or {}
        command: Optional[CommandType] = payload.get("command")
        if not command:
            await self._finish_failed(task, store, "missing command in payload")
            return

        shell = bool(payload.get("shell", isinstance(command, str)))
        timeout_raw = payload.get("timeout")
        cwd = payload.get("cwd")
        env_overrides = payload.get("env")

        if cwd is not None and not isinstance(cwd, str):
            await self._finish_failed(task, store, "cwd must be a string path")
            return

        if env_overrides is not None and not isinstance(env_overrides, Mapping):
            await self._finish_failed(task, store, "env must be a mapping of environment overrides")
            return

        timeout: Optional[float]
        if timeout_raw is None:
            timeout = None
        else:
            try:
                timeout = float(timeout_raw)
            except (TypeError, ValueError):
                await self._finish_failed(task, store, "timeout must be a number of seconds")
                return
            if timeout <= 0:
                await self._finish_failed(task, store, "timeout must be greater than zero")
                return

        try:
            normalized = self._normalize_command(command)
        except ValueError as exc:
            await self._finish_failed(task, store, str(exc))
            return

        if self.allowed_commands and not self._is_allowed(normalized, shell):
            await self._finish_failed(task, store, "command not allowed by HostExecutor policy")
            return

        self._log_event(
            store,
            task,
            "tool_requested",
            "host command execution requested",
            {
                "command": normalized,
                "shell": shell,
                "cwd": cwd,
                "timeout": timeout,
            },
        )

        try:
            stdout, stderr, returncode = await self._run_command(
                normalized,
                shell=shell,
                cwd=cwd,
                env=env_overrides,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            task.result = {"stdout": "", "stderr": "", "returncode": "timeout"}
            self._log_event(
                store,
                task,
                "tool_failed",
                "command timed out",
                {
                    "command": normalized,
                    "shell": shell,
                    "cwd": cwd,
                    "timeout": timeout,
                },
            )
            await self._finish_failed(task, store, "command timed out")
            return
        except FileNotFoundError as exc:
            self._log_event(
                store,
                task,
                "tool_failed",
                f"command not found: {exc}",
                {
                    "command": normalized,
                    "shell": shell,
                    "cwd": cwd,
                    "timeout": timeout,
                },
            )
            await self._finish_failed(task, store, f"command not found: {exc}")
            return
        except Exception as exc:  # pylint: disable=broad-except
            self._log_event(
                store,
                task,
                "tool_failed",
                "command execution raised exception",
                {
                    "command": normalized,
                    "shell": shell,
                    "cwd": cwd,
                    "timeout": timeout,
                    "error": str(exc),
                },
            )
            await self._finish_failed(task, store, f"command execution failed: {exc}")
            return

        summary = stdout.strip() or stderr.strip()
        if summary:
            summary = summary.splitlines()[0]
        if summary and len(summary) > 200:
            summary = summary[:197] + "..."

        detailed_result = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
        }

        self._append_log(store, task, "stdout", stdout, self.stdout_log_limit)
        self._append_log(store, task, "stderr", stderr, self.stderr_log_limit)

        if returncode == 0:
            task.result = detailed_result
            self._log_event(
                store,
                task,
                "tool_completed",
                "command executed successfully",
                {
                    "command": normalized,
                    "shell": shell,
                    "cwd": cwd,
                    "timeout": timeout,
                    "returncode": returncode,
                },
            )
            await self._finish_success(task, store, result=summary or "command exited with 0")
        else:
            task.result = detailed_result
            self._log_event(
                store,
                task,
                "tool_failed",
                f"command exited with code {returncode}",
                {
                    "command": normalized,
                    "shell": shell,
                    "cwd": cwd,
                    "timeout": timeout,
                    "returncode": returncode,
                    "stdout": stdout[: self.stdout_log_limit],
                    "stderr": stderr[: self.stderr_log_limit],
                },
            )
            await self._finish_failed(task, store, f"command exited with code {returncode}")

    def _normalize_command(self, command: CommandType) -> CommandType:
        if isinstance(command, str):
            return command
        if isinstance(command, Sequence):
            if not command:
                raise ValueError("command sequence may not be empty")
            if not all(isinstance(item, str) for item in command):
                raise ValueError("command sequence must contain only strings")
            return tuple(command)
        raise ValueError("command must be a string or a sequence of strings")

    def _is_allowed(self, command: CommandType, shell: bool) -> bool:
        if not self.allowed_commands:
            return True
        head: Optional[str]
        if isinstance(command, str):
            if shell:
                try:
                    head = shlex.split(command, posix=os.name != "nt")[0]
                except (ValueError, IndexError):
                    head = None
            else:
                head = command
        else:
            head = command[0]
        return bool(head and head in self.allowed_commands)

    async def _run_command(
        self,
        command: CommandType,
        *,
        shell: bool,
        cwd: Optional[str],
        env: Optional[Mapping[str, str]],
        timeout: Optional[float],
    ) -> tuple[str, str, int]:
        env_vars: Optional[Mapping[str, str]]
        if self.base_env:
            env_vars = {**dict(os.environ), **self.base_env}
        else:
            env_vars = None

        if env:
            overrides = {str(k): str(v) for k, v in env.items()}
            base_env = dict(env_vars) if env_vars is not None else dict(os.environ)
            base_env.update(overrides)
            env_vars = base_env

        if shell:
            process = await asyncio.create_subprocess_shell(
                command if isinstance(command, str) else self._shell_join(command),
                stdout=PIPE,
                stderr=PIPE,
                cwd=cwd,
                env=env_vars,
            )
        else:
            if isinstance(command, str):
                args = shlex.split(command, posix=os.name != "nt")
            else:
                args = list(command)
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=PIPE,
                stderr=PIPE,
                cwd=cwd,
                env=env_vars,
            )

        communicate = process.communicate()
        try:
            if timeout is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(communicate, timeout=timeout)
            else:
                stdout_bytes, stderr_bytes = await communicate
        except asyncio.TimeoutError:
            process.kill()
            with contextlib.suppress(Exception):
                await process.communicate()
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        return stdout, stderr, process.returncode

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

    def _shell_join(self, command: Sequence[str]) -> str:
        try:
            join = shlex.join
        except AttributeError:  # pragma: no cover - Python < 3.8 safeguard
            return " ".join(shlex.quote(part) for part in command)
        return join(command)


__all__ = ["HostExecutor"]
