"""Executor that delegates task handling to an LLM agent with tool awareness."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml
from attrs import define

from soul_speak.sto.executors.base import Executor
from soul_speak.sto.models import Task, TaskStatus
from soul_speak.sto.store.interface import TaskStoreProtocol

try:  # Import lazily so tests can inject dummy agents without heavy deps.
    from soul_speak.llm.openai_native import build_agent
except Exception:  # pragma: no cover - fallback when dependencies are missing
    build_agent = None  # type: ignore


DEFAULT_CONFIG: Dict[str, Any] = {
    "defaults": {
        "system_prompt": "你是 SoulSpeak 的执行助手，需要根据任务要求调用工具或生成答案，输出必须清晰、可靠，并遵守安全策略。",
        "user_template": "{context_block}\n{conversation_block}\n### 当前任务\n{prompt}",
        "params": {
            "temperature": 0.3,
            "max_tokens": 800,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "output_guidelines": "- 使用中文回答，除非用户明确要求其他语言。\n- 回答需给出关键结论与必要的步骤或建议。\n- 如果缺失信息，应明确指出并请求补充。",
    },
    "types": {
        "llm": {
            "description": "通用执行任务。",
        },
        "plan_and_execute": {
            "system_prompt": "你是 SoulSpeak 的任务执行助手，需要对用户目标给出一步到位的处理或明确后续操作建议。",
            "user_template": "{context_block}\n{conversation_block}\n### 任务说明\n{prompt}\n\n如需操作工具，请给出即将执行的命令或步骤，并说明目的。",
            "params": {"temperature": 0.2, "max_tokens": 1200},
            "output_guidelines": "- 如果能够直接解决，请提供具体输出。\n- 如果需要进一步操作，列出清晰步骤和所需工具。\n- 标明潜在风险或注意事项。",
        },
        "qa": {
            "system_prompt": "你是 SoulSpeak 的知识助手，需要结合上下文信息准确回答问题。",
            "user_template": "{context_block}\n### 问题\n{prompt}",
            "output_guidelines": "- 优先基于上下文回答，并引用关键句。\n- 若无法回答，请坦诚说明。",
        },
    },
}


@define(init=False, slots=False)
class LLMExecutor(Executor):
    """Executor that relies on an async LLM agent to produce a textual result.

    The executor expects task payloads to contain at least a ``prompt`` field.
    Optional keys:
      * ``context`` – extra context prepended to the prompt
      * ``conversation`` – 历史对话，列表形式，每项包含 ``role``/``content``
      * ``metadata`` – arbitrary metadata recorded alongside the result
      * ``llm_params`` – overrides for模型参数（温度、max_tokens等）
      * ``tools`` – 工具列表（透传给代理）
      * ``stop_sequences`` – 终止生成的字符串列表
    """

    def __init__(
        self,
        agent: Optional[Any] = None,
        supported_types: Optional[Iterable[str]] = None,
        *,
        config_path: Optional[Path] = None,
        template_config: Optional[Mapping[str, Any]] = None,
        output_filter: Optional[Any] = None,
        retry_limit: int = 0,
        prompt_log_limit: int = 1000,
        response_log_limit: int = 2000,
        tool_registry: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._agent = agent
        self.supported_types = set(supported_types or {"plan_and_execute", "llm", "qa"})
        self.retry_limit = max(0, retry_limit)
        self.prompt_log_limit = prompt_log_limit
        self.response_log_limit = response_log_limit
        self.output_filter = output_filter
        self.tool_registry: Dict[str, Any] = dict(tool_registry or {})

        self._config_path = config_path
        self._template_config = template_config or self._load_config(config_path)

    def can_handle(self, task: Task) -> bool:
        return task.type in self.supported_types

    async def execute(self, task: Task, store: TaskStoreProtocol) -> None:
        await self._mark_running(task, store)

        payload: Dict[str, Any] = task.payload or {}
        prompt = payload.get("prompt") or payload.get("instruction")
        if not prompt:
            await self._finish_failed(task, store, "missing prompt in payload")
            return

        template = self._resolve_template(task.type)
        plan_data = payload.get("plan") if isinstance(payload.get("plan"), dict) else None
        metadata: Dict[str, Any] = {}
        reply_preview = ""

        plan_attempts = 0
        plan_duration_ms: Optional[int] = None

        if plan_data is None:
            rendered_prompt = self._render_prompt(template, prompt, payload, mode="planner")

            merged_params = dict(template.get("params", {}))
            merged_params.update(payload.get("llm_params", {}))

            call_kwargs: Dict[str, Any] = {
                key: value
                for key, value in merged_params.items()
                if value is not None
            }

            tools = payload.get("tools")
            if tools:
                call_kwargs["tools"] = tools
            stop_sequences = payload.get("stop_sequences")
            if stop_sequences:
                call_kwargs["stop_sequences"] = stop_sequences

            self._log_event(
                store,
                task,
                "prompt_compiled",
                "planner prompt prepared",
                {
                    "task_type": task.type,
                    "prompt_preview": rendered_prompt[: self.prompt_log_limit],
                    "params": call_kwargs,
                },
            )

            attempts = 0
            start_time = time.perf_counter()
            last_error: Optional[str] = None
            while attempts <= self.retry_limit:
                attempts += 1
                try:
                    response = await self._call_agent(rendered_prompt, call_kwargs)
                    duration_ms = int((time.perf_counter() - start_time) * 1000)
                    plan_attempts = attempts
                    plan_duration_ms = duration_ms

                    plan_data, reply_preview, metadata = self._extract_plan(response)
                    if plan_data is None:
                        raise ValueError("agent response missing plan")

                    self._log_event(
                        store,
                        task,
                        "plan_generated",
                        "plan produced by agent",
                        {
                            "attempts": attempts,
                            "plan": plan_data.get("plan", []),
                            "final_summary": plan_data.get("final_summary"),
                        },
                    )

                    break
                except Exception as exc:  # pylint: disable=broad-except
                    last_error = str(exc)
                    self._log_event(
                        store,
                        task,
                        "llm_error",
                        "agent call or plan generation failed",
                        {
                            "attempt": attempts,
                            "error": last_error,
                        },
                    )
                    if attempts > self.retry_limit:
                        break
                    await asyncio.sleep(min(0.5 * attempts, 2.0))

            if plan_data is None:
                fallback_plan = self._heuristic_plan(prompt)
                if fallback_plan:
                    plan_data = fallback_plan
                    reply_preview = fallback_plan.get("final_summary", "")
                    metadata = fallback_plan.get("metadata", {})
                    self._log_event(
                        store,
                        task,
                        "plan_generated",
                        "plan produced by heuristic fallback",
                        {
                            "attempts": plan_attempts or attempts,
                            "plan": plan_data.get("plan", []),
                            "final_summary": plan_data.get("final_summary"),
                        },
                    )
                else:
                    await self._finish_failed(task, store, last_error or "agent response missing plan")
                    return
        else:
            self._log_event(
                store,
                task,
                "plan_loaded",
                "plan provided in payload",
                {"plan": plan_data.get("plan", [])},
            )

        if plan_data is None:
            await self._finish_failed(task, store, "planner returned no plan")
            return

        metrics = {
            "plan_generated": bool(payload.get("plan") is None),
        }
        if plan_attempts:
            metrics["plan_generation_attempts"] = plan_attempts
        if plan_duration_ms is not None:
            metrics["plan_generation_duration_ms"] = plan_duration_ms
        token_usage = self._extract_token_usage(metadata)
        if token_usage:
            metrics["token_usage"] = token_usage
        step_results, plan_success = await self._execute_plan(plan_data.get("plan", []), task, store)

        final_summary = plan_data.get("final_summary") or reply_preview or ""

        if self.output_filter and final_summary:
            try:
                filter_result = self.output_filter(final_summary, metadata)
                if asyncio.iscoroutine(filter_result):
                    filter_result = await filter_result  # type: ignore[assignment]
                if not filter_result:
                    self._log_event(
                        store,
                        task,
                        "llm_filtered",
                        "output rejected by filter",
                    )
                    await self._finish_failed(task, store, "response rejected by output filter")
                    return
            except Exception as filter_exc:  # pragma: no cover
                self._log_event(
                    store,
                    task,
                    "llm_filtered",
                    "output filter raised an error",
                    {"error": str(filter_exc)},
                )
                await self._finish_failed(task, store, f"output filter failed: {filter_exc}")
                return

        task.result = {
            "plan": plan_data,
            "step_results": step_results,
            "reply": final_summary,
            "metadata": metadata,
            "metrics": metrics,
        }

        self._log_event(
            store,
            task,
            "plan_completed" if plan_success else "plan_failed",
            "plan execution finished" if plan_success else "plan execution failed",
            {
                "success": plan_success,
                "final_summary": final_summary,
            },
        )

        if plan_success:
            summary = final_summary.strip().splitlines()[0] if final_summary.strip() else "ok"
            await self._finish_success(task, store, result=summary)
        else:
            await self._finish_failed(task, store, "one or more plan steps failed")

    async def _call_agent(self, prompt: str, params: Dict[str, Any]) -> Any:
        agent = await self._ensure_agent()
        try:
            return await agent.generate(prompt, **params)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" in message:
                return await agent.generate(prompt)
            raise

    async def _ensure_agent(self) -> Any:
        if self._agent is not None:
            return self._agent
        if build_agent is None:
            raise RuntimeError("LLM agent not available; set agent=... when instantiating LLMExecutor")
        self._agent = build_agent()
        return self._agent

    def _render_prompt(
        self,
        template: Mapping[str, Any],
        prompt: str,
        payload: Mapping[str, Any],
        *,
        mode: str = "planner",
    ) -> str:
        context = payload.get("context")
        context_block = f"### 上下文\n{context}\n" if context else ""

        conversation_block = ""
        conversation = payload.get("conversation")
        if isinstance(conversation, list) and conversation:
            lines = []
            for turn in conversation:
                role = str(turn.get("role", "user")).upper()
                content = str(turn.get("content", ""))
                lines.append(f"{role}: {content}")
            conversation_block = "### 历史对话\n" + "\n".join(lines) + "\n"

        filled = template["user_template"].format(
            context_block=context_block,
            conversation_block=conversation_block,
            prompt=prompt,
        ).strip()

        system_prompt = template.get("system_prompt")
        output_guidelines = template.get("output_guidelines")
        planner_instructions = template.get("planner_instructions")

        full_prompt_parts = []
        if system_prompt:
            full_prompt_parts.append(system_prompt.strip())
        full_prompt_parts.append(filled)
        if mode == "planner" and planner_instructions:
            full_prompt_parts.append("### 计划要求\n" + planner_instructions.strip())
        elif output_guidelines:
            full_prompt_parts.append("### 输出要求\n" + output_guidelines.strip())

        return "\n\n".join(part for part in full_prompt_parts if part)

    def _extract_plan(self, response: Any) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        reply: str = ""
        raw: Any = response
        if isinstance(response, str):
            reply = response
            try:
                raw = json.loads(response)
            except json.JSONDecodeError:
                return None, reply, metadata
        elif hasattr(response, "text"):
            reply = str(getattr(response, "text"))
            metadata = getattr(response, "metadata", {}) or {}
            try:
                raw = json.loads(reply)
            except json.JSONDecodeError:
                raw = response
        elif isinstance(response, dict):
            reply = (
                response.get("text")
                or response.get("reply")
                or response.get("content")
                or response.get("output")
                or ""
            )
            metadata = response.get("metadata") or {
                k: v for k, v in response.items() if k not in {"text", "reply", "content", "output"}
            }
            raw = response

        plan_candidate: Optional[Dict[str, Any]] = None

        if isinstance(raw, dict) and "plan" in raw:
            plan_candidate = raw
        elif isinstance(raw, dict) and "choices" in raw:
            try:
                content = raw["choices"][0]["message"]["content"]
                plan_candidate = json.loads(content)
                reply = content
            except Exception:  # pragma: no cover
                plan_candidate = None

        if plan_candidate is None and reply:
            try:
                parsed = json.loads(reply)
                if isinstance(parsed, dict) and "plan" in parsed:
                    plan_candidate = parsed
            except json.JSONDecodeError:
                pass

        return plan_candidate, reply, metadata

    def _extract_token_usage(self, metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        usage = metadata.get("usage") if isinstance(metadata, Mapping) else None
        if isinstance(usage, Mapping):
            return dict(usage)
        return None

    def _resolve_template(self, task_type: str) -> Mapping[str, Any]:
        defaults = self._template_config.get("defaults", {})
        type_cfg = self._template_config.get("types", {}).get(task_type, {})

        merged_params = dict(defaults.get("params", {}))
        merged_params.update(type_cfg.get("params", {}) or {})

        template = {
            "system_prompt": type_cfg.get("system_prompt") or defaults.get("system_prompt"),
            "user_template": type_cfg.get("user_template") or defaults.get("user_template", "{prompt}"),
            "output_guidelines": type_cfg.get("output_guidelines") or defaults.get("output_guidelines"),
            "params": merged_params,
        }
        return template

    def _load_config(self, config_path: Optional[Path]) -> Mapping[str, Any]:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[3] / "conf" / "sto" / "llm_executor.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or DEFAULT_CONFIG
        except FileNotFoundError:
            return DEFAULT_CONFIG
        except Exception as exc:  # pragma: no cover - config errors should not crash executor
            raise RuntimeError(f"failed to load LLM executor config: {exc}") from exc

    def _heuristic_plan(self, prompt: str) -> Optional[Dict[str, Any]]:
        normalized = prompt.lower()
        steps: list[Dict[str, Any]] = []

        def make_step(
            step_id: str,
            description: str,
            *,
            task_type: str = "host_command",
            command: list[str] | str | None = None,
            timeout: float = 120.0,
            shell: bool = False,
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {"timeout": timeout}
            if command is not None:
                payload["command"] = command
            if shell:
                payload["shell"] = True
            return {
                "id": step_id,
                "task_type": task_type,
                "description": description,
                "payload": payload,
            }

        if "ffmpeg" in normalized and any(word in normalized for word in ("install", "安装", "安裝")):
            steps.append(
                make_step(
                    "step-install-ffmpeg",
                    "使用 brew 安装 ffmpeg",
                    task_type="host_command",
                    command=["brew", "install", "ffmpeg"],
                    timeout=600.0,
                )
            )

        system_keywords = ["系统占用", "system usage", "cpu", "内存", "memory", "monitor"]
        if any(keyword in normalized for keyword in system_keywords):
            if "linux" in normalized:
                steps.append(
                    make_step(
                        "step-check-usage",
                        "使用 top 查看 Linux 系统占用",
                        task_type="host_command",
                        command=["top", "-b", "-n", "1"],
                        timeout=90.0,
                    )
                )
            else:
                steps.append(
                    make_step(
                        "step-check-usage",
                        "使用内置采集器获取系统占用",
                        task_type="system_usage",
                        command=None,
                        timeout=30.0,
                    )
                )

        if not steps:
            return None

        return {
            "plan": steps,
            "final_summary": "执行计划已通过规则生成，请检查每步结果。",
            "metadata": {"generated_by": "heuristic"},
        }

    async def _execute_plan(
        self,
        plan_steps: Sequence[Mapping[str, Any]],
        parent_task: Task,
        store: TaskStoreProtocol,
    ) -> Tuple[Sequence[Dict[str, Any]], bool]:
        step_results: list[Dict[str, Any]] = []
        all_success = True

        for index, step in enumerate(plan_steps or []):
            step_id = str(step.get("id") or f"step-{index+1}")
            task_type = str(step.get("task_type") or step.get("tool") or "").strip()
            payload = step.get("payload") or {}
            description = step.get("description") or step.get("summary") or ""

            self._log_event(
                store,
                parent_task,
                "plan_step_started",
                f"step {step_id} started",
                {
                    "step_id": step_id,
                    "task_type": task_type,
                    "description": description,
                },
            )

            if not task_type:
                self._log_event(
                    store,
                    parent_task,
                    "plan_step_failed",
                    f"step {step_id} missing task_type",
                    {"step_id": step_id},
                )
                step_results.append(
                    {
                        "step_id": step_id,
                        "status": TaskStatus.FAILED.value,
                        "error": "missing task_type",
                    }
                )
                all_success = False
                break

            child_task_id = f"{parent_task.id}::{step_id}"
            child_task = Task(
                id=child_task_id,
                type=task_type,
                payload=dict(payload),
                status=TaskStatus.PENDING,
            )

            try:
                store.create_task(child_task)
            except Exception as exc:
                self._log_event(
                    store,
                    parent_task,
                    "plan_step_failed",
                    f"step {step_id} create_task failed",
                    {"error": str(exc)},
                )
                step_results.append(
                    {
                        "step_id": step_id,
                        "status": TaskStatus.FAILED.value,
                        "error": str(exc),
                    }
                )
                all_success = False
                break

            tool = self.tool_registry.get(task_type)
            if tool is None:
                self._log_event(
                    store,
                    parent_task,
                    "plan_step_failed",
                    f"step {step_id} tool not registered",
                    {"task_type": task_type},
                )
                step_results.append(
                    {
                        "step_id": step_id,
                        "task_id": child_task_id,
                        "status": TaskStatus.FAILED.value,
                        "error": "tool not registered",
                    }
                )
                all_success = False
                break

            try:
                if isinstance(tool, Executor):
                    await tool.execute(child_task, store)
                elif callable(tool):
                    result = tool(child_task, store)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    raise TypeError(f"unsupported tool type for {task_type}")
            except Exception as exc:  # pylint: disable=broad-except
                self._log_event(
                    store,
                    parent_task,
                    "plan_step_failed",
                    f"step {step_id} execution raised",
                    {"error": str(exc)},
                )
                step_results.append(
                    {
                        "step_id": step_id,
                        "task_id": child_task_id,
                        "status": TaskStatus.FAILED.value,
                        "error": str(exc),
                    }
                )
                all_success = False
                break

            updated_child = store.get_task(child_task_id) or child_task
            success = updated_child.status == TaskStatus.SUCCESS
            step_results.append(
                {
                    "step_id": step_id,
                    "task_id": child_task_id,
                    "status": updated_child.status.value,
                    "result": updated_child.result,
                    "error": updated_child.error,
                }
            )

            if success:
                self._log_event(
                    store,
                    parent_task,
                    "plan_step_completed",
                    f"step {step_id} completed",
                    {
                        "task_id": child_task_id,
                        "description": description,
                    },
                )
            else:
                self._log_event(
                    store,
                    parent_task,
                    "plan_step_failed",
                    f"step {step_id} failed",
                    {
                        "task_id": child_task_id,
                        "status": updated_child.status.value,
                        "error": updated_child.error,
                    },
                )
                all_success = False
                break

        return step_results, all_success


__all__ = ["LLMExecutor"]
