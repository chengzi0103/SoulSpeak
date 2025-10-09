"""基于原生 OpenAI SDK 的对话代理，使用 Ray 异步写入 Mem0 记忆并输出性能指标。"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Deque, Dict, List, Optional, Set

from attrs import define, field
from dotenv import load_dotenv
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
import ray

from soul_speak.llm.memory import Mem0MemoryStore
from soul_speak.utils.hydra_config.init import conf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@ray.remote
class Mem0MemoryActor:
    """Ray actor 封装的 Mem0 写入/检索服务。"""

    def __init__(self, store_kwargs: Dict[str, Any]) -> None:
        self.store = Mem0MemoryStore(**store_kwargs)

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
    ) -> Any:
        return self.store.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
        )

    def store_turn(
        self,
        user_message: str,
        assistant_message: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.store.store_turn(
            user_message=user_message,
            assistant_message=assistant_message,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )


@define
class OpenAINativeAgent:
    """直接调用 OpenAI 官方客户端的异步对话代理。"""

    model: str
    temperature: float
    prompt_template: str
    history_window: int = 10
    base_url: Optional[str] = None

    client: AsyncOpenAI = field(init=False)
    chat_history: Deque[str] = field(init=False)

    memory_enabled: bool = field(init=False, default=False)
    memory_actor: Optional["ray.actor.ActorHandle"] = field(init=False, default=None)
    memory_user_id: Optional[str] = field(init=False, default=None)
    memory_agent_id: Optional[str] = field(init=False, default=None)
    memory_run_id: Optional[str] = field(init=False, default=None)
    memory_default_metadata: Dict[str, Any] = field(init=False, factory=dict)

    tools_enabled: bool = field(init=False, default=True)
    tool_services: Optional[Set[str]] = field(init=False, default=None)
    tool_specs: List[Dict[str, Any]] = field(init=False, factory=list)
    tool_map: Dict[str, Any] = field(init=False, factory=dict)
    tools_initialized: bool = field(init=False, default=False)
    tool_lock: asyncio.Lock = field(init=False, repr=False)
    mcp_manager: Optional[Any] = field(init=False, default=None, repr=False)
    ray_initialized_here: bool = field(init=False, default=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._load_env()
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError("未找到可用的 API Key，请设置 OPENAI_API_KEY、DEEPSEEK_API_KEY 或 LLM_API_KEY。")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = AsyncOpenAI(**client_kwargs)
        self.chat_history = deque(maxlen=self.history_window)
        self.tool_lock = asyncio.Lock()

        self._configure_tools()

        memory_conf = getattr(conf.llm, "memory", None)
        if memory_conf is not None:
            memory_dict = OmegaConf.to_container(memory_conf, resolve=True)
            self._setup_memory(memory_dict)

    @staticmethod
    def _load_env() -> None:
        load_dotenv()
        repo_root = Path(__file__).resolve().parents[2]
        load_dotenv(repo_root / ".env")
        load_dotenv((repo_root / "soul_speak" / ".env"))

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o777)
        except Exception:
            pass

    def _setup_memory(self, memory_conf: Dict[str, Any]) -> None:
        if not memory_conf:
            return
        if not memory_conf.get("enable", True):
            logger.info("Mem0 记忆已在配置中禁用")
            return

        mode = str(memory_conf.get("mode", "local")).lower()
        self.memory_user_id = memory_conf.get("default_user_id")
        self.memory_agent_id = memory_conf.get("default_agent_id")
        self.memory_run_id = memory_conf.get("default_run_id")
        default_metadata = memory_conf.get("default_metadata") or {}
        if not isinstance(default_metadata, dict):
            default_metadata = dict(default_metadata)
        self.memory_default_metadata = default_metadata

        store_kwargs: Dict[str, Any] = {
            "mode": mode,
            "default_user_id": self.memory_user_id,
            "default_agent_id": self.memory_agent_id,
            "default_run_id": self.memory_run_id,
            "default_metadata": default_metadata,
        }
        for key in ("api_key", "host", "org_id", "project_id"):
            value = memory_conf.get(key)
            if value:
                store_kwargs[key] = value

        conf_llm_dir = Path(__file__).resolve().parents[2] / "conf" / "llm"
        demo_root = Path(__file__).resolve().parent / "memory" / "mem0_demo_data"
        self._ensure_dir(demo_root)

        if mode == "local":
            local_config = deepcopy(memory_conf.get("local_config") or {})
            if not local_config:
                logger.warning("memory.local_config 缺失，跳过记忆初始化")
                return

            llm_section = local_config.setdefault("llm", {})
            embed_section = local_config.setdefault("embedder", {})
            llm_cfg = llm_section.setdefault("config", {})
            embed_cfg = embed_section.setdefault("config", {})

            provider_cfgs = (
                ("llm", llm_section.get("provider"), llm_cfg),
                ("embedder", embed_section.get("provider"), embed_cfg),
            )

            api_key = self._resolve_api_key()
            if api_key:
                for _, provider_name, cfg in provider_cfgs:
                    if not self._requires_api_key(provider_name, cfg):
                        continue
                    current = str(cfg.get("api_key") or "").strip()
                    if not current or current.lower().startswith("your-"):
                        cfg["api_key"] = api_key
            else:
                missing = [
                    label
                    for label, provider_name, cfg in provider_cfgs
                    if self._requires_api_key(provider_name, cfg)
                    and (
                        not cfg.get("api_key")
                        or str(cfg.get("api_key")).strip().lower().startswith("your-")
                    )
                ]
                if missing:
                    logger.warning(
                        "缺少 API Key（OPENAI_API_KEY / DEEPSEEK_API_KEY / LLM_API_KEY），无法启用 Mem0 记忆"
                    )
                    return

            vector_cfg = local_config.setdefault("vector_store", {}).setdefault("config", {})
            raw_vector_path = vector_cfg.get("path") or "../llm/memory/mem0_demo_data/chroma"
            vector_path = Path(raw_vector_path).expanduser()
            if not vector_path.is_absolute():
                vector_path = (conf_llm_dir / raw_vector_path).resolve()
            vector_cfg["path"] = str(vector_path)
            self._ensure_dir(vector_path)

            raw_history = local_config.get("history_db_path") or "../llm/memory/mem0_demo_data/history.db"
            history_path = Path(raw_history).expanduser()
            if not history_path.is_absolute():
                history_path = (conf_llm_dir / raw_history).resolve()
            self._ensure_dir(history_path.parent)
            local_config["history_db_path"] = str(history_path)

            store_kwargs["local_config"] = local_config
        elif mode != "cloud":
            logger.warning("不支持的 memory.mode=%s，跳过记忆初始化", mode)
            return

        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
                self.ray_initialized_here = True
            self.memory_actor = Mem0MemoryActor.remote(store_kwargs)
            self.memory_enabled = True
            logger.info("Mem0 记忆已启用 (Ray actor)")
        except Exception as exc:
            logger.exception("初始化 Mem0 记忆失败: %s", exc)

    def _configure_tools(self) -> None:
        tools_conf = getattr(conf.llm, "tools", None)
        if tools_conf is None:
            self.tools_enabled = False
            return

        enabled = getattr(tools_conf, "enable", True)
        self.tools_enabled = bool(enabled)
        services = getattr(tools_conf, "services", None)
        if services:
            try:
                self.tool_services = {str(item) for item in services}
            except TypeError:
                self.tool_services = {str(services)}
        else:
            self.tool_services = None

    async def _ensure_tools(self) -> List[Dict[str, Any]]:
        if not self.tools_enabled:
            return []

        if self.tools_initialized:
            return self.tool_specs

        async with self.tool_lock:
            if self.tools_initialized:
                return self.tool_specs

            try:
                from soul_speak.llm.mcp_manager import MCPConfig, MCPManager
            except Exception as exc:  # pragma: no cover
                logger.error("无法导入 MCP 管理器: %s", exc)
                self.tools_enabled = False
                return []

            services_data = dict(conf.mcp)
            mcp_configs = []
            services = services_data.get('services', {})
            for service_name, service_config in services.items():
                if self.tool_services and service_name not in self.tool_services:
                    continue
                config = dict(service_config)
                if not config.get('enabled', False):
                    continue
                mcp_configs.append(MCPConfig(
                    name=config.get('name', service_name),
                    url=config.get('url', ''),
                    description=config.get('description', ''),
                    type=config.get('type', 'sse'),
                    enabled=config.get('enabled', True),
                    timeout=config.get('timeout', 30)
                ))

            if not mcp_configs:
                logger.info("工具配置为空或已被过滤，跳过 MCP 初始化")
                self.tools_enabled = False
                self.tools_initialized = True
                return []

            try:
                self.mcp_manager = MCPManager(mcp_configs)
                await self.mcp_manager.initialize()
                tools = self.mcp_manager.get_tools()
                self.tool_map = {tool.name: tool for tool in tools}
                self.tool_specs = [convert_to_openai_tool(tool) for tool in tools]
                logger.info("Loaded %d MCP tools for native agent", len(self.tool_specs))
            except Exception as exc:
                logger.exception("初始化 MCP 工具失败: %s", exc)
                self.tool_specs = []
                self.tool_map = {}
                self.tools_enabled = False
            finally:
                self.tools_initialized = True

        return self.tool_specs

    async def _memory_search(self, user_input: str) -> List[str]:
        if not self.memory_enabled or not self.memory_actor:
            return []
        timeout = 5.0
        if hasattr(conf.llm.memory, "search_timeout"):
            timeout = float(conf.llm.memory.search_timeout)
        try:
            result_ref = self.memory_actor.search.remote(
                user_input,
                user_id=self.memory_user_id,
                agent_id=self.memory_agent_id,
                run_id=self.memory_run_id,
                limit=5,
            )
            fetch_task = asyncio.create_task(asyncio.to_thread(ray.get, result_ref))
            result = await asyncio.wait_for(fetch_task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("检索记忆超时(%.1fs)，跳过本轮记忆", timeout)
            return []
        except Exception as exc:
            logger.warning("检索记忆失败: %s", exc)
            return []

        items = result.get("results", []) if isinstance(result, dict) else result or []
        snippets: List[str] = []
        for item in items:
            if isinstance(item, dict):
                memory_text = item.get("memory")
            else:
                memory_text = None
            if memory_text:
                snippets.append(f"- {memory_text}")
        return snippets

    async def generate(self, user_input: str) -> str:
        total_start = perf_counter()
        mem_search_duration = 0.0
        openai_duration = 0.0

        history_text = "\n".join(self.chat_history)
        memory_context = ""

        if self.memory_enabled and self.memory_actor:
            mem_start = perf_counter()
            snippets = await self._memory_search(user_input)
            mem_search_duration = perf_counter() - mem_start
            if snippets:
                memory_context = "### Relevant Memories\n" + "\n".join(snippets)

        if memory_context:
            history_text = f"{memory_context}\n\n{history_text}" if history_text else memory_context

        prompt = self.prompt_template.format(
            chat_history=history_text or "(无历史对话)",
            user_input=user_input,
        )

        tool_specs = await self._ensure_tools()

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]

        async def _call_openai(req_messages: List[Dict[str, Any]]):
            params: Dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": req_messages,
            }
            if tool_specs:
                params["tools"] = tool_specs
                params["tool_choice"] = "auto"
            return await self.client.chat.completions.create(**params)

        openai_start = perf_counter()
        response = await _call_openai(messages)
        openai_duration = perf_counter() - openai_start

        if tool_specs:
            loop_guard = 0
            max_loops = 5
            while response.choices[0].message.tool_calls and loop_guard < max_loops:
                loop_guard += 1
                assistant_msg = response.choices[0].message
                assistant_payload = {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [],
                }
                tool_results: List[Dict[str, Any]] = []
                for tool_call in assistant_msg.tool_calls:
                    call_id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments or "{}"
                    assistant_payload["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": tool_call.type,
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    )
                    tool_output = await self._execute_tool(name, arguments)
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_output,
                        }
                    )

                messages.append(assistant_payload)
                messages.extend(tool_results)

                openai_start = perf_counter()
                response = await _call_openai(messages)
                openai_duration += perf_counter() - openai_start

            if loop_guard >= max_loops and response.choices[0].message.tool_calls:
                logger.warning("工具调用超过最大迭代次数，提前结束")
                # Attach final assistant message to avoid hanging
                messages.append(
                    {
                        "role": "assistant",
                        "content": "工具调用次数过多，已停止尝试。",
                    }
                )
                openai_start = perf_counter()
                response = await _call_openai(messages)
                openai_duration += perf_counter() - openai_start

        content = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

        self.chat_history.append(f"用户: {user_input}")
        self.chat_history.append(f"Emilia: {content}")

        if self.memory_enabled and self.memory_actor:
            metadata = dict(self.memory_default_metadata or {})
            try:
                self.memory_actor.store_turn.remote(
                    user_message=user_input,
                    assistant_message=content,
                    user_id=self.memory_user_id,
                    agent_id=self.memory_agent_id,
                    run_id=self.memory_run_id,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning("调度记忆写入失败: %s", exc)

        total_duration = perf_counter() - total_start
        logger.info(
            "Timings(total=%.3fs, openai=%.3fs, mem_search=%.3fs)",
            total_duration,
            openai_duration,
            mem_search_duration,
        )

        return content

    async def shutdown(self) -> None:
        if self.mcp_manager:
            try:
                await self.mcp_manager.shutdown()
            except Exception:
                logger.exception("关闭 MCP 管理器失败")
            finally:
                self.mcp_manager = None
                self.tool_specs = []
                self.tool_map = {}
                self.tools_initialized = False

        if self.memory_actor is not None:
            try:
                ray.kill(self.memory_actor)
            except Exception:
                logger.warning("释放记忆 actor 失败", exc_info=True)
            finally:
                self.memory_actor = None
                self.memory_enabled = False

        if self.ray_initialized_here and ray.is_initialized():
            try:
                ray.shutdown()
            except Exception:
                logger.warning("关闭 Ray 失败", exc_info=True)
            finally:
                self.ray_initialized_here = False

    def reset(self) -> None:
        self.chat_history.clear()

    async def _execute_tool(self, name: str, arguments: str) -> str:
        tool = self.tool_map.get(name)
        if tool is None:
            return json.dumps({"error": f"Tool '{name}' not available"}, ensure_ascii=False)

        try:
            parsed_args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parsed_args = {"_raw": arguments}

        try:
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(parsed_args)
            else:
                result = await asyncio.to_thread(tool.invoke, parsed_args)
        except Exception as exc:
            logger.exception("工具 %s 执行失败: %s", name, exc)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        if result is None:
            return json.dumps({"result": None}, ensure_ascii=False)
        return str(result)

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        for env_name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY"):
            value = os.getenv(env_name)
            if value:
                return value
        return None

    @staticmethod
    def _requires_api_key(provider: Optional[str], cfg: Dict[str, Any]) -> bool:
        provider_name = (provider or "").lower()
        if provider_name not in {"openai", "azure_openai", "deepseek"}:
            return False
        base_url = str(
            cfg.get("base_url")
            or cfg.get("api_base")
            or cfg.get("openai_api_base")
            or cfg.get("deepseek_base_url")
            or ""
        )
        if base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost"):
            return False
        return True


async def interactive_loop(agent: OpenAINativeAgent) -> None:
    print("Emilia 原生客户端模式 (输入 exit 退出)")
    while True:
        try:
            sys.stdout.write("你: ")
            sys.stdout.flush()
            user_message = sys.stdin.readline()
        except EOFError:
            break
        if not user_message:
            break
        user_message = user_message.strip()
        if user_message.lower() == "exit":
            break
        reply = await agent.generate(user_message)
        print(f"Emilia: {reply}")


def build_agent() -> OpenAINativeAgent:
    llm_conf = conf.llm
    return OpenAINativeAgent(
        model=llm_conf.model_name,
        temperature=llm_conf.temperature,
        prompt_template=llm_conf.prompt,
        base_url=getattr(llm_conf, "base_url", None),
    )


if __name__ == "__main__":
    asyncio.run(interactive_loop(build_agent()))
