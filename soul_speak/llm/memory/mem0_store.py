"""Mem0-backed memory utilities for the SoulSpeak LLM runtime."""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from attrs import define, field

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from mem0 import Memory, MemoryClient


MessageDict = Mapping[str, Any]
MessagesInput = Union[str, MessageDict, Sequence[MessageDict]]


@define
class Mem0MemoryStore:
    """Thin wrapper around Mem0's Memory/MemoryClient APIs.

    Set ``mode`` to ``"cloud"`` to talk to the hosted Mem0 service via ``MemoryClient``
    (requires an API key) or to ``"local"`` to use the open-source runtime via
    ``Memory`` with an explicit configuration dictionary.
    """

    mode: str = field(default="cloud")
    api_key: Optional[str] = field(default=None)
    host: Optional[str] = field(default=None)
    org_id: Optional[str] = field(default=None)
    project_id: Optional[str] = field(default=None)
    default_user_id: Optional[str] = field(default=None)
    default_agent_id: Optional[str] = field(default=None)
    default_run_id: Optional[str] = field(default=None)
    default_metadata: Dict[str, Any] = field(factory=dict)
    default_filters: Dict[str, Any] = field(factory=dict)
    local_config: Optional[Dict[str, Any]] = field(default=None)
    _logger: logging.Logger = field(init=False, repr=False, factory=lambda: logging.getLogger(__name__))
    _client: Optional["MemoryClient"] = field(init=False, default=None, repr=False)
    _memory: Optional["Memory"] = field(init=False, default=None, repr=False)

    def __attrs_post_init__(self) -> None:
        mode = self.mode.lower().strip()
        if mode not in {"cloud", "local"}:
            raise ValueError("mode must be either 'cloud' or 'local'")

        if mode == "cloud":
            try:
                from mem0 import MemoryClient
            except ImportError as exc:  # pragma: no cover - runtime guard
                raise RuntimeError("mem0ai is required when mode='cloud'") from exc

            self._client = MemoryClient(
                api_key=self.api_key,
                host=self.host,
                org_id=self.org_id,
                project_id=self.project_id,
            )
        else:
            if not self.local_config:
                raise ValueError("local_config must be provided when mode='local'")

            try:
                from mem0 import Memory
            except ImportError as exc:  # pragma: no cover - runtime guard
                raise RuntimeError("mem0ai is required when mode='local'") from exc

            self._memory = Memory.from_config(self.local_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_messages(
        self,
        messages: MessagesInput,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        filters: Optional[Mapping[str, Any]] = None,
        infer: Optional[bool] = None,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Persist a batch of chat messages or raw text into Mem0."""

        payload = self._normalise_messages(messages)
        resolved = self._resolve_context(user_id, agent_id, run_id, metadata, filters)
        user_id, agent_id, run_id, merged_metadata, merged_filters = resolved
        self._ensure_scope(user_id, agent_id, run_id)

        if self._client is not None:
            return self._store_cloud(
                payload,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=merged_metadata,
                filters=merged_filters,
                infer=infer,
                memory_type=memory_type,
                prompt=prompt,
                extra=extra,
            )

        if self._memory is not None:
            return self._store_local(
                payload,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=merged_metadata,
                infer=infer,
                memory_type=memory_type,
                prompt=prompt,
                extra=extra,
            )

        raise RuntimeError("Mem0MemoryStore is not initialised correctly")

    def store_turn(
        self,
        user_message: str,
        assistant_message: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Helper to store a dialog turn consisting of user + assistant text."""

        payload: List[Dict[str, str]] = [{"role": "user", "content": user_message}]
        if assistant_message:
            payload.append({"role": "assistant", "content": assistant_message})
        return self.store_messages(payload, **kwargs)

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Mapping[str, Any]] = None,
        threshold: Optional[float] = None,
        version: str = "v1",
        **extra: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Retrieve memories relevant to a query."""

        resolved = self._resolve_context(user_id, agent_id, run_id, metadata=None, filters=filters)
        user_id, agent_id, run_id, _, merged_filters = resolved
        self._ensure_scope(user_id, agent_id, run_id)

        if self._client is not None:
            return self._search_cloud(
                query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                filters=merged_filters,
                threshold=threshold,
                version=version,
                extra=extra,
            )

        if self._memory is not None:
            return self._search_local(
                query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                filters=merged_filters,
                threshold=threshold,
                extra=extra,
            )

        raise RuntimeError("Mem0MemoryStore is not initialised correctly")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _store_cloud(
        self,
        payload: List[Dict[str, str]],
        *,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
        infer: Optional[bool],
        memory_type: Optional[str],
        prompt: Optional[str],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_args: Dict[str, Any] = {}
        if user_id:
            request_args["user_id"] = user_id
        if agent_id:
            request_args["agent_id"] = agent_id
        if run_id:
            request_args["run_id"] = run_id
        if metadata:
            request_args["metadata"] = metadata
        if filters:
            request_args["filters"] = filters
        if infer is not None:
            request_args["infer"] = infer
        if memory_type is not None:
            request_args["memory_type"] = memory_type
        if prompt is not None:
            request_args["prompt"] = prompt
        request_args.update(extra)
        return self._client.add(payload, **request_args)  # type: ignore[union-attr]

    def _store_local(
        self,
        payload: List[Dict[str, str]],
        *,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        infer: Optional[bool],
        memory_type: Optional[str],
        prompt: Optional[str],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_args: Dict[str, Any] = {
            k: v
            for k, v in {
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "metadata": metadata,
                "infer": infer,
                "memory_type": memory_type,
                "prompt": prompt,
                **extra,
            }.items()
            if v is not None
        }
        return self._memory.add(payload, **request_args)  # type: ignore[union-attr]

    def _search_cloud(
        self,
        query: str,
        *,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        limit: int,
        filters: Optional[Dict[str, Any]],
        threshold: Optional[float],
        version: str,
        extra: Dict[str, Any],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        request_args: Dict[str, Any] = {
            k: v
            for k, v in {
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "filters": filters,
                "threshold": threshold,
                "top_k": limit,
                **extra,
            }.items()
            if v is not None
        }
        return self._client.search(query, version=version, **request_args)  # type: ignore[union-attr]

    def _search_local(
        self,
        query: str,
        *,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        limit: int,
        filters: Optional[Dict[str, Any]],
        threshold: Optional[float],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_args: Dict[str, Any] = {
            k: v
            for k, v in {
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "limit": limit,
                "filters": filters,
                "threshold": threshold,
                **extra,
            }.items()
            if v is not None
        }
        return self._memory.search(query, **request_args)  # type: ignore[union-attr]

    def _resolve_context(
        self,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        metadata: Optional[Mapping[str, Any]],
        filters: Optional[Mapping[str, Any]],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        resolved_user = user_id or self.default_user_id
        resolved_agent = agent_id or self.default_agent_id
        resolved_run = run_id or self.default_run_id
        merged_metadata = self._merge_dicts(self.default_metadata, metadata)
        merged_filters = self._merge_dicts(self.default_filters, filters)
        return resolved_user, resolved_agent, resolved_run, merged_metadata, merged_filters

    @staticmethod
    def _merge_dicts(
        base: Mapping[str, Any],
        overrides: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not base and not overrides:
            return None
        merged: Dict[str, Any] = dict(base) if base else {}
        if overrides:
            merged.update(overrides)
        return merged or None

    @staticmethod
    def _normalise_messages(messages: MessagesInput) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        if isinstance(messages, Mapping):
            return [Mem0MemoryStore._coerce_message(messages)]

        if isinstance(messages, Sequence):
            return [Mem0MemoryStore._coerce_message(item) for item in messages]

        raise TypeError("messages must be a str, mapping, or sequence of mappings")

    @staticmethod
    def _coerce_message(message: MessageDict) -> Dict[str, str]:
        if not isinstance(message, Mapping):
            raise TypeError("Each message must be a mapping with 'role' and 'content'")
        try:
            role = str(message["role"])
            content = str(message["content"])
        except KeyError as exc:
            raise KeyError("message missing required key: 'role' or 'content'") from exc
        return {"role": role, "content": content}

    @staticmethod
    def _ensure_scope(
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
    ) -> None:
        if not any((user_id, agent_id, run_id)):
            raise ValueError("Provide at least one of user_id, agent_id, or run_id for scoping memories")


__all__ = ["Mem0MemoryStore"]

if __name__ == "__main__":
    import json
    import os
    from copy import deepcopy
    from pathlib import Path

    import yaml
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)

    package_root = Path(__file__).resolve().parents[2]
    repo_root = package_root.parent
    load_dotenv()
    load_dotenv(repo_root / '.env')
    load_dotenv(package_root / '.env')

    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o777)
        except (PermissionError, NotImplementedError):
            pass

    def _set_perms(path: Path) -> None:
        try:
            path.chmod(0o777)
        except (PermissionError, NotImplementedError, FileNotFoundError):
            pass

    def _resolve_env_api_key() -> Optional[str]:
        for env_name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY"):
            value = os.getenv(env_name)
            if value:
                return value
        return None

    def _requires_api_key(provider: Optional[str], cfg: Mapping[str, Any]) -> bool:
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

    package_root = Path(__file__).resolve().parents[2]
    conf_dir = package_root / "conf"
    gpt_path = conf_dir / "llm" / "gpt.yaml"

    if not gpt_path.exists():
        raise SystemExit(f"Missing configuration file: {gpt_path}")

    config_data = yaml.safe_load(gpt_path.read_text()) or {}
    if not isinstance(config_data, dict):
        raise SystemExit(f"Unexpected configuration format in {gpt_path}")

    memory_section = config_data.get("memory")
    if not isinstance(memory_section, dict):
        raise SystemExit("Add a 'memory' section to conf/llm/gpt.yaml before running this demo.")

    enabled = bool(memory_section.get("enable", True))
    if not enabled:
        print("Memory demo disabled via conf/llm/gpt.yaml (memory.enable=false)")
        raise SystemExit(0)

    mode = str(memory_section.get("mode", "local")).lower()
    default_user_id = memory_section.get("default_user_id")
    default_agent_id = memory_section.get("default_agent_id")
    default_run_id = memory_section.get("default_run_id")
    default_metadata = memory_section.get("default_metadata") or {}
    if not isinstance(default_metadata, Mapping):
        raise SystemExit("memory.default_metadata must be a mapping if provided.")
    default_metadata = dict(default_metadata)

    demo_root = package_root / "llm" / "memory" / "mem0_demo_data"
    _ensure_dir(conf_dir)
    _ensure_dir(demo_root)

    store_kwargs = {
        "mode": mode,
        "default_user_id": default_user_id,
        "default_agent_id": default_agent_id,
        "default_run_id": default_run_id,
        "default_metadata": default_metadata,
    }
    for key in ("api_key", "host", "org_id", "project_id"):
        if key in memory_section and memory_section[key] is not None:
            store_kwargs[key] = memory_section[key]

    vector_store_path = None
    history_db_path = None

    if mode == "local":
        local_config = deepcopy(memory_section.get("local_config") or {})
        if not local_config:
            raise SystemExit("Provide memory.local_config in conf/llm/gpt.yaml for local mode.")

        llm_section = local_config.setdefault("llm", {})
        embed_section = local_config.setdefault("embedder", {})
        llm_cfg = llm_section.setdefault("config", {})
        embed_cfg = embed_section.setdefault("config", {})
        llm_provider = llm_section.get("provider")
        embed_provider = embed_section.get("provider")

        provider_cfgs = (
            ("llm", llm_provider, llm_cfg),
            ("embedder", embed_provider, embed_cfg),
        )

        api_key = _resolve_env_api_key()
        if api_key:
            for _, provider_name, cfg in provider_cfgs:
                if not _requires_api_key(provider_name, cfg):
                    continue
                current = str(cfg.get("api_key") or "").strip()
                if not current or current.lower().startswith('your-'):
                    cfg["api_key"] = api_key
        else:
            missing = [
                label
                for label, provider_name, cfg in provider_cfgs
                if _requires_api_key(provider_name, cfg)
                and (
                    not cfg.get("api_key")
                    or str(cfg.get("api_key")).strip().lower().startswith('your-')
                )
            ]
            if missing:
                raise SystemExit(
                    "Set an API key via OPENAI_API_KEY/DEEPSEEK_API_KEY/LLM_API_KEY or "
                    "specify api_key for llm/embedder in memory.local_config."
                )

        vector_cfg = local_config.setdefault("vector_store", {}).setdefault("config", {})
        raw_vector_path = vector_cfg.get("path") or "../llm/memory/mem0_demo_data/chroma"
        vector_store_path = Path(raw_vector_path).expanduser()
        if not vector_store_path.is_absolute():
            vector_store_path = (gpt_path.parent / vector_store_path).resolve()
        vector_cfg["path"] = str(vector_store_path)
        _ensure_dir(vector_store_path)

        raw_history_path = local_config.get("history_db_path") or "../llm/memory/mem0_demo_data/history.db"
        history_db_path = Path(raw_history_path).expanduser()
        if not history_db_path.is_absolute():
            history_db_path = (gpt_path.parent / history_db_path).resolve()
        _ensure_dir(history_db_path.parent)
        local_config["history_db_path"] = str(history_db_path)

        store_kwargs["local_config"] = local_config
    elif mode != "cloud":
        raise SystemExit("Unsupported memory mode. Use 'local' or 'cloud'.")

    store = Mem0MemoryStore(**store_kwargs)

    try:
        batch_result = store.store_messages(
            [
                {"role": "user", "content": "I drink oat milk lattes every morning."},
                {"role": "assistant", "content": "Noted, you prefer oat milk lattes."},
            ],
            metadata={"topic": "preferences"},
        )
        print("store_messages ->", json.dumps(batch_result, ensure_ascii=False, indent=2, default=str))

        turn_result = store.store_turn(
            user_message="Remind me to buy coffee beans this weekend.",
            assistant_message="Sure, I will remember to remind you about coffee beans.",
            metadata={"topic": "reminders"},
        )
        print("store_turn ->", json.dumps(turn_result, ensure_ascii=False, indent=2, default=str))

        search_result = store.search(
            "coffee preferences",
            limit=3,
            filters={"topic": "preferences"},
        )
        print("search ->", json.dumps(search_result, ensure_ascii=False, indent=2, default=str))
    finally:
        paths_to_touch = [conf_dir]
        if vector_store_path is not None:
            paths_to_touch.append(vector_store_path)
        if history_db_path is not None:
            paths_to_touch.append(history_db_path)
            paths_to_touch.append(history_db_path.parent)
        paths_to_touch.append(demo_root)
        for path_obj in paths_to_touch:
            if path_obj.exists():
                _set_perms(path_obj)
