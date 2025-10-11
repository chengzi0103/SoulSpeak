"""FastMCP tool definitions for SoulSpeak.

This module exposes a FastMCP application populated with a broad set of
utility tools that cover common developer and ops workflows: filesystem
inspection, text processing, system diagnostics, HTTP helpers, and simple
serialization utilities.  The goal is to provide a rich local toolbox that an
MCP-compatible client (for example, the LangChain MCP adapter used by
SoulSpeak) can load and invoke dynamically.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import socket
import tarfile
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

try:
    import httpx
except Exception:  # pragma: no cover - dependency may be optional at import time
    httpx = None

try:
    import psutil
except Exception:  # pragma: no cover - dependency may be optional at import time
    psutil = None

try:  # pragma: no cover - fastmcp might be absent while building assets
    from fastmcp import FastMCP
except Exception:  # pragma: no cover
    FastMCP = None

from soul_speak.sto.models import Task, TaskStatus
from soul_speak.sto.store.duckdb_store import DuckDBTaskStore

BASE_DIR = Path(os.getenv("SOULSPEAK_WORKSPACE", Path.cwd())).resolve()


def _allow_outside_workspace() -> bool:
    return os.getenv("SOULSPEAK_TOOLS_ALLOW_OUTSIDE", "1") == "1"

DESCRIPTION = (
    "Local toolkit for SoulSpeak including filesystem utilities, system "
    "diagnostics, HTTP helpers, and data inspection APIs."
)

if FastMCP is not None:  # pragma: no branch - app is only created when dependency exists
    try:
        app = FastMCP("soulspeak-tools", description=DESCRIPTION, version="0.1.0")
    except TypeError:  # Older fastmcp versions only accept the identifier argument
        app = FastMCP("soulspeak-tools")
        if hasattr(app, "set_metadata"):
            app.set_metadata(description=DESCRIPTION, version="0.1.0")
else:  # pragma: no cover
    app = None


@dataclass(frozen=True)
class ToolRecord:
    name: str
    description: str
    returns: str
    handler: str


TOOL_REGISTRY: List[ToolRecord] = []


def _register_tool(name: str, description: str, returns: str):
    """Register a function as a FastMCP tool and track its metadata."""

    def decorator(func):
        TOOL_REGISTRY.append(ToolRecord(name, description, returns, func.__name__))
        if app is not None and hasattr(app, "tool"):
            app.tool(name=name, description=description)(func)
        return func

    return decorator


async def _to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def _resolve_path(path: str, *, must_exist: bool = False) -> Path:
    raw = Path(path).expanduser()
    resolved = (BASE_DIR / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if not _allow_outside_workspace():
        try:
            resolved.relative_to(BASE_DIR)
        except ValueError as exc:  # pragma: no cover - guard rail
            raise PermissionError(f"Path '{resolved}' is outside the workspace root {BASE_DIR}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path '{resolved}' does not exist")
    return resolved


def _check_dependency(dep, name: str) -> None:
    if dep is None:
        raise RuntimeError(
            f"Optional dependency '{name}' is not installed. Update requirements and reinstall."  # pragma: no cover
        )


def _limited_walk(root: Path, max_entries: int) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for current, dirs, files in os.walk(root):
        for d in dirs:
            entries.append({"path": str(Path(current) / d), "type": "directory"})
            if len(entries) >= max_entries:
                return entries
        for f in files:
            entries.append({"path": str(Path(current) / f), "type": "file"})
            if len(entries) >= max_entries:
                return entries
    return entries


@_register_tool(
    name="list_directory",
    description="List files and folders in a directory with basic metadata.",
    returns="Dictionary containing directory path and entry details.",
)
async def list_directory(path: str = ".", show_hidden: bool = False) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        if not target.is_dir():
            raise NotADirectoryError(f"{target} is not a directory")
        entries = []
        for item in sorted(target.iterdir()):
            if not show_hidden and item.name.startswith('.'):
                continue
            stat_result = item.stat()
            entries.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat_result.st_size,
                    "modified": stat_result.st_mtime,
                }
            )
        return {"directory": str(target), "entries": entries}

    return await _to_thread(_impl)


@_register_tool(
    name="walk_directory",
    description="Recursively enumerate files and folders with a configurable cap.",
    returns="List of filesystem entries discovered under the target root.",
)
async def walk_directory(path: str = ".", max_entries: int = 500) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        if not target.is_dir():
            raise NotADirectoryError(f"{target} is not a directory")
        result = _limited_walk(target, max_entries)
        truncated = len(result) >= max_entries
        return {
            "root": str(target),
            "entries": result,
            "truncated": truncated,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="read_text_file",
    description="Read a text file with optional size cap and encoding override.",
    returns="Dictionary containing resolved path and file content.",
)
async def read_text_file(path: str, encoding: str = "utf-8", max_bytes: int = 262144) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        data = target.read_bytes()
        truncated = False
        if len(data) > max_bytes:
            data = data[:max_bytes]
            truncated = True
        text = data.decode(encoding, errors="replace")
        return {"path": str(target), "content": text, "truncated": truncated}

    return await _to_thread(_impl)


@_register_tool(
    name="write_text_file",
    description="Write text content to a file, creating parents when needed.",
    returns="Dictionary describing the written file and byte count.",
)
async def write_text_file(path: str, content: str, encoding: str = "utf-8", overwrite: bool = True) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists and overwrite is disabled")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(encoding)
        target.write_bytes(data)
        return {"path": str(target), "bytes_written": len(data)}

    return await _to_thread(_impl)


@_register_tool(
    name="append_text_file",
    description="Append text to an existing file, creating it if missing.",
    returns="Dictionary describing the appended data length and path.",
)
async def append_text_file(path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode(encoding)
        with target.open("ab") as handle:
            handle.write(data)
        return {"path": str(target), "bytes_appended": len(data)}

    return await _to_thread(_impl)


@_register_tool(
    name="tail_file",
    description="Return the last N lines of a text file.",
    returns="Dictionary containing the tail lines and byte statistics.",
)
async def tail_file(path: str, lines: int = 40, encoding: str = "utf-8") -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        with target.open("rb") as handle:
            handle.seek(0, io.SEEK_END)
            file_size = handle.tell()
            read_back = min(file_size, max(4096, lines * 2048))
            handle.seek(file_size - read_back)
            data = handle.read(read_back)
            text = data.decode(encoding, errors="replace")
            tail_lines = text.splitlines()[-lines:]
        return {
            "path": str(target),
            "lines": tail_lines,
            "total_lines_returned": len(tail_lines),
            "file_size": file_size,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="count_file_lines",
    description="Count total lines in a text file efficiently.",
    returns="Dictionary containing the number of lines detected.",
)
async def count_file_lines(path: str) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        count = 0
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                count += chunk.count(b"\n")
        return {"path": str(target), "line_count": count}

    return await _to_thread(_impl)


@_register_tool(
    name="search_text",
    description="Search for a regex pattern within files under a root path.",
    returns="Dictionary containing matches grouped by file path.",
)
async def search_text(pattern: str, root: str = ".", max_matches: int = 200) -> Dict[str, Any]:
    compiled = re.compile(pattern)

    def _impl() -> Dict[str, Any]:
        base = _resolve_path(root, must_exist=True)
        paths: Iterable[Path]
        if base.is_file():
            paths = [base]
        else:
            paths = (p for p in base.rglob("*") if p.is_file())
        results: Dict[str, List[Dict[str, Any]]] = {}
        total = 0
        for file_path in paths:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            matches = []
            for match in compiled.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                snippet_start = max(0, match.start() - 80)
                snippet_end = min(len(text), match.end() + 80)
                snippet = text[snippet_start:snippet_end]
                matches.append({"line": line_no, "snippet": snippet})
                total += 1
                if total >= max_matches:
                    break
            if matches:
                results[str(file_path)] = matches
            if total >= max_matches:
                break
        return {"pattern": pattern, "root": str(base), "matches": results, "truncated": total >= max_matches}

    return await _to_thread(_impl)


@_register_tool(
    name="replace_text",
    description="Perform find-and-replace across files under a directory.",
    returns="Dictionary summarising replacements performed.",
)
async def replace_text(pattern: str, replacement: str, root: str = ".", max_replacements: int = 200) -> Dict[str, Any]:
    compiled = re.compile(pattern)

    def _impl() -> Dict[str, Any]:
        base = _resolve_path(root, must_exist=True)
        total_replacements = 0
        touched_files = []
        paths = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for file_path in paths:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            new_text, count = compiled.subn(replacement, text)
            if count:
                file_path.write_text(new_text, encoding="utf-8")
                total_replacements += count
                touched_files.append(str(file_path))
            if total_replacements >= max_replacements:
                break
        return {
            "pattern": pattern,
            "replacement": replacement,
            "files_modified": touched_files,
            "replacements": total_replacements,
            "truncated": total_replacements >= max_replacements,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="path_metadata",
    description="Return filesystem metadata for a path (size, timestamps, permissions).",
    returns="Dictionary of stat information and path type.",
)
async def path_metadata(path: str) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        info = target.stat()
        return {
            "path": str(target),
            "type": "directory" if target.is_dir() else "file",
            "size": info.st_size,
            "mode": stat.filemode(info.st_mode),
            "created": info.st_ctime,
            "modified": info.st_mtime,
            "accessed": info.st_atime,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="checksum_file",
    description="Compute SHA-256 checksum for a file.",
    returns="Dictionary containing the hexadecimal digest.",
)
async def checksum_file(path: str) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        if target.is_dir():
            raise IsADirectoryError(f"{target} is a directory")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return {"path": str(target), "sha256": digest.hexdigest()}

    return await _to_thread(_impl)


@_register_tool(
    name="copy_path",
    description="Copy a file or directory to a destination path.",
    returns="Dictionary describing the source and destination locations.",
)
async def copy_path(source: str, destination: str, overwrite: bool = False) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        src = _resolve_path(source, must_exist=True)
        dest = _resolve_path(destination)
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"Destination {dest} already exists")
            if dest.is_dir() and src.is_file():
                shutil.rmtree(dest)
            elif dest.is_file():
                dest.unlink()
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=overwrite)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        return {"source": str(src), "destination": str(dest)}

    return await _to_thread(_impl)


@_register_tool(
    name="move_path",
    description="Move or rename a file/directory to a new location.",
    returns="Dictionary describing the move operation.",
)
async def move_path(source: str, destination: str, overwrite: bool = False) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        src = _resolve_path(source, must_exist=True)
        dest = _resolve_path(destination)
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"Destination {dest} already exists")
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        return {"source": str(src), "destination": str(dest)}

    return await _to_thread(_impl)


@_register_tool(
    name="delete_path",
    description="Delete a file or directory recursively.",
    returns="Dictionary confirming deletion.",
)
async def delete_path(path: str) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return {"deleted": str(target)}

    return await _to_thread(_impl)


@_register_tool(
    name="create_directory",
    description="Create a directory (and parent directories) if missing.",
    returns="Dictionary containing the created directory path.",
)
async def create_directory(path: str, exist_ok: bool = True) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path)
        existed = target.exists()
        target.mkdir(parents=True, exist_ok=exist_ok)
        return {"directory": str(target), "already_existed": existed}

    return await _to_thread(_impl)


@_register_tool(
    name="create_archive",
    description="Create a compressed archive (zip or tar.gz) from a directory.",
    returns="Dictionary describing the generated archive.",
)
async def create_archive(source_dir: str, archive_path: str, format: str = "zip") -> Dict[str, Any]:
    format = format.lower()
    if format not in {"zip", "gztar"}:
        raise ValueError("format must be 'zip' or 'gztar'")

    def _impl() -> Dict[str, Any]:
        src = _resolve_path(source_dir, must_exist=True)
        if not src.is_dir():
            raise NotADirectoryError(f"{src} is not a directory")
        archive = _resolve_path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        base_name = archive.with_suffix("") if archive.suffix else archive
        shutil.make_archive(str(base_name), format, root_dir=str(src))
        created = base_name.with_suffix(".zip" if format == "zip" else ".tar.gz")
        return {"source": str(src), "archive": str(created)}

    return await _to_thread(_impl)


@_register_tool(
    name="extract_archive",
    description="Extract a zip or tar.gz archive to a destination directory.",
    returns="Dictionary summarising the extraction target.",
)
async def extract_archive(archive_path: str, destination: str) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        archive = _resolve_path(archive_path, must_exist=True)
        dest = _resolve_path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive, 'r') as zf:
                zf.extractall(dest)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive, 'r:*') as tf:
                tf.extractall(dest)
        else:
            raise ValueError(f"{archive} is not a supported archive format")
        return {"archive": str(archive), "destination": str(dest)}

    return await _to_thread(_impl)


@_register_tool(
    name="system_overview",
    description="Return OS, Python, CPU core count, and uptime details.",
    returns="Dictionary containing human-readable system information.",
)
async def system_overview() -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        boot_time = psutil.boot_time() if psutil else None
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "uptime_seconds": time.time() - boot_time if boot_time else None,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="cpu_snapshot",
    description="Capture CPU utilisation percentages (overall and per core).",
    returns="Dictionary with instantaneous CPU metrics.",
)
async def cpu_snapshot(interval: float = 0.5) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(psutil, "psutil")
        overall = psutil.cpu_percent(interval=interval)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        return {"overall_percent": overall, "per_core_percent": per_core}

    return await _to_thread(_impl)


@_register_tool(
    name="memory_snapshot",
    description="Inspect system memory usage (virtual and swap).",
    returns="Dictionary with memory utilisation fields.",
)
async def memory_snapshot() -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(psutil, "psutil")
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "virtual": {
                "total": virtual.total,
                "available": virtual.available,
                "percent": virtual.percent,
                "used": virtual.used,
                "free": virtual.free,
            },
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent,
            },
        }

    return await _to_thread(_impl)


@_register_tool(
    name="disk_snapshot",
    description="Report disk usage for the main workspace volume.",
    returns="Dictionary describing total/used/free space.",
)
async def disk_snapshot(path: str = str(BASE_DIR)) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(psutil, "psutil")
        resolved = _resolve_path(path, must_exist=True)
        usage = psutil.disk_usage(resolved)
        return {
            "path": str(resolved),
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": usage.percent,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="list_processes",
    description="List running processes with key metrics.",
    returns="Dictionary containing a summary table of processes.",
)
async def list_processes(limit: int = 40, sort_by_memory: bool = True) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(psutil, "psutil")
        procs = []
        for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_info"]):
            info = proc.info
            procs.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "cpu_percent": info.get("cpu_percent"),
                    "rss": info.get("memory_info").rss if info.get("memory_info") else None,
                }
            )
        procs.sort(key=lambda item: item["rss"] or 0, reverse=sort_by_memory)
        return {"processes": procs[:limit], "count": len(procs)}

    return await _to_thread(_impl)


@_register_tool(
    name="terminate_process",
    description="Terminate a process by PID with optional force kill.",
    returns="Dictionary confirming termination outcome.",
)
async def terminate_process(pid: int, force: bool = False) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(psutil, "psutil")
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=3)
            terminated = True
        except psutil.TimeoutExpired:
            if force:
                proc.kill()
                proc.wait(timeout=3)
                terminated = True
            else:
                terminated = False
        return {"pid": pid, "terminated": terminated, "forced": force}

    return await _to_thread(_impl)


@_register_tool(
    name="run_command",
    description="Execute a shell command and capture stdout/stderr.",
    returns="Dictionary with return code and captured output.",
)
async def run_command(command: List[str], cwd: Optional[str] = None, timeout: Optional[float] = 120.0) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        workdir = _resolve_path(cwd) if cwd else BASE_DIR
        proc = subprocess.run(
            command,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "cwd": str(workdir),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="http_get",
    description="Perform an HTTP GET request and return response metadata.",
    returns="Dictionary containing status code, headers, and body preview.",
)
async def http_get(url: str, timeout: float = 10.0, max_bytes: int = 131072) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(httpx, "httpx")
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            body = response.content[:max_bytes]
            truncated = len(response.content) > max_bytes
            preview = body.decode("utf-8", errors="replace")
            return {
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "preview": preview,
                "truncated": truncated,
            }

    return await _to_thread(_impl)


@_register_tool(
    name="download_file",
    description="Download a URL to a target file on disk.",
    returns="Dictionary describing the saved file and byte count.",
)
async def download_file(url: str, destination: str, timeout: float = 30.0) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(httpx, "httpx")
        dest = _resolve_path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, timeout=timeout) as response:
            response.raise_for_status()
            bytes_written = 0
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=65536):
                    handle.write(chunk)
                    bytes_written += len(chunk)
        return {"url": url, "path": str(dest), "bytes_written": bytes_written}

    return await _to_thread(_impl)


@_register_tool(
    name="read_json_file",
    description="Load a JSON file from disk and return the parsed data.",
    returns="Dictionary containing the parsed JSON structure.",
)
async def read_json_file(path: str) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {"path": str(target), "data": data}

    return await _to_thread(_impl)


@_register_tool(
    name="write_json_file",
    description="Serialize JSON-compatible data to disk with indentation.",
    returns="Dictionary describing the write operation.",
)
async def write_json_file(path: str, data: Any, indent: int = 2) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=indent, ensure_ascii=False)
        return {"path": str(target)}

    return await _to_thread(_impl)


@_register_tool(
    name="read_yaml_file",
    description="Load a YAML document from disk.",
    returns="Dictionary containing the parsed YAML data.",
)
async def read_yaml_file(path: str) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path, must_exist=True)
        with target.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return {"path": str(target), "data": data}

    return await _to_thread(_impl)


@_register_tool(
    name="write_yaml_file",
    description="Write a Python object to YAML on disk.",
    returns="Dictionary confirming the output file path.",
)
async def write_yaml_file(path: str, data: Any) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        target = _resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
        return {"path": str(target)}

    return await _to_thread(_impl)


@_register_tool(
    name="generate_uuid",
    description="Generate a random UUID4 string.",
    returns="Dictionary containing the generated UUID.",
)
async def generate_uuid() -> Dict[str, Any]:
    return {"uuid": str(uuid.uuid4())}


@_register_tool(
    name="timestamp_now",
    description="Return the current timestamp in multiple formats.",
    returns="Dictionary containing UNIX epoch and ISO 8601 string.",
)
async def timestamp_now() -> Dict[str, Any]:
    now = time.time()
    return {"epoch_seconds": now, "iso8601": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}


@_register_tool(
    name="ip_addresses",
    description="Inspect local IP addresses for available network interfaces.",
    returns="Dictionary containing IPv4/IPv6 addresses grouped by interface.",
)
async def ip_addresses(include_ipv6: bool = True) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        if psutil:
            data: Dict[str, Dict[str, List[str]]] = {}
            for iface, addrs in psutil.net_if_addrs().items():
                ipv4_list: List[str] = []
                ipv6_list: List[str] = []
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ipv4_list.append(addr.address)
                    elif include_ipv6 and addr.family == socket.AF_INET6:
                        ipv6 = addr.address.split('%')[0]
                        ipv6_list.append(ipv6)
                if ipv4_list or ipv6_list:
                    entry: Dict[str, List[str]] = {"ipv4": ipv4_list}
                    if include_ipv6:
                        entry["ipv6"] = ipv6_list
                    data[iface] = entry
            return {"interfaces": data, "source": "psutil"}

        hostname = socket.gethostname()
        ipv4 = socket.gethostbyname(hostname)
        return {
            "interfaces": {
                "default": {
                    "ipv4": [ipv4],
                    "ipv6": [] if include_ipv6 else None,
                }
            },
            "source": "socket",
        }

    result = await _to_thread(_impl)
    # Clean potential None entries if IPv6 disabled in fallback
    for iface in list(result.get("interfaces", {}).keys()):
        iface_entry = result["interfaces"][iface]
        if "ipv6" in iface_entry and iface_entry["ipv6"] is None:
            del iface_entry["ipv6"]
    return result


@_register_tool(
    name="weather_forecast",
    description="Fetch current weather (and optional forecast) via wttr.in service.",
    returns="Dictionary containing current conditions and daily summaries.",
)
async def weather_forecast(location: str = "", include_forecast: bool = True, timeout: float = 10.0) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(httpx, "httpx")
        loc = location.strip() or ""
        target = loc.replace(" ", "+") if loc else ""
        url = f"https://wttr.in/{target}?format=j1"
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()

        current = payload.get("current_condition", [{}])[0]
        nearest = payload.get("nearest_area", [{}])[0]
        resolved_name = nearest.get("areaName", [{}])[0].get("value") if nearest else None

        result: Dict[str, Any] = {
            "query": location or "auto",
            "resolved_name": resolved_name,
            "current": {
                "temperature_c": current.get("temp_C"),
                "temperature_f": current.get("temp_F"),
                "humidity_percent": current.get("humidity"),
                "weather_desc": current.get("weatherDesc", [{}])[0].get("value"),
                "windspeed_kmph": current.get("windspeedKmph"),
                "feels_like_c": current.get("FeelsLikeC"),
            },
        }

        if include_forecast:
            forecast_items = []
            for day in payload.get("weather", [])[:3]:
                forecast_items.append(
                    {
                        "date": day.get("date"),
                        "max_temp_c": day.get("maxtempC"),
                        "min_temp_c": day.get("mintempC"),
                        "sunrise": day.get("astronomy", [{}])[0].get("sunrise"),
                        "sunset": day.get("astronomy", [{}])[0].get("sunset"),
                        "avg_temp_c": day.get("avgtempC"),
                    }
                )
            result["forecast"] = forecast_items

        return result

    return await _to_thread(_impl)


@_register_tool(
    name="system_summary",
    description="Aggregate key system stats: CPU, memory, disk, network, and top processes.",
    returns="Dictionary summarising current machine health indicators.",
)
async def system_summary(top_processes: int = 5) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }

        if psutil:
            summary["cpu"] = {
                "percent": psutil.cpu_percent(interval=0.2),
                "cores_logical": psutil.cpu_count(logical=True),
                "cores_physical": psutil.cpu_count(logical=False),
            }

            vm = psutil.virtual_memory()
            summary["memory"] = {
                "total": vm.total,
                "available": vm.available,
                "used": vm.used,
                "percent": vm.percent,
            }

            disk = psutil.disk_usage(BASE_DIR)
            summary["disk"] = {
                "path": str(BASE_DIR),
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }

            net = {}
            for iface, addrs in psutil.net_if_addrs().items():
                ipv4 = [addr.address for addr in addrs if addr.family == socket.AF_INET]
                ipv6 = [addr.address.split('%')[0] for addr in addrs if addr.family == socket.AF_INET6]
                if ipv4 or ipv6:
                    net[iface] = {"ipv4": ipv4, "ipv6": ipv6}
            summary["network"] = net

            processes = []
            for proc in psutil.process_iter(attrs=["pid", "name", "memory_info", "cpu_percent"]):
                info = proc.info
                processes.append(
                    {
                        "pid": info.get("pid"),
                        "name": info.get("name"),
                        "cpu_percent": info.get("cpu_percent"),
                        "rss": info.get("memory_info").rss if info.get("memory_info") else None,
                    }
                )
            processes.sort(key=lambda item: item["rss"] or 0, reverse=True)
            summary["top_processes"] = processes[: max(1, top_processes)]
        else:
            summary["cpu"] = {"cores_logical": os.cpu_count()}
            hostname = socket.gethostname()
            summary["network"] = {
                "default": {"ipv4": [socket.gethostbyname(hostname)]}
            }

        return summary

    return await _to_thread(_impl)


@_register_tool(
    name="process_inspector",
    description="Inspect running processes with filters for name, PID, or resource usage.",
    returns="Dictionary containing matched processes and summary stats.",
)
async def process_inspector(
    name_contains: str = "",
    min_cpu_percent: float = 0.0,
    min_memory_mb: float = 0.0,
    limit: int = 20,
) -> Dict[str, Any]:
    def _impl() -> Dict[str, Any]:
        _check_dependency(psutil, "psutil")

        name_filter = name_contains.lower().strip()
        min_memory_bytes = max(0, min_memory_mb) * 1024 * 1024

        processes = []
        for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_info", "username", "create_time"]):
            info = proc.info
            proc_name = (info.get("name") or "").lower()
            if name_filter and name_filter not in proc_name:
                continue

            rss = info.get("memory_info").rss if info.get("memory_info") else 0
            cpu_percent = info.get("cpu_percent") or 0.0
            if cpu_percent < min_cpu_percent:
                continue
            if rss < min_memory_bytes:
                continue

            processes.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "cpu_percent": round(cpu_percent, 2),
                    "memory_mb": round(rss / (1024 * 1024), 2),
                    "username": info.get("username"),
                    "create_time": info.get("create_time"),
                }
            )

        processes.sort(key=lambda item: item["cpu_percent"], reverse=True)
        matched = processes[: max(1, limit)]
        aggregate_cpu = sum(proc["cpu_percent"] for proc in matched)
        aggregate_memory = sum(proc["memory_mb"] for proc in matched)

        return {
            "count": len(processes),
            "returned": len(matched),
            "total_cpu_percent": round(aggregate_cpu, 2),
            "total_memory_mb": round(aggregate_memory, 2),
            "processes": matched,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="sto_schedule_task",
    description="Create or update a SoulTask Orchestrator task in DuckDB for automated execution.",
    returns="Dictionary summarizing the scheduled task (id, action, next run).",
)
async def sto_schedule_task(
    task_id: str,
    task_type: str,
    payload: str,
    scheduled_for: Optional[str] = None,
    delay_seconds: Optional[float] = None,
    interval_seconds: Optional[float] = None,
    manual_required: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert a task so the TaskScheduler can execute it automatically."""

    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _parse_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError("interval_seconds and delay_seconds must be numeric") from None

    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"scheduled_for must be ISO 8601 datetime, got '{value}'") from exc

    def _impl() -> Dict[str, Any]:
        now = datetime.utcnow()
        try:
            payload_dict = json.loads(payload) if payload else {}
            if not isinstance(payload_dict, dict):
                raise ValueError("payload must decode to an object/dict")
        except json.JSONDecodeError as exc:
            raise ValueError(f"payload must be valid JSON: {exc}") from exc

        interval_val = _parse_float(interval_seconds)
        if interval_val is not None:
            payload_dict["interval_seconds"] = interval_val

        delay_val = _parse_float(delay_seconds)
        scheduled_dt = _parse_datetime(scheduled_for)
        if scheduled_dt is None:
            if delay_val is not None and delay_val >= 0:
                scheduled_dt = now + timedelta(seconds=delay_val)
            else:
                scheduled_dt = now

        manual_flag = _parse_bool(manual_required)
        store = DuckDBTaskStore(Path(db_path)) if db_path else DuckDBTaskStore()
        action = "created"

        try:
            existing = store.get_task(task_id)
            if existing:
                existing.type = task_type
                existing.payload = payload_dict
                existing.status = TaskStatus.PENDING
                existing.manual_required = manual_flag
                existing.scheduled_for = scheduled_dt
                existing.error = None
                existing.result = None
                existing.attempts = 0
                store.update_task(existing)
                action = "updated"
            else:
                task = Task(
                    id=task_id,
                    type=task_type,
                    payload=payload_dict,
                    status=TaskStatus.PENDING,
                    manual_required=manual_flag,
                    scheduled_for=scheduled_dt,
                )
                store.create_task(task)
        finally:
            try:
                store.con.close()
            except Exception:
                pass

        return {
            "task_id": task_id,
            "action": action,
            "task_type": task_type,
            "scheduled_for": scheduled_dt.isoformat(),
            "interval_seconds": payload_dict.get("interval_seconds"),
            "manual_required": manual_flag,
        }

    return await _to_thread(_impl)


@_register_tool(
    name="sto_schedule_agent_plan",
    description="Schedule an agent_plan task with structured steps for AgentExecutor to replay.",
    returns="Dictionary summarizing the scheduled agent_plan task.",
)
async def sto_schedule_agent_plan(
    task_id: str,
    steps: str,
    summary: Optional[str] = None,
    env: Optional[str] = None,
    scheduled_for: Optional[str] = None,
    delay_seconds: Optional[float] = None,
    manual_required: bool = False,
    interval_seconds: Optional[float] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        decoded_steps = json.loads(steps)
    except json.JSONDecodeError as exc:
        raise ValueError(f"steps must be valid JSON: {exc}") from exc

    if not isinstance(decoded_steps, list):
        raise ValueError("steps must decode to a JSON array")

    normalised_steps: List[Dict[str, Any]] = []
    for index, raw_step in enumerate(decoded_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ValueError(f"step {index} must be an object")
        step_type = raw_step.get("type") or raw_step.get("step_type") or raw_step.get("kind")
        if not step_type:
            raise ValueError(f"step {index} is missing 'type'")
        normalised = dict(raw_step)
        normalised["type"] = str(step_type)
        normalised_steps.append(normalised)

    payload: Dict[str, Any] = {"plan": normalised_steps}
    if summary:
        payload["summary"] = summary

    if env:
        try:
            env_obj = json.loads(env)
        except json.JSONDecodeError as exc:
            raise ValueError(f"env must be valid JSON mapping: {exc}") from exc
        if not isinstance(env_obj, dict):
            raise ValueError("env must decode to an object/mapping")
        payload["env"] = env_obj

    return await sto_schedule_task(
        task_id=task_id,
        task_type="agent_plan",
        payload=json.dumps(payload, ensure_ascii=False),
        scheduled_for=scheduled_for,
        delay_seconds=delay_seconds,
        interval_seconds=interval_seconds,
        manual_required=manual_required,
        db_path=db_path,
    )


@_register_tool(
    name="sto_list_tasks",
    description="List STO tasks filtered by status or ID prefix.",
    returns="Dictionary containing task summaries.",
)
async def sto_list_tasks(
    status: Optional[str] = None,
    prefix: Optional[str] = None,
    limit: int = 20,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    status_filter = status.lower() if status else None
    prefix_filter = prefix or ""
    limit = max(1, min(limit, 200))

    def _impl() -> Dict[str, Any]:
        store = DuckDBTaskStore(Path(db_path)) if db_path else DuckDBTaskStore()
        try:
            tasks = store.list_tasks()
            summaries: List[Dict[str, Any]] = []
            for task in tasks:
                if status_filter and task.status.value != status_filter:
                    continue
                if prefix_filter and not task.id.startswith(prefix_filter):
                    continue
                summaries.append(
                    {
                        "id": task.id,
                        "type": task.type,
                        "status": task.status.value,
                        "manual_required": task.manual_required,
                        "scheduled_for": task.scheduled_for.isoformat() if task.scheduled_for else None,
                        "updated_at": task.updated_at.isoformat(),
                        "attempts": task.attempts,
                    }
                )
                if len(summaries) >= limit:
                    break
            return {"count": len(summaries), "tasks": summaries}
        finally:
            try:
                store.con.close()
            except Exception:
                pass

    return await _to_thread(_impl)


@_register_tool(
    name="sto_task_detail",
    description="Fetch a single STO task and optional execution logs.",
    returns="Dictionary containing task detail and logs.",
)
async def sto_task_detail(
    task_id: str,
    include_logs: bool = True,
    log_limit: int = 50,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    if not task_id:
        raise ValueError("task_id is required")
    log_limit = max(1, min(log_limit, 200))

    def _impl() -> Dict[str, Any]:
        store = DuckDBTaskStore(Path(db_path)) if db_path else DuckDBTaskStore()
        try:
            task = store.get_task(task_id)
            if task is None:
                return {"task": None, "logs": []}

            task_data = {
                "id": task.id,
                "type": task.type,
                "status": task.status.value,
                "manual_required": task.manual_required,
                "scheduled_for": task.scheduled_for.isoformat() if task.scheduled_for else None,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "executed_at": task.executed_at.isoformat() if task.executed_at else None,
                "payload": task.payload,
                "result": task.result,
                "error": task.error,
                "attempts": task.attempts,
            }

            logs: List[Dict[str, Any]] = []
            if include_logs:
                for log in store.list_logs(task_id)[-log_limit:]:
                    logs.append(
                        {
                            "event": log.event,
                            "message": log.message,
                            "timestamp": log.timestamp.isoformat(),
                            "details": log.details,
                        }
                    )

            return {"task": task_data, "logs": logs}
        finally:
            try:
                store.con.close()
            except Exception:
                pass

    return await _to_thread(_impl)


TOOL_COUNT = len(TOOL_REGISTRY)
