"""Run the SoulSpeak FastMCP tool server."""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

import os

from soul_speak.llm.tools import app

from soul_speak.utils.hydra_config.init import conf

logger = logging.getLogger(__name__)


async def _serve(host: str, port: int) -> None:
    if app is None:
        raise RuntimeError("fastmcp is not installed; cannot start the tool server.")
    if hasattr(app, "run_sse_async"):
        await app.run_sse_async(host=host, port=port)
    else:  # pragma: no cover - compatibility with older fastmcp
        await app.run_async(host=host, port=port)


def main(argv: Optional[list[str]] = None) -> None:
    defaults = getattr(conf.mcp, "tools", {})
    default_host = getattr(defaults, "host", "127.0.0.1")
    bind_host = getattr(defaults, "bind_host", default_host)
    default_port = int(getattr(defaults, "port", 8822))
    allow_outside = bool(getattr(defaults, "allow_outside", True))

    os.environ["SOULSPEAK_TOOLS_ALLOW_OUTSIDE"] = "1" if allow_outside else "0"

    parser = argparse.ArgumentParser(description="SoulSpeak FastMCP tool server")
    parser.add_argument("--host", default=bind_host, help=f"Host interface to bind (default: {bind_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"TCP port to expose (default: {default_port})")
    parser.add_argument("--public-host", default=default_host, help=f"Public host for clients (default: {default_host})")
    args = parser.parse_args(argv)

    os.environ.setdefault("SOULSPEAK_TOOLS_PUBLIC_HOST", args.public_host)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info("Starting SoulSpeak FastMCP tool server on %s:%s", args.host, args.port)
    try:
        asyncio.run(_serve(args.host, args.port))
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("FastMCP tool server stopped by user")


if __name__ == "__main__":  # pragma: no cover
    main()
