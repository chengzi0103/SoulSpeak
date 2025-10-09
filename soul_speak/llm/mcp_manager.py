import asyncio
from typing import List, Dict, Any, Optional
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
import logging
from dataclasses import dataclass, field
from contextlib import AsyncExitStack
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP client libraries not available, skipping MCP integration")

logger = logging.getLogger(__name__)

@dataclass
class MCPConfig:
    name: str
    url: str
    description: str
    type: str = "sse"  # sse, stdio, http
    enabled: bool = True
    timeout: int = 30

class MCPConnection:
    """Maintain a persistent MCP connection so tools remain usable."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.client_cm = None
        self.session_cm = None
        self.session: Optional[ClientSession] = None

    async def connect(self) -> List[BaseTool]:
        if self.config.type != "sse":
            raise ValueError(f"Unsupported MCP type: {self.config.type}")

        self.client_cm = sse_client(self.config.url)
        read, write = await self.client_cm.__aenter__()
        self.session_cm = ClientSession(read, write)
        self.session = await self.session_cm.__aenter__()
        await self.session.initialize()
        tools = await load_mcp_tools(self.session)
        return tools

    async def close(self) -> None:
        if self.session_cm is not None:
            try:
                await self.session_cm.__aexit__(None, None, None)
            finally:
                self.session_cm = None
                self.session = None
        if self.client_cm is not None:
            try:
                await self.client_cm.__aexit__(None, None, None)
            finally:
                self.client_cm = None


class MCPManager:
    def __init__(self, mcp_configs: List[MCPConfig]):
        self.mcp_configs = mcp_configs
        self.tools: List[BaseTool] = []
        self.initialized = False
        self.connections: List[MCPConnection] = []
        
    async def initialize(self) -> None:
        """初始化所有配置的MCP客户端"""
        if self.initialized:
            return
            
        for config in self.mcp_configs:
            if not config.enabled:
                continue
                
            try:
                connection = MCPConnection(config)
                tools = await connection.connect()
                if tools:
                    self.connections.append(connection)
                    self.tools.extend(tools)
                    logger.info(f"MCP service '{config.name}' initialized with {len(tools)} tools")
                else:
                    await connection.close()
            except Exception as e:
                logger.error(f"Failed to initialize MCP service '{config.name}': {e}")
                
        self.initialized = True
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有可用的MCP工具"""
        return self.tools
    
    async def shutdown(self) -> None:
        """关闭所有MCP连接"""
        self.tools.clear()
        self.initialized = False
        while self.connections:
            connection = self.connections.pop()
            try:
                await connection.close()
            except Exception:
                logger.exception("关闭 MCP 连接失败")
