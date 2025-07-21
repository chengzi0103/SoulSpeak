import asyncio
from typing import List, Dict, Any, Optional
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
import logging
from dataclasses import dataclass
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

class MCPManager:
    def __init__(self, mcp_configs: List[MCPConfig]):
        self.mcp_configs = mcp_configs
        self.tools: List[BaseTool] = []
        self.initialized = False
        
    async def initialize(self) -> None:
        """初始化所有配置的MCP客户端"""
        if self.initialized:
            return
            
        for config in self.mcp_configs:
            if not config.enabled:
                continue
                
            try:
                tools = await self._create_mcp_tools(config)
                self.tools.extend(tools)
                logger.info(f"MCP service '{config.name}' initialized with {len(tools)} tools")
            except Exception as e:
                logger.error(f"Failed to initialize MCP service '{config.name}': {e}")
                
        self.initialized = True
        
    async def _create_mcp_tools(self, config: MCPConfig) -> List[BaseTool]:
        """根据配置创建MCP工具列表"""
        if config.type == "sse":
            async with sse_client(config.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    return tools
        else:
            raise ValueError(f"Unsupported MCP type: {config.type}")
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有可用的MCP工具"""
        return self.tools
    
    async def shutdown(self) -> None:
        """关闭所有MCP连接"""
        self.tools.clear()
        self.initialized = False