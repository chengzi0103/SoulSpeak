import os
import asyncio
import logging
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from soul_speak.utils.hydra_config.init import conf
from soul_speak.llm.mcp_manager import MCPManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv('.env')

# 全局变量
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
mcp_manager: MCPManager = None
agent_executor = None

async def initialize_llm_system():
    """初始化LLM系统，包括MCP管理器"""
    global mcp_manager, agent_executor
    
    llm_conf = conf.llm
    
    # 初始化LLM
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name=llm_conf.model_name,
        temperature=llm_conf.temperature
    )
    
# 初始化MCP管理器
    try:
        from soul_speak.llm.mcp_manager import MCPConfig
        
        mcp_configs = []
        # 最终修正嵌套访问
        services_data = dict(conf.mcp)
        if 'services' in services_data:
            services = services_data['services']
            for service_name, service_config in services.items():
                config = dict(service_config)
                if config.get('enabled', False):
                    mcp_configs.append(MCPConfig(
                        name=config.get('name', service_name),
                        url=config.get('url', ''),
                        description=config.get('description', ''),
                        type=config.get('type', 'sse'),
                        enabled=config.get('enabled', True),
                        timeout=config.get('timeout', 30)
                    ))
        
        if mcp_configs:
            mcp_manager = MCPManager(mcp_configs)
            await mcp_manager.initialize()
            tools = mcp_manager.get_tools()
            logger.info(f"Loaded {len(tools)} tools from {len(mcp_configs)} MCP services")
        else:
            tools = []
            logger.warning("No enabled MCP services found")
            
    except Exception as e:
        logger.error(f"Failed to initialize MCP: {e}")
        import traceback
        traceback.print_exc()
        tools = []
    
    # 初始化Agent - 使用支持工具的agent
    if tools:
        try:
            # 使用Structured Chat Agent支持多输入工具
            agent_executor = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
        except Exception as e:
            logger.warning(f"Failed to create structured agent: {e}, using simple LLM")
            agent_executor = llm
    else:
        # 如果没有工具，使用简单的LLM
        agent_executor = llm

# 异步交互函数
async def generate_emilia_tagged(user_input: str):
    """生成AI响应，确保系统已初始化"""
    global agent_executor
    
    if agent_executor is None:
        await initialize_llm_system()
    
    try:
        if hasattr(agent_executor, 'ainvoke'):
            # 新的AgentExecutor
            response = await agent_executor.ainvoke({"input": user_input})
            content = response["output"]
        else:
            # 简单的LLM
            response = await agent_executor.ainvoke(user_input)
            content = response.content
            
        lines = [l.strip() for l in str(content).splitlines() if l.strip()]
        return lines
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return ["抱歉，我在处理您的请求时遇到了问题。请稍后再试。"]

async def shutdown_llm_system():
    """关闭LLM系统，清理资源"""
    global mcp_manager
    if mcp_manager:
        await mcp_manager.shutdown()

async def main():
    print("Emilia 聊天 (输入 exit 退出)")
    while True:
        user_input = input("你: ")
        if user_input.strip().lower() == 'exit':
            break
        lines = await generate_emilia_tagged(user_input)
        for line in lines:
            print(f"Emilia: {line}")

if __name__ == '__main__':
    asyncio.run(main())
