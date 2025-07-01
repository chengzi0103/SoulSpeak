import os
from dotenv import load_dotenv
# --- FIX 1: Import ConversationChain from langchain.chains ---
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import asyncio

from soul_speak.utils.hydra_config.init import conf

# 加载环境变量
load_dotenv('.env')

# 初始化 memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm_conf = conf.llm
# 初始化 LLM
# 注意: "gpt-4.1" 不是一个标准的OpenAI模型名称。
# 请将其更改为有效的模型，例如 "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"。
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name=llm_conf.model_name, # 建议更改为可用的模型
    temperature=llm_conf.temperature
)

# 定义带 user_input 和历史的 PromptTemplate
CO_STAR_PROMPT_TAGGED = llm_conf.prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template=CO_STAR_PROMPT_TAGGED
)

# 构建 ConversationChain
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    # --- FIX 2: Specify the input_key to match your prompt's user input variable ---
    input_key="user_input",
    verbose=False
)

# 异步生成函数
async def generate_emilia_tagged(user_input: str):
    # 使用 predict 传递 user_input
    response = await chain.apredict(user_input=user_input)
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    return lines

# 交互示例
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