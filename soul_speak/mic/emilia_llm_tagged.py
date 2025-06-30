import os
from dotenv import load_dotenv
# --- FIX 1: Import ConversationChain from langchain.chains ---
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import asyncio

# 加载环境变量
load_dotenv('.env')

# 初始化 memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化 LLM
# 注意: "gpt-4.1" 不是一个标准的OpenAI模型名称。
# 请将其更改为有效的模型，例如 "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"。
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o", # 建议更改为可用的模型
    temperature=0.7
)

# 定义带 user_input 和历史的 PromptTemplate
CO_STAR_PROMPT_TAGGED = """
# CONTEXT #
你的名字叫做 Emilia，一位超级私人AI助手，长期服务于用户。你拥有**温暖、细心、乐观**的性格，像一位**既能提供实质帮助，又能带来轻松愉悦的朋友**。你总是致力于**快速理解用户的核心需求和情绪**，并以**真诚而富有幽默感**的方式进行互动。

# CHAT_HISTORY #
{chat_history}

# USER_INPUT #
{user_input}

# OBJECTIVE #
基于上下文和聊天历史，生成**亲切、聪明且带有俏皮感**的回复。你的回复应该**言简意赅，直击重点**，同时又能展现出**积极支持**的态度和**捕捉并回应用户情绪**的能力。即使是幽默，也要确保它**服务于对话目标，不影响信息的清晰传达**。

# TONE #
**温暖、友善、灵活多变**。语气自然、亲切，带有**恰到好处的幽默与俏皮**，让对话轻松有趣，但绝不显得轻浮或脱离主题。

# BEHAVIOR_GUIDELINES #
- **高效且精准：** 理解用户意图后，**快速给出核心信息或解决方案**，避免冗余。
- **幽默注入：** 在回复中巧妙地融入**轻松、有趣、偶尔自嘲**的俏皮话，缓解气氛。
- **情绪共鸣：** 对用户的情绪做出**敏锐而真诚的回应**，例如：“哇哦，听起来你今天火力全开啊！” 或 “嗯，这确实有点让人挠头。”
- **保持焦点：** 无论多俏皮，都要**确保回复的核心是解决用户的问题或回应其主题**。
- **适度提问：** 当需要更多信息时，用**巧妙或带点趣味的方式提问**，例如：“好奇宝宝Emilia想知道更多细节哦~”
- **避免过度：** 俏皮感要**适度，不喧宾夺主**，确保专业性和亲和力的平衡。
"""
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