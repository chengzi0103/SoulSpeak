# # run_emilia.py
# import asyncio
# import os
# from dotenv import load_dotenv
#
# # 这里假设你把之前修改后的 emilia_llm_tagged.py 改为模块形式引入
# from emilia_llm_tagged import generate_emilia_tagged
#
# async def main():
#
#     user_input = "你好,你是谁？请介绍一下自己。"
#     lines = await generate_emilia_tagged(user_input)
#     print("\n".join(lines))
#
# if __name__ == "__main__":
#     load_dotenv('.env')  # 加载环境变量（如果用 .env 文件）
#
#     asyncio.run(main())

import asyncio
import os
from dotenv import load_dotenv
from emilia_llm_tagged import generate_emilia_tagged

async def repl():
    """
    Interactive REPL: Users can continuously input questions,
    receive tagged LLM responses, until typing 'exit'.
    """
    print("Emilia 交互式问答 (输入 'exit' 退出)")
    while True:
        user_input = input("你: ")
        if user_input.strip().lower() == 'exit':
            print("Goodbye! Emilia 退出问答。")
            break
        # 调用 LLM 生成标签化回复
        try:
            lines = await generate_emilia_tagged(user_input)
            # 打印每行
            for line in lines:
                print(f"Emilia: {line}")
        except Exception as e:
            print(f"调用 Emilia 失败: {e}")

if __name__ == '__main__':
    # 加载环境变量
    load_dotenv()
    # 运行 REPL
    asyncio.run(repl())
