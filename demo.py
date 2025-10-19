import os

import openai
# 在代码中设置 no_proxy
os.environ['no_proxy'] = '10.100.1.115,127.0.0.1,localhost'
os.environ['NO_PROXY'] = '10.100.1.115,127.0.0.1,localhost'
client = openai.OpenAI(
    base_url="http://10.100.1.115:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="gpt-oss:latest",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1024
)

print("AI Response:", response.choices[0].message.content)
