#
# import os
# from starlette.websockets import WebSocketState
# import json
# import sys
#
# sys.path.append('third_party/Matcha-TTS')
# import time
# import torch
# import numpy as np
# import torchaudio
# import contextlib
# import uvicorn
# import traceback
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from vllm import ModelRegistry
# from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
# ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
# # 项目路径配置
# COSYSPEECH_PATH = '/home/chengzi/projects/github/CosyVoice'
# MATCHA_TTS_PATH = os.path.join(COSYSPEECH_PATH, 'third_party/Matcha-TTS')
# sys.path.append(COSYSPEECH_PATH)
# sys.path.append(MATCHA_TTS_PATH)
# sys.path.append(os.path.join(MATCHA_TTS_PATH, 'matcha'))
#
# from cosyvoice.utils.file_utils import load_wav
# from cosyvoice.cli.cosyvoice import CosyVoice2
#
# # 全局参数
# tts_model: CosyVoice2 = None
# global_prompt = None
# REAL_SR = 24000
# global_prompt_text = (
#     "爆炸豆家的智能电脑,虽然我只是一台电脑.但我可是超级智能的,"
#     "我可以回答各种问题,还能和你们聊天,分享我的见闻.最重要的是,"
#     "我还是爆炸豆的好伙伴"
# )
#
# # 生命周期管理
# @contextlib.asynccontextmanager
# async def lifespan(app: FastAPI):
#     global tts_model, global_prompt, REAL_SR
#     print("[Startup] 正在初始化并加载模型...")
#     try:
#         # ✅ 改为启用 vLLM
#         tts_model = CosyVoice2(
#             model_dir='pretrained_models/CosyVoice2-0.5B',
#             load_jit=True,
#             load_trt=True,
#             load_vllm=True,  # ✅ 使用 vLLM 推理
#             fp16=True
#         )
#         global_prompt = load_wav('susu.wav', 16000)
#         print("="*50)
#         print(f"[Startup] ✅ 模型加载成功！（vLLM 模式）")
#         print(f"[Startup] 真实采样率 (Real Sample Rate): {REAL_SR} Hz")
#         print("="*50)
#     except Exception:
#         print("="*50)
#         print("[Startup] ❌❌❌ 致命错误：模型加载失败！请检查以下信息：")
#         print(f"1. 模型路径 'pretrained_models/CosyVoice2-0.5B' 是否正确？")
#         print(f"2. Prompt 音频 'susu.wav' 是否存在？")
#         print(f"3. 项目路径配置是否正确？当前配置为: {COSYSPEECH_PATH}")
#         print("\n详细错误信息:")
#         traceback.print_exc()
#         print("="*50)
#         sys.exit(1)
#
#     yield
#
#     print("[Shutdown] 正在释放资源...")
#     del tts_model
#     del global_prompt
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     print("[Shutdown] ✅ 资源已释放。")
#
#
# # FastAPI 实例
# app = FastAPI(lifespan=lifespan)
#
#
# @app.websocket("/ws/synthesize")
# async def websocket_endpoint(ws: WebSocket):
#     await ws.accept()
#     print(f"[WebSocket] 客户端 {ws.client.host}:{ws.client.port} 已连接。")
#     try:
#         while True:
#             raw = await ws.receive_text()
#             try:
#                 data = json.loads(raw)
#             except json.JSONDecodeError:
#                 continue  # 忽略非法格式
#
#             event = data.get("event")
#             if event == "text":
#                 text = data.get("text", "").strip()
#                 if not text:
#                     continue
#
#                 print(f"[WebSocket] 📝 收到合成请求文本：'{text[:30]}…'")
#                 stream_gen = tts_model.inference_zero_shot(
#                     tts_text=text,
#                     prompt_text=global_prompt_text,
#                     prompt_speech_16k=global_prompt,
#                     stream=True
#                 )
#
#                 for res in stream_gen:
#                     chunk = res['tts_speech'].cpu().float()
#                     if chunk.dim() == 1:
#                         chunk = chunk.unsqueeze(0)
#                     if chunk.shape[1] == 0:
#                         continue
#                     pcm16 = (chunk.numpy().flatten() * 32767).astype(np.int16)
#                     await ws.send_bytes(pcm16.tobytes())
#
#             elif event == "end_of_speech":
#                 print("[WebSocket] ⚡ 收到结束命令，发送 END_OF_SPEECH 标记")
#                 await ws.send_text("END_OF_SPEECH")
#
#             else:
#                 continue
#
#     except WebSocketDisconnect:
#         print(f"[WebSocket] 客户端 {ws.client.host}:{ws.client.port} 已断开。")
#     except Exception:
#         print(f"[WebSocket ERROR] 处理客户端 {ws.client.host}:{ws.client.port} 时发生错误:")
#         traceback.print_exc()
#     finally:
#         if ws.client_state != WebSocketState.DISCONNECTED:
#             await ws.close()
#         print(f"[WebSocket] 连接 {ws.client.host}:{ws.client.port} 已关闭。")
#
#
# # 启动服务
# if __name__ == "__main__":
#     print("启动 Uvicorn 服务器，监听 0.0.0.0:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
from starlette.websockets import WebSocketState
import json
import sys

sys.path.append('third_party/Matcha-TTS')
import time
import torch
import numpy as np
import torchaudio
import contextlib
import uvicorn
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
# 项目路径配置
COSYSPEECH_PATH = '/home/chengzi/projects/github/CosyVoice'
MATCHA_TTS_PATH = os.path.join(COSYSPEECH_PATH, 'third_party/Matcha-TTS')
sys.path.append(COSYSPEECH_PATH)
sys.path.append(MATCHA_TTS_PATH)
sys.path.append(os.path.join(MATCHA_TTS_PATH, 'matcha'))

from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice2

# 全局参数
tts_model: CosyVoice2 = None
global_prompt = None
REAL_SR = 24000
global_prompt_text = (
    "爆炸豆家的智能电脑,虽然我只是一台电脑.但我可是超级智能的,"
    "我可以回答各种问题,还能和你们聊天,分享我的见闻.最重要的是,"
    "我还是爆炸豆的好伙伴"
)

# 生命周期管理
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, global_prompt, REAL_SR
    print("[Startup] 正在初始化并加载模型...")
    try:
        # ✅ 改为启用 vLLM
        tts_model = CosyVoice2(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            load_jit=True,
            load_trt=True,
            load_vllm=True,  # ✅ 使用 vLLM 推理
            fp16=True
        )
        global_prompt = load_wav('susu.wav', 16000)
        print("="*50)
        print(f"[Startup] ✅ 模型加载成功！（vLLM 模式）")
        print(f"[Startup] 真实采样率 (Real Sample Rate): {REAL_SR} Hz")
        print("="*50)
    except Exception:
        print("="*50)
        print("[Startup] ❌❌❌ 致命错误：模型加载失败！请检查以下信息：")
        print(f"1. 模型路径 'pretrained_models/CosyVoice2-0.5B' 是否正确？")
        print(f"2. Prompt 音频 'susu.wav' 是否存在？")
        print(f"3. 项目路径配置是否正确？当前配置为: {COSYSPEECH_PATH}")
        print("\n详细错误信息:")
        traceback.print_exc()
        print("="*50)
        sys.exit(1)

    yield

    print("[Shutdown] 正在释放资源...")
    del tts_model
    del global_prompt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] ✅ 资源已释放。")


# FastAPI 实例
app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/synthesize")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print(f"[WebSocket] 客户端 {ws.client.host}:{ws.client.port} 已连接。")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue  # 忽略非法格式

            event = data.get("event")
            if event == "text":
                text = data.get("text", "").strip()
                if not text:
                    continue

                print(f"[WebSocket] 📝 收到合成请求文本：'{text[:30]}…'")
                stream_gen = tts_model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=global_prompt_text,
                    prompt_speech_16k=global_prompt,
                    stream=True
                )

                for res in stream_gen:
                    chunk = res['tts_speech'].cpu().float()
                    if chunk.dim() == 1:
                        chunk = chunk.unsqueeze(0)
                    if chunk.shape[1] == 0:
                        continue
                    pcm16 = (chunk.numpy().flatten() * 32767).astype(np.int16)
                    await ws.send_bytes(pcm16.tobytes())

            elif event == "end_of_speech":
                print("[WebSocket] ⚡ 收到结束命令，发送 END_OF_SPEECH 标记")
                await ws.send_text("END_OF_SPEECH")

            else:
                continue

    except WebSocketDisconnect:
        print(f"[WebSocket] 客户端 {ws.client.host}:{ws.client.port} 已断开。")
    except Exception:
        print(f"[WebSocket ERROR] 处理客户端 {ws.client.host}:{ws.client.port} 时发生错误:")
        traceback.print_exc()
    finally:
        if ws.client_state != WebSocketState.DISCONNECTED:
            await ws.close()
        print(f"[WebSocket] 连接 {ws.client.host}:{ws.client.port} 已关闭。")


# 启动服务
if __name__ == "__main__":
    print("启动 Uvicorn 服务器，监听 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
