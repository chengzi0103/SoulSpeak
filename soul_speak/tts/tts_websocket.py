
import os
from starlette.websockets import WebSocketState
import json
import sys

from soul_speak.utils.hydra_config.init import conf

tts_conf = conf.tts
print(tts_conf)
for sys_path in tts_conf.sys_path: sys.path.append(sys_path)
import torch
import numpy as np
import contextlib
import uvicorn
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from vllm import ModelRegistry
if tts_conf.vllm_enable:
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# # 项目路径配置
# COSYSPEECH_PATH = '/home/chengzi/projects/github/CosyVoice'
# MATCHA_TTS_PATH = os.path.join(COSYSPEECH_PATH, 'third_party/Matcha-TTS')
# sys.path.append(COSYSPEECH_PATH)
# sys.path.append(MATCHA_TTS_PATH)
# sys.path.append(os.path.join(MATCHA_TTS_PATH, 'matcha'))

from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice2

# 全局参数
tts_model: CosyVoice2 = None
global_prompt = None
REAL_SR = tts_conf.REAL_SR
voice_selection = tts_conf.voice_select
audio_path = None
global_prompt_text = None
for voice in tts_conf.audio_data:
    if voice['name'] == voice_selection:
        print('-------  : ',voice)
        global_prompt_text = voice.prompt
        audio_path = voice.audio_file_path
        break
else:
    raise ValueError(f"未找到指定的音频数据：{voice_selection}")

model_path = conf.cosyspeech_path + '/' + tts_conf.model_name
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, global_prompt, REAL_SR
    print("[Startup] 正在初始化并加载模型...")
    try:
        # ✅ 改为启用 vLLM
        if tts_conf.vllm_enable:

            tts_model = CosyVoice2(
                model_dir=model_path,
                load_jit=True,
                load_trt=True,
                load_vllm=True,  # ✅ 使用 vLLM 推理
                fp16=True
            )
            global_prompt = load_wav(audio_path, 16000)
            print("="*50)
            print(f"[Startup] ✅ 模型加载成功！（vLLM 模式）")
            print(f"[Startup] 真实采样率 (Real Sample Rate): {REAL_SR} Hz")
            print("="*50)
        else:
            tts_model = CosyVoice2(
                model_dir=model_path,
                load_jit=False,
                load_trt=False,
                fp16=False
            )
            global_prompt = load_wav(audio_path, 16000)
            print("="*50)
            print(f"[Startup] ✅ 模型加载成功！（非 vLLM 模式）")
            print(f"[Startup] 真实采样率 (Real Sample Rate): {REAL_SR} Hz")
            print("="*50)
    except Exception as e :
        print("\n详细错误信息:",e)
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
    print(f"启动 Uvicorn 服务器，监听 0.0.0.0:{tts_conf.websocket.port}")
    uvicorn.run(app, host="0.0.0.0", port=tts_conf.websocket.port)