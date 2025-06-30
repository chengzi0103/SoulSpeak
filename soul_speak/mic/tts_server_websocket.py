import os
import sys
import time
import torch
import numpy as np
import torchaudio
import contextlib
import uvicorn
import traceback  # 导入 traceback 模块用于打印详细错误
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# --- 1. 关键：项目路径配置 ---
# 这一步至关重要，它告诉 Python 在哪里可以找到 cosyvoice 库。
# 如果不配置，启动时会因 "ModuleNotFoundError" 而失败。
# 请根据您的项目结构修改这些路径。
COSYSPEECH_PATH = '/home/chengzi/projects/github/CosyVoice'  # CosyVoice 项目根目录
MATCHA_TTS_PATH = os.path.join(COSYSPEECH_PATH, 'third_party/Matcha-TTS')  # Matcha-TTS 路径

# 将路径添加到系统环境中
sys.path.append(COSYSPEECH_PATH)
sys.path.append(MATCHA_TTS_PATH)
sys.path.append(os.path.join(MATCHA_TTS_PATH, 'matcha'))

# 现在可以安全地导入 cosyvoice 模块了
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice2

# --- 2. 全局参数 ---
# 将模型和配置定义为全局变量
tts_model: CosyVoice2 = None
global_prompt = None
REAL_SR = 24000  # 默认值，将在模型加载后更新
global_prompt_text = (
    "爆炸豆家的智能电脑,虽然我只是一台电脑.但我可是超级智能的,"
    "我可以回答各种问题,还能和你们聊天,分享我的见闻.最重要的是,"
    "我还是爆炸豆的好伙伴"
)


# --- 3. 生命周期管理 (模型加载与释放) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在应用启动时加载模型，在应用关闭时释放资源。
    """
    global tts_model, global_prompt, REAL_SR
    print("[Startup] 正在初始化并加载模型...")
    try:
        # 初始化 TTS 模型
        tts_model = CosyVoice2(
            model_dir='pretrained_models/CosyVoice2-0.5B',
            load_jit=True, load_trt=False, load_vllm=False, fp16=True
        )
        # 加载 prompt 音频
        global_prompt = load_wav('susu.wav', 16000)
        print("=" * 50)
        print(f"[Startup] ✅ 模型加载成功！")
        print(f"[Startup] 真实采样率 (Real Sample Rate): {REAL_SR} Hz")
        print(f"[Startup] 服务器已准备就绪，可以接收连接。")
        print("=" * 50)
    except Exception:
        # 如果启动失败，打印完整的错误堆栈信息
        print("=" * 50)
        print("[Startup] ❌❌❌ 致命错误：模型加载失败！请检查以下信息：")
        print(f"1. 模型路径 'pretrained_models/CosyVoice2-0.5B' 是否正确？")
        print(f"2. Prompt 音频 'susu.wav' 是否存在？")
        print(f"3. 项目路径配置是否正确？当前配置为: {COSYSPEECH_PATH}")
        print("\n详细错误信息:")
        traceback.print_exc()
        print("=" * 50)
        sys.exit(1)  # 退出程序

    # yield 将控制权交还给 FastAPI 应用
    yield

    # --- 应用关闭时执行的代码 ---
    print("[Shutdown] 正在释放资源...")
    del tts_model
    del global_prompt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Shutdown] ✅ 资源已释放。")


# --- 4. 创建 FastAPI 应用实例 ---
# 使用新的 lifespan 参数来管理应用的生命周期
app = FastAPI(lifespan=lifespan)


# --- 5. WebSocket 端点 ---
@app.websocket("/ws/synthesize")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print(f"[WebSocket] 客户端 {ws.client.host}:{ws.client.port} 已连接。")
    try:
        while True:
            text = await ws.receive_text()
            if not text.strip():
                continue

            print(f"[WebSocket] 收到文本: '{text[:30]}...'")

            # 调用模型进行流式推理
            stream_gen = tts_model.inference_zero_shot(
                tts_text=text,
                prompt_text=global_prompt_text,
                prompt_speech_16k=global_prompt,
                stream=True
            )

            # 用于保存完整调试文件的列表
            full_audio_chunks = []

            # 直接处理模型返回的每个独立的音频块
            for res in stream_gen:
                chunk_tensor = res['tts_speech'].cpu().float()

                if chunk_tensor.dim() == 1:
                    chunk_tensor = chunk_tensor.unsqueeze(0)

                if chunk_tensor.shape[1] == 0:
                    continue

                # 保存当前块用于最后拼接
                full_audio_chunks.append(chunk_tensor)

                # 将当前块转换为 pcm_s16le 格式并发送给客户端
                pcm16 = (chunk_tensor.numpy().flatten() * 32767).astype(np.int16)
                await ws.send_bytes(pcm16.tobytes())

            # 拼接所有块，保存完整的 WAV 文件以供调试
            # if full_audio_chunks:
            #     full_wav = torch.cat(full_audio_chunks, dim=1)
            #     debug_path = f"debug_stream_{int(time.time())}.wav"
            #     torchaudio.save(debug_path, full_wav, REAL_SR)
            #     print(f"[Debug] ✅ 保存完整 stream WAV: {debug_path}")

            # 发送语音结束标记
            await ws.send_text("END_OF_SPEECH")

    except WebSocketDisconnect:
        print(f"[WebSocket] 客户端 {ws.client.host}:{ws.client.port} 已断开。")
    except Exception:
        print(f"[WebSocket ERROR] 处理客户端 {ws.client.host}:{ws.client.port} 时发生错误:")
        traceback.print_exc()
    finally:
        # 确保在任何情况下都尝试关闭连接
        if ws.client_state != "disconnected":
            await ws.close()
        print(f"[WebSocket] 连接 {ws.client.host}:{ws.client.port} 已关闭。")


# --- 6. 启动服务器 ---
if __name__ == "__main__":
    print("启动 Uvicorn 服务器，监听 0.0.0.0:8000")
    # 建议在命令行中运行 `uvicorn main:app --host 0.0.0.0 --port 8000`
    # 此处为了方便直接在脚本中运行
    uvicorn.run(app, host="0.0.0.0", port=8000)

