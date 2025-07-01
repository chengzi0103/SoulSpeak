
import asyncio
import json
import websockets
import sounddevice as sd
import numpy as np
from scipy import signal
from dotenv import load_dotenv
from soul_speak.llm.llm_tagged import generate_emilia_tagged
from soul_speak.utils.hydra_config.init import conf

llm_conf = conf.llm
WS_URL = f"ws://{llm_conf.websocket.host}:{llm_conf.websocket.port}/ws/synthesize"
SERVER_SR = 24000    # 服务端采样率
PING_INTERVAL = 60
PING_TIMEOUT = 30

def setup_audio_stream():
    dev = sd.query_devices(kind="output")
    sr = int(dev["default_samplerate"])
    ch = 2 if dev["max_output_channels"] >= 2 else 1
    print(f"[Client] Audio device SR={sr}Hz, channels={ch}")
    return sr, ch

async def tts_pipeline(user_input: str):
    sentences = await generate_emilia_tagged(user_input)
    print("[LLM] Sentences:", sentences)

    device_sr, channels = setup_audio_stream()

    # 1) 打开唯一一次的 OutputStream
    stream = sd.OutputStream(
        samplerate=device_sr,
        channels=channels,
        dtype="float32",
        blocksize=0  # use default
    )
    stream.start()

    try:
        # 2) 建立一次 WebSocket 连接
        async with websockets.connect(
            WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT
        ) as ws:
            await ws.ping()

            # 对每一句，连续写入到同一个 stream
            for idx, sent in enumerate(sentences, 1):
                print(f"[Client] Synthesizing {idx}/{len(sentences)}: {sent[:30]}…")
                # 发送合成请求
                await ws.send(json.dumps({"event": "text", "text": sent}))
                await ws.send(json.dumps({"event": "end_of_speech"}))

                # 读取并写入音频数据，直到本句结束
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, (bytes, bytearray)):
                        pcm16 = np.frombuffer(msg, dtype=np.int16)
                        pcm = pcm16.astype(np.float32) / 32767.0
                        # 重采样
                        if SERVER_SR != device_sr:
                            pcm = signal.resample(pcm, int(len(pcm)*device_sr/SERVER_SR))
                        # 扩声道
                        if channels == 2:
                            pcm = np.stack([pcm, pcm], axis=-1)
                        else:
                            pcm = pcm.reshape(-1,1)
                        # 直接写到流中，保持连续
                        stream.write(pcm)
                    else:
                        if msg == "END_OF_SPEECH":
                            break
                        # 忽略其他消息

    except Exception as e:
        print(f"[Error] {e}")
    finally:
        # 3) 停止并关闭流
        stream.stop()
        stream.close()

async def main():
    load_dotenv()
    print("=== TTS 客户端启动 ===")
    while True:
        text = input("输入文本（exit 退出）: ").strip()
        if text.lower() == 'exit':
            break
        await tts_pipeline(text)

if __name__ == "__main__":
    asyncio.run(main())