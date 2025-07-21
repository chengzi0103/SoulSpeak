import asyncio
import json
import websockets
import sounddevice as sd
import numpy as np
from scipy import signal
from dotenv import load_dotenv
from soul_speak.llm.llm_tagged import generate_emilia_tagged, shutdown_llm_system
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

async def tts_pipeline(user_input: str, interrupt_event: asyncio.Event = None):
    """
    接收用户指令并进行 TTS 合成。
    如果 interrupt_event 被 set()，则在合成前或合成中立刻中断并退出。
    """
    # 1. 文本切句
    sentences = await generate_emilia_tagged(user_input)
    print("[LLM] Sentences:", sentences)

    # 2. 准备播放流
    device_sr, channels = setup_audio_stream()
    stream = sd.OutputStream(
        samplerate=device_sr,
        channels=channels,
        dtype="float32",
        blocksize=0
    )
    stream.start()

    try:
        # 3. 建立到 TTS 服务的 WebSocket 连接
        async with websockets.connect(
            WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT
        ) as ws:
            await ws.ping()

            # 4. 逐句合成并播放
            for idx, sent in enumerate(sentences, 1):
                # —— 合成前检查打断 —— 
                if interrupt_event and interrupt_event.is_set():
                    print("[TTS] Detected interrupt before sentence, exiting.")
                    return

                print(f"[Client] Synthesizing {idx}/{len(sentences)}: {sent[:30]}…")
                await ws.send(json.dumps({"event": "text", "text": sent}))
                await ws.send(json.dumps({"event": "end_of_speech"}))

                # 5. 接收并播放音频
                while True:
                    # —— 播放中也要随时检查打断 —— 
                    if interrupt_event and interrupt_event.is_set():
                        print("[TTS] Detected interrupt during playback, aborting.")
                        return

                    msg = await ws.recv()
                    # 二进制帧是音频
                    if isinstance(msg, (bytes, bytearray)):
                        pcm16 = np.frombuffer(msg, dtype=np.int16)
                        pcm = pcm16.astype(np.float32) / 32767.0

                        # 重采样
                        if SERVER_SR != device_sr:
                            pcm = signal.resample(pcm,
                                int(len(pcm) * device_sr / SERVER_SR)
                            )
                        # 通道扩展
                        if channels == 2:
                            pcm = np.stack([pcm, pcm], axis=-1)
                        else:
                            pcm = pcm.reshape(-1, 1)

                        stream.write(pcm)
                    else:
                        # 文本消息：END_OF_SPEECH 表示本句结束
                        if msg == "END_OF_SPEECH":
                            break
                        # 其他文本忽略

    except Exception as e:
        print(f"[Error] {e}")
    finally:
        # 确保流关闭
        try:
            stream.stop()
            stream.close()
        except:
            pass

async def main():
    load_dotenv()
    print("=== TTS 客户端启动 ===")
    while True:
        text = input("输入文本（exit 退出）: ").strip()
        if text.lower() == 'exit':
            break

        # 每次调用前都为打断创建新的 Event
        interrupt_event = asyncio.Event()
        await tts_pipeline(text, interrupt_event)

if __name__ == "__main__":
    asyncio.run(main())
