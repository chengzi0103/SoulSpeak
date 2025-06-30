import asyncio
import json
import websockets
import sounddevice as sd
import numpy as np
from scipy import signal
from dotenv import load_dotenv

WS_URL = "ws://10.100.1.111:8000/ws/synthesize"  # 修改为你的服务器地址
SERVER_SR = 24000
PING_INTERVAL = 60
PING_TIMEOUT = 30

def setup_audio_stream():
    dev = sd.query_devices(kind="output")
    sr = int(dev["default_samplerate"])
    ch = 2 if dev["max_output_channels"] >= 2 else 1
    print(f"[Client] Audio device SR={sr}Hz, channels={ch}")
    return sr, ch

async def tts_pipeline(sentences):
    device_sr, channels = setup_audio_stream()

    stream = sd.OutputStream(
        samplerate=device_sr,
        channels=channels,
        dtype="float32",
        blocksize=0
    )
    stream.start()

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT
        ) as ws:
            await ws.ping()

            for idx, sent in enumerate(sentences, 1):
                print(f"[Client] Synthesizing {idx}/{len(sentences)}: {sent[:30]}…")
                await ws.send(json.dumps({"event": "text", "text": sent}))
                await ws.send(json.dumps({"event": "end_of_speech"}))

                while True:
                    msg = await ws.recv()
                    if isinstance(msg, (bytes, bytearray)):
                        pcm16 = np.frombuffer(msg, dtype=np.int16)
                        pcm = pcm16.astype(np.float32) / 32767.0
                        if SERVER_SR != device_sr:
                            pcm = signal.resample(pcm, int(len(pcm) * device_sr / SERVER_SR))
                        if channels == 2:
                            pcm = np.stack([pcm, pcm], axis=-1)
                        else:
                            pcm = pcm.reshape(-1, 1)
                        stream.write(pcm)
                    else:
                        if msg == "END_OF_SPEECH":
                            break
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        stream.stop()
        stream.close()

async def main():
    load_dotenv()
    print("=== 多句 TTS 客户端启动 ===")
    print("请输入多句文本（每句一行），输入空行结束：")

    while True:
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line.strip())

        if not lines:
            print("输入为空，退出程序。")
            break

        await tts_pipeline(lines)

if __name__ == "__main__":
    asyncio.run(main())
