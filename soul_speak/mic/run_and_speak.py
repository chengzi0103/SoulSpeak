#
# import asyncio
# import json
# import websockets
# import sounddevice as sd
# import numpy as np
# from scipy import signal
# from dotenv import load_dotenv
# from emilia_llm_tagged import generate_emilia_tagged
#
# # WebSocket TTS 服务地址
# WS_URL = "ws://10.100.1.111:8000/ws/synthesize"
#
# # 服务端音频参数
# SERVER_SR = 24000    # 24 kHz
# BLOCK_TIME = 0.1     # 每次写入 100 ms
# PING_INTERVAL = 60
# PING_TIMEOUT = 30
#
# def setup_audio_stream():
#     dev = sd.query_devices(kind='output')
#     sr = int(dev['default_samplerate'])
#     ch = 2 if dev['max_output_channels'] >= 2 else 1
#     print(f"[Client] Device SR={sr}Hz, channels={ch}")
#     return sr, ch
#
# async def receive_and_play_audio(ws, device_sr, channels):
#     """
#     1) 累积 PCM 数据到 buffer。
#     2) 当缓存 >= BLOCK_TIME 时，写入一次 OutputStream。
#     3) 接收到 END_OF_SPEECH 后，将剩余缓存全部写入并返回。
#     """
#     block_frames = int(device_sr * BLOCK_TIME)
#     buffer = np.zeros((0, channels), dtype=np.float32)
#
#     with sd.OutputStream(samplerate=device_sr, channels=channels, dtype='float32') as stream:
#         while True:
#             msg = await ws.recv()
#             if isinstance(msg, (bytes, bytearray)):
#                 # bytes -> float32 PCM [-1,1]
#                 pcm16 = np.frombuffer(msg, dtype=np.int16)
#                 pcm = pcm16.astype(np.float32) / 32767.0
#                 # 重采样
#                 if SERVER_SR != device_sr:
#                     pcm = signal.resample(pcm, int(len(pcm)*device_sr/SERVER_SR))
#                 # 单声道->多声道
#                 if channels == 2:
#                     pcm = np.stack([pcm, pcm], axis=-1)
#                 else:
#                     pcm = pcm.reshape(-1,1)
#                 # 累积
#                 buffer = np.concatenate([buffer, pcm], axis=0)
#
#                 # 当缓存超过一块时，写入一次
#                 while len(buffer) >= block_frames:
#                     to_write = buffer[:block_frames]
#                     stream.write(to_write)
#                     buffer = buffer[block_frames:]
#
#             else:
#                 # 控制结束
#                 if msg == "END_OF_SPEECH":
#                     # 写入剩余
#                     if len(buffer) > 0:
#                         stream.write(buffer)
#                     break
#
# async def tts_pipeline(user_input: str):
#     sentences = await generate_emilia_tagged(user_input)
#     print("[LLM] Generated tagged sentences:", sentences)
#
#     device_sr, channels = setup_audio_stream()
#
#     async with websockets.connect(
#         WS_URL,
#         ping_interval=PING_INTERVAL,
#         ping_timeout=PING_TIMEOUT
#     ) as ws:
#         await ws.ping()
#         for idx, sent in enumerate(sentences, 1):
#             print(f"[Client] Synthesizing sentence {idx}/{len(sentences)}")
#             # 发送文本和控制命令
#             await ws.send(json.dumps({"event":"text","text":sent}))
#             await ws.send(json.dumps({"event":"end_of_speech"}))
#             # 等服务器稍微开始推流
#             await asyncio.sleep(0.1)
#             # 接收并播放
#             await receive_and_play_audio(ws, device_sr, channels)
#
# async def main():
#     load_dotenv()
#     print("=== TTS 客户端 ===")
#     while True:
#         text = input("请输入合成文本（exit 退出）: ").strip()
#         if text.lower() == 'exit':
#             break
#         try:
#             await tts_pipeline(text)
#         except Exception as e:
#             print(f"[Client] Error: {e}")
#             await asyncio.sleep(1)
#
# if __name__ == "__main__":
#     asyncio.run(main())



import asyncio
import json
import websockets
import sounddevice as sd
import numpy as np
from scipy import signal
from dotenv import load_dotenv
from emilia_llm_tagged import generate_emilia_tagged

WS_URL = "ws://10.100.1.111:8000/ws/synthesize"
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
