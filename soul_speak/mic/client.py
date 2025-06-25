import asyncio
import websockets
import sounddevice as sd
import numpy as np
from scipy import signal

WS_URL = "ws://10.100.1.111:8000/ws/synthesize"

SERVER_SAMPLERATE = 24000  # 修正为 24kHz
SERVER_DTYPE = 'int16'     # 和服务器一致
SERVER_CHANNELS = 1        # 服务端输出为单声道

async def receive_and_play_audio(websocket):
    print("\n--- Client: Starting audio reception/playback ---")
    try:
        dev = sd.query_devices(kind='output')
        device_sr = int(dev['default_samplerate'])
        max_ch = dev['max_output_channels']
        print(f"[Client] Output device samplerate = {device_sr} Hz, max_channels = {max_ch}")
    except Exception as e:
        print(f"[Client] Failed to query device info: {e}")
        return

    channels = 2 if max_ch >= 2 else 1

    with sd.OutputStream(
        samplerate=device_sr,
        channels=channels,
        dtype='float32',
        blocksize=2048
    ) as stream:
        print(f"[Client] Audio stream started: {device_sr} Hz × {channels}ch")

        while True:
            msg = await websocket.recv()
            if isinstance(msg, bytes):
                pcm_i16 = np.frombuffer(msg, dtype=np.int16)
                audio = pcm_i16.astype(np.float32) / 32767.0

                # 重采样
                if SERVER_SAMPLERATE != device_sr:
                    new_len = int(len(audio) * (device_sr / SERVER_SAMPLERATE))
                    audio = signal.resample(audio, new_len)

                # 单声道转多声道
                if channels == 2:
                    audio = np.stack([audio, audio], axis=-1)

                stream.write(audio)
            else:
                print(f"[Client] Control message: {msg}")
                if msg == "END_OF_SPEECH":
                    print("[Client] End of segment.")
                    break

    print("[Client] Playback stream closed.")

async def main():
    while True:
        text = input("请输入合成文本（exit 退出）: ")
        if text.strip().lower() == 'exit':
            break

        print("[Client] Connecting to server...")
        try:
            async with websockets.connect(WS_URL) as ws:
                await ws.send(text)
                await receive_and_play_audio(ws)
        except Exception as e:
            print(f"[Client] Error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
