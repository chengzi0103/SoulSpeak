import asyncio
import websockets
import pyaudio
import collections
import webrtcvad
import numpy as np

from soul_speak.modules.stream_asr import  ASRStream
from soul_speak.utils.port.port import kill_process_on_port

# 端口释放函数

asr = ASRStream()

# VAD 配置
vad = webrtcvad.Vad(2)
frame_duration = 30  # 毫秒
frame_size = int(16000 * frame_duration / 1000 * 2)  # 字节数（16kHz * 2字节）
frame_buffer = collections.deque(maxlen=10)  # 10帧 = 300ms 缓存

# 初始化 PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)

audio_buffer = np.array([], dtype=np.int16)  # 全局缓存

# 音频处理器
async def audio_handler(websocket):
    global audio_buffer
    try:
        async for message in websocket:
            chunk = np.frombuffer(message, dtype=np.int16)
            audio_buffer = np.concatenate((audio_buffer, chunk))

            chunk_len = 24000  # 1.5秒 @16kHz
            while len(audio_buffer) >= chunk_len:
                segment = audio_buffer[:chunk_len]
                audio_buffer = audio_buffer[chunk_len:]

                text = await asr.recognize(segment.tobytes(), is_final=False)
                if text:
                    print("识别结果:", text)

        # 断开连接时最后识别剩余音频
        if len(audio_buffer) > 0:
            text = await asr.recognize(audio_buffer.tobytes(), is_final=True)
            if text:
                print("最后识别结果:", text)

    except websockets.ConnectionClosed:
        print("客户端断开连接")



# 模拟处理函数（你可以改成 ASR/SER）
def process_audio(audio_data):
    print("🧠 正在处理音频数据...（此处可接 ASR）",audio_data)


# 主入口
async def main():
    port = 8765
    kill_process_on_port(port)
    server = await websockets.serve(audio_handler, "0.0.0.0", port)
    print(f"🚀 服务器已启动，监听端口 {port}...")
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("❎ 手动终止服务")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
