import asyncio
import websockets
import pyaudio
import collections
import webrtcvad
import numpy as np
# 设置 VAD 参数
vad = webrtcvad.Vad(2)
frame_duration = 30  # 每帧时长（毫秒）
frame_size = int(16000 * frame_duration / 1000 * 2)  # 每帧字节数（16-bit 单声道）
frame_buffer = collections.deque(maxlen=10)  # 缓存 10 帧（300ms）

# 初始化 PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                output=True)

async def audio_handler(websocket, path):
    """处理接收到的音频数据"""
    print("客户端已连接")
    try:
        async for message in websocket:
            frame = np.frombuffer(message, dtype=np.int16)
            frame_buffer.append(frame)
            if len(frame_buffer) == frame_buffer.maxlen:
                audio_data = np.concatenate(list(frame_buffer), axis=0)
                if not any(vad.is_speech(f.tobytes(), 16000) for f in frame_buffer):
                    print("检测到用户停止说话")
                    # 触发 ASR/SER 处理
                    process_audio(audio_data)
                    frame_buffer.clear()
                stream.write(audio_data.tobytes())
    except websockets.ConnectionClosed:
        print("客户端连接已关闭")

def process_audio(audio_data):
    """处理音频数据（例如：ASR/SER）"""
    print("处理音频数据...")
    # 在这里添加 ASR/SER 处理逻辑

start_server = websockets.serve(audio_handler, "0.0.0.0", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
print("服务器已启动，等待客户端连接...")
asyncio.get_event_loop().run_forever()
