import sounddevice as sd
import websocket
import threading
import time
import webrtcvad
import numpy as np
import collections

# WebSocket 服务器地址
WS_URL = "ws://服务器IP:8765"
# 设置 VAD 参数
vad = webrtcvad.Vad(2)
frame_duration = 30  # 每帧时长（毫秒）
frame_size = int(16000 * frame_duration / 1000 * 2)  # 每帧字节数（16-bit 单声道）

# 用于缓存音频帧
frame_buffer = collections.deque(maxlen=10)

def audio_callback(indata, frames, time_info, status):
    """音频回调函数，将音频帧发送到 WebSocket 服务器"""
    if status:
        print("Error:", status)
    frame_buffer.append(indata)
    if len(frame_buffer) == frame_buffer.maxlen:
        frame = np.concatenate(list(frame_buffer), axis=0)
        if vad.is_speech(frame.tobytes(), 16000):
            ws.send(frame.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)

def on_open(ws):
    """WebSocket 连接建立时的回调函数"""
    print("WebSocket 连接已建立")
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, dtype='int16')
    stream.start()
    print("开始录音... 按 Ctrl+C 停止")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stream.stop()
        ws.close()

def on_message(ws, message):
    """接收到服务器消息时的回调函数"""
    print("收到服务器消息:", message)

def on_error(ws, error):
    """WebSocket 错误时的回调函数"""
    print("WebSocket 错误:", error)

def on_close(ws, close_status_code, close_msg):
    """WebSocket 连接关闭时的回调函数"""
    print("WebSocket 连接已关闭")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(WS_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
