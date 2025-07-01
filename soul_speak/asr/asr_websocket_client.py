import sounddevice as sd
import websocket
import threading
import time
import webrtcvad

from soul_speak.utils.hydra_config.init import conf

# WebSocket 服务器地址
# WS_URL = "ws://10.100.1.111:8765"
WS_URL = f"ws://{conf.asr.websocket.host}:{conf.asr.websocket.port}"
# 设置 VAD 参数
vad = webrtcvad.Vad(2)
# frame_duration = 30  # 每帧时长（毫秒）
frame_duration = conf.asr.send_audio.frame_duration  # 每帧时长（毫秒）
frame_size = int(16000 * frame_duration / 1000 * 2)  # 每帧字节数（16-bit 单声道）


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Error:", status)
    frame = indata[:, 0].copy()
    try:
        if vad.is_speech(frame.tobytes(), 16000):
            if ws is not None:
                ws.send(frame.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
            else:
                print("WebSocket 尚未连接")
    except webrtcvad.VadError as e:
        print("VAD 处理错误:", e)


def record_audio():
    stream = sd.InputStream(callback=audio_callback, channels=1,
                            samplerate=conf.asr.send_audio.samplerate, dtype='int16', blocksize=480)
    stream.start()
    print("开始录音... 按 Ctrl+C 停止")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stream.stop()
        print("录音停止")

def on_open(ws):
    print("WebSocket 连接已建立")
    threading.Thread(target=record_audio, daemon=True).start()

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
    global ws
    ws = websocket.WebSocketApp(WS_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
