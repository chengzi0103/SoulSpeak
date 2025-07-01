# import sounddevice as sd
# import websocket
# import threading
# import time
# import webrtcvad
#
# from soul_speak.utils.hydra_config.init import conf
#
# # WebSocket 服务器地址
# # WS_URL = "ws://10.100.1.111:8765"
# WS_URL = f"ws://{conf.asr.websocket.host}:{conf.asr.websocket.port}"
# # 设置 VAD 参数
# vad = webrtcvad.Vad(2)
# # frame_duration = 30  # 每帧时长（毫秒）
# frame_duration = conf.asr.send_audio.frame_duration  # 每帧时长（毫秒）
# frame_size = int(16000 * frame_duration / 1000 * 2)  # 每帧字节数（16-bit 单声道）
#
#
# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print("Error:", status)
#     frame = indata[:, 0].copy()
#     try:
#         if vad.is_speech(frame.tobytes(), 16000):
#             if ws is not None:
#                 ws.send(frame.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
#             else:
#                 print("WebSocket 尚未连接")
#     except webrtcvad.VadError as e:
#         print("VAD 处理错误:", e)
#
#
# def record_audio():
#     stream = sd.InputStream(callback=audio_callback, channels=1,
#                             samplerate=conf.asr.send_audio.samplerate, dtype='int16', blocksize=480)
#     stream.start()
#     print("开始录音... 按 Ctrl+C 停止")
#     try:
#         while True:
#             time.sleep(0.1)
#     except KeyboardInterrupt:
#         stream.stop()
#         print("录音停止")
#
# def on_open(ws):
#     print("WebSocket 连接已建立")
#     threading.Thread(target=record_audio, daemon=True).start()
#
# def on_message(ws, message):
#     """接收到服务器消息时的回调函数"""
#     print("收到服务器消息:", message)
#
# def on_error(ws, error):
#     """WebSocket 错误时的回调函数"""
#     print("WebSocket 错误:", error)
#
# def on_close(ws, close_status_code, close_msg):
#     """WebSocket 连接关闭时的回调函数"""
#     print("WebSocket 连接已关闭")
#
# if __name__ == "__main__":
#     global ws
#     ws = websocket.WebSocketApp(WS_URL,
#                                 on_open=on_open,
#                                 on_message=on_message,
#                                 on_error=on_error,
#                                 on_close=on_close)
#     ws.run_forever()
import asyncio
import sounddevice as sd
import webrtcvad
import collections
import time
import websockets

from soul_speak.llm.run_and_speak import tts_pipeline
from soul_speak.utils.hydra_config.init import conf

# --- 配置 ---
WS_URL = f"ws://{conf.asr.websocket.host}:{conf.asr.websocket.port}"
SAMPLE_RATE = conf.asr.send_audio.samplerate
FRAME_DURATION_MS = conf.asr.send_audio.frame_duration
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_MODE = 2
VAD_SILENCE_TIMEOUT = conf.asr.send_audio.vad_silence_timeout
MIN_CHAR_COUNT = 3  # 最少字符数

# --- 全局状态 ---
vad = webrtcvad.Vad(VAD_MODE)
ring_buffer = collections.deque(maxlen=int(300 / FRAME_DURATION_MS))
triggered = False
speech_buffer = bytearray()
silence_start = None

current_utterance = ""
last_text_time = time.time()
processing = False
loop = None
global_ws = None

# --- 发送音频段 ---
async def send_audio_segment(ws):
    global speech_buffer
    if speech_buffer:
        await ws.send(speech_buffer)
        speech_buffer = bytearray()
    await ws.send("$$END$$")

# --- 音频回调 ---
def audio_callback(indata, frames, time_info, status):
    global triggered, ring_buffer, speech_buffer, silence_start, loop, global_ws
    pcm = indata[:, 0].tobytes()
    is_speech = vad.is_speech(pcm, SAMPLE_RATE)

    if not triggered:
        ring_buffer.append(pcm)
        if sum(vad.is_speech(f, SAMPLE_RATE) for f in ring_buffer) > 0.9 * ring_buffer.maxlen:
            triggered = True
            speech_buffer.extend(b"".join(ring_buffer))
            ring_buffer.clear()
    else:
        speech_buffer.extend(pcm)
        if not is_speech:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > VAD_SILENCE_TIMEOUT:
                # 触发发送
                if loop and global_ws:
                    asyncio.run_coroutine_threadsafe(
                        send_audio_segment(global_ws), loop)
                triggered = False
                silence_start = None
        else:
            silence_start = None

# --- 用户话语检测 ---
async def utterance_detector():
    global current_utterance, last_text_time, processing
    while True:
        await asyncio.sleep(0.5)
        if (current_utterance.strip() and
            time.time() - last_text_time > VAD_SILENCE_TIMEOUT and
            not processing and
            len(current_utterance.strip()) >= MIN_CHAR_COUNT):

            processing = True
            text = current_utterance.strip()
            current_utterance = ""
            print(f"📣 用户话语完成: {text}")
            try:
                await tts_pipeline("""This is the content of the user's voice. Since it is recognized by ASR, there may be text errors. Please analyze the user's input.  : """ + text)
            except Exception as e:
                print(f"❗ TTS 失败: {e}")
            processing = False

# --- 主流程 ---
async def main():
    global loop, global_ws, current_utterance, last_text_time, processing
    loop = asyncio.get_running_loop()

    print(f"连接 ASR 服务: {WS_URL}")
    async with websockets.connect(WS_URL) as ws:
        global_ws = ws
        # 启动录音
        stream = sd.InputStream(callback=audio_callback,
                                channels=1, samplerate=SAMPLE_RATE,
                                dtype='int16', blocksize=FRAME_SIZE,)
        sd.default.device = (1, 2)
        stream.start()
        print("🎙 录音开始")

        # 启动话语检测任务
        detector_task = asyncio.create_task(utterance_detector())

        try:
            async for msg in ws:
                if isinstance(msg, str):
                    # 声明全局变量
                    global current_utterance, last_text_time
                    current_utterance += msg
                    last_text_time = time.time()
                    print(f"⏱ 收到 ASR 文本: {msg}")
                else:
                    print("⚠️ 非文本消息忽略")
        except Exception as e:
            print(f"❗ WS 错误: {e}")
        finally:
            detector_task.cancel()
            stream.stop()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("退出客户端")
