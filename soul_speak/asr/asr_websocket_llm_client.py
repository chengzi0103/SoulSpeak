
import asyncio
import sounddevice as sd
import webrtcvad
import collections
import time
import websockets

from soul_speak.llm.run_and_speak import tts_pipeline
from soul_speak.utils.hydra_config.init import conf

# --- 新增：全局打断事件，用来通知 TTS 停止播放 ---
interrupt_event: asyncio.Event = None

WS_URL = f"ws://{conf.asr.websocket.host}:{conf.asr.websocket.port}"
SAMPLE_RATE = conf.asr.send_audio.samplerate
FRAME_DURATION_MS = conf.asr.send_audio.frame_duration
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_MODE = 0
VAD_SILENCE_TIMEOUT = conf.asr.send_audio.vad_silence_timeout
MIN_CHAR_COUNT = 3

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

# --- 音频回调，新增“打断检测” ---
def audio_callback(indata, frames, time_info, status):
    global triggered, ring_buffer, speech_buffer, silence_start
    global loop, global_ws, processing, interrupt_event

    pcm = indata[:, 0].tobytes()
    is_speech = vad.is_speech(pcm, SAMPLE_RATE)

    # —— 当 TTS 正在播报且检测到用户说话时，触发打断 —— 
    if processing and is_speech and interrupt_event:
        print("🔴 Detected user speech during TTS, interrupting playback")
        interrupt_event.set()

    if not triggered:
        ring_buffer.append(pcm)
        # detect start of speech
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

# --- 用户话语检测 & TTS 调用，复用／重置 interrupt_event ---
async def utterance_detector():
    global current_utterance, last_text_time, processing, interrupt_event
    while True:
        await asyncio.sleep(0.5)
        if (current_utterance.strip()
            and time.time() - last_text_time > VAD_SILENCE_TIMEOUT
            and not processing
            and len(current_utterance.strip()) >= MIN_CHAR_COUNT):

            processing = True
            text = current_utterance.strip()
            current_utterance = ""
            print(f"📣 用户话语完成: {text}")

            # —— 如果已有未结束的 TTS，就先打断并创建新事件 —— 
            if interrupt_event:
                interrupt_event.set()
            interrupt_event = asyncio.Event()

            try:
                await tts_pipeline(
                    # 传入新的 interrupt_event，TTS 流程中会检测并中断
                    "Please analyze and respond: " + text,
                    interrupt_event=interrupt_event
                )
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
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype='int16',
            blocksize=FRAME_SIZE,
        )
        stream.start()
        print("🎙 录音开始")

        # 启动话语检测任务
        detector_task = asyncio.create_task(utterance_detector())

        try:
            async for msg in ws:
                if isinstance(msg, str):
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
