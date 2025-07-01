
import asyncio
import websockets
import time

from soul_speak.asr.funasr_loader import ASRStream
from soul_speak.utils.hydra_config.init import conf

# ----------- 参数配置 -------------
PUNCTUATION = ("。", "？", "！", ".", "?", "!")
MIN_SENT_LEN = 10
MAX_HISTORY = 5
SILENCE_THRESHOLD = 1.0    # 静默 1s 触发
PERIODIC_INTERVAL = 3.0    # 每 3s 触发一次

# ----------- 全局 ASR & 缓存 -------------
asr = ASRStream()
audio_buffer = bytearray()
last_recv = time.time()
recognize_lock = asyncio.Lock()

# ----------- WebSocket 处理 -------------
async def handler(ws):
    global audio_buffer, last_recv

    text_buffer = ""
    history = []

    async def recognize_and_send():
        nonlocal text_buffer, history

        if recognize_lock.locked():
            return
        async with recognize_lock:
            if not audio_buffer:
                return

            data = bytes(audio_buffer)
            audio_buffer.clear()

            # ASR
            text = await asr.recognize(data, is_final=True)
            print(f"[DEBUG] ASR returned: '{text}'")
            if not text:
                return

            # 累积并判断句子
            text_buffer += text
            is_sentence = any(text_buffer.endswith(p) for p in PUNCTUATION) \
                          or len(text_buffer) >= MIN_SENT_LEN
            if not is_sentence:
                print(f"[DEBUG] Partial buffer (waiting): '{text_buffer}'")
                return

            # 完整一句
            sentence = text_buffer
            text_buffer = ""
            history.append(sentence)
            if len(history) > MAX_HISTORY:
                history.pop(0)

            prompt = "\n".join(history) + "\n请分析以上内容的情绪："
            print(prompt)

    async def silence_checker():
        while True:
            await asyncio.sleep(0.5)
            if time.time() - last_recv > SILENCE_THRESHOLD:
                await recognize_and_send()

    async def periodic_checker():
        while True:
            await asyncio.sleep(PERIODIC_INTERVAL)
            print("[DEBUG] periodic triggered")
            await recognize_and_send()

    # 启动检测任务
    asyncio.create_task(silence_checker())
    asyncio.create_task(periodic_checker())

    print("📡 Client connected")
    try:
        async for msg in ws:
            audio_buffer.extend(msg)
            last_recv = time.time()
    except websockets.ConnectionClosed:
        print("❎ Client disconnected")
        await recognize_and_send()

async def main():
    print("🚀 Server starting on port 8765...")
    async with websockets.serve(handler, "0.0.0.0", conf.asr.websocket.port):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
