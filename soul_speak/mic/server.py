# server.py

import asyncio
import websockets
import time
import torch
import numpy as np
from io import BytesIO

from funasr import AutoModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ----------- ASRStream 单例化加载 -------------
print("🔄 Loading ASR model...")
ASR_MODEL = AutoModel(
    model="paraformer-zh-streaming",
    model_revision="v2.0.4",
    disable_update=True,
    hub="hf"
)
print("✅ ASR model loaded.")

class ASRStream:
    def __init__(self):
        self.model = ASR_MODEL
        self.cache = {}
        self.chunk_size = [0, 10, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

    async def recognize(self, audio_chunk: bytes, is_final=False):
        np_audio = np.frombuffer(audio_chunk, dtype=np.int16)
        results = self.model.generate(
            input=np_audio,
            cache=self.cache,
            is_final=is_final,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back
        )
        if not results:
            return ""
        rst = results[0]
        if isinstance(rst, dict):
            return rst.get("text", "")
        return rst

# ----------- Qwen2-Audio 模型加载 -------------
print("🔄 Loading Qwen2-Audio model...")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True
)
qa_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
).eval()
print("✅ Qwen2-Audio model loaded.")

def analyze_with_qwen_and_context(audio_bytes: bytes, prompt: str) -> str:
    """
    直接把 PCM int16 bytes 转为 float waveform，不走 librosa.load。
    """
    # 1. 原始 PCM int16 -> float32 in [-1,1]
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # 2. 构造输入给 Qwen2-Audio
    inputs = processor(text=prompt, audios=pcm, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt")
    inputs = {k: v.to(qa_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = qa_model.generate(**inputs, max_new_tokens=256)
    return processor.batch_decode(
        out[:, inputs["input_ids"].shape[-1]:]
    )[0].strip()

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

            # SER via Qwen2-Audio
            resp = analyze_with_qwen_and_context(data, prompt)
            print(f"[DEBUG] Qwen response: {resp}")
            await ws.send(resp)

    async def silence_checker():
        while True:
            await asyncio.sleep(0.5)
            if time.time() - last_recv > SILENCE_THRESHOLD:
                print("[DEBUG] silence triggered")
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
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
