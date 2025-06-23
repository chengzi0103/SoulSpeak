# test_qwen2_ser_fixed.py

import numpy as np
import torch
import librosa
from io import BytesIO
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

def main():
    # —— 1. 加载模型与处理器 ——
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        trust_remote_code=True
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    ).eval()

    # —— 2. 读取并预处理音频 ——
    wav_path = "extracted_segment.wav"  # 替换成你的文件名
    audio_np, sr = librosa.load(
        wav_path,
        sr=processor.feature_extractor.sampling_rate,
        mono=True
    )
    # 确保 waveform 是 float32 in [-1,1]
    audio_np = audio_np.astype(np.float32)

    # —— 3. 构造 Prompt ——
    # 必须包含一对 <|AUDIO|> 标记和一个 <|text_bos|> 标记
    prompt = (
        "<|audio_bos|><|AUDIO|><|audio_eos|>"
        "请分析这段语音的情绪。"
        "<|text_bos|>"
    )

    # —— 4. 构造模型输入 ——
    inputs = processor(
        text=prompt,
        audio=audio_np,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    # 把所有 tensor 移动到 GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # —— 5. 推理 ——
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128
        )

    # —— 6. 解码输出 ——
    # 去掉 Prompt 部分，只保留模型生成的新内容
    gen_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(
        gen_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    print("=== ASR+SER Output ===")
    print(response)

if __name__ == "__main__":
    main()
