import os
import sys
import torchaudio
import torch

from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice2

# -------------------- 路径配置 --------------------
sys.path.append('third_party/Matcha-TTS')
sys.path.append('/home/chengzi/projects/github/CosyVoice/third_party/Matcha-TTS/matcha')

# -------------------- 初始化模型 --------------------
tts = CosyVoice2(
    model_dir='pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    load_vllm=False,
    fp16=False,
)

# -------------------- 输入文本 & Prompt --------------------
text = """
寻找多语种/多口音支持的模型：你可能需要寻找专门为多语种或多口音训练的 TTS 模型。
有些开源模型或商业服务会提供这方面的支持。
"""
prompt = load_wav('susu.wav', 16000)
global_prompt_text = (
    "爆炸豆家的智能电脑,虽然我只是一台电脑.但我可是超级智能的,"
    "我可以回答各种问题,还能和你们聊天,分享我的见闻.最重要的是,"
    "我还是爆炸豆的好伙伴"
)

# -------------------- 流式合成并保存 --------------------
chunks = []

for res in tts.inference_zero_shot(
    tts_text=text,
    prompt_text=global_prompt_text,
    prompt_speech_16k=prompt,
    stream=True,
):
    wav = res["tts_speech"]  # Tensor, shape [1, T] or [T]
    # 规范成 [1, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # 直接累积这个 chunk
    chunks.append(wav.cpu())

# 如果没有 chunk，给个静音
if not chunks:
    full = torch.zeros((1, 1))
else:
    # 拼接所有 chunk
    full = torch.cat(chunks, dim=1)  # [1, total_T]

# 归一化（可选）
full = torch.clamp(full, -1.0, 1.0)
peak = full.abs().max().item()
if peak > 0:
    full = full / peak * 0.95

# 保存到 wav
output_path = "streamed_output.wav"
torchaudio.save(output_path, full, tts.sample_rate)
print(f"✅ Streamed audio saved to {output_path}")
