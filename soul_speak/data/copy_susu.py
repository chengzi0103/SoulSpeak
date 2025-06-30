import os

from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice2
import sys

sys.path.append('third_party/Matcha-TTS')
model_dir = os.path.expanduser('~/.cache/modelscope/hub/iic/CosyVoice2-0.5B')

tts = CosyVoice2(
    model_dir=model_dir,
    load_jit=False, load_trt=False, load_vllm=False, fp16=False
)
prompt = load_wav('output.wav', 16000)
assert tts.add_zero_shot_spk('', prompt, 'susu_spk') is True
tts.save_spkinfo()
