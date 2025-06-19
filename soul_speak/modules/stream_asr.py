# from funasr import AutoModel
#
# chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
# encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
# decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention
#
# model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
#
# import soundfile
# import os
#
# wav_file = os.path.join(model.model_path, "../fa-zh/example/asr_example.wav")
# speech, sample_rate = soundfile.read(wav_file)
# chunk_stride = chunk_size[1] * 960  # 600ms
#
# cache = {}
# total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
# for i in range(total_chunk_num):
#     speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
#     is_final = i == total_chunk_num - 1
#     res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
#                          encoder_chunk_look_back=encoder_chunk_look_back,
#                          decoder_chunk_look_back=decoder_chunk_look_back)
#     print(res)
# asr_module.py
from funasr import AutoModel
import numpy as np
import soundfile as sf
import collections
class ASRStream:
    def __init__(self):
        self.model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
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
        if results:
            return results[0]
        return ""

