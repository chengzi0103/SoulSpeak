
from funasr import AutoModel
import numpy as np

class ASRStream:
    def __init__(self):
        self.model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4",hub="hf",)
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

