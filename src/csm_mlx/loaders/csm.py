from dataclasses import dataclass
from huggingface_hub import snapshot_download
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
from tokenizers import Tokenizer
from typing import Optional, List, Tuple
from tqdm import tqdm

from csm_mlx.codec.mimi import load_mimi, MimiModel
from csm_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
)
from csm_mlx.lm.csm import CSMModel
from csm_mlx.lm.cache import make_prompt_cache
from csm_mlx.lm.config import ModelType
from csm_mlx.lm.utils.prompt import CSMPromptEncoder
from csm_mlx.generate.csm import SingleBatchGenerator
from csm_mlx.generate.utils import GenerationSettings
from csm_mlx.io.wav import pcm_to_wav_bytes


@dataclass
class Segment:
    speaker: int
    text: str
    audio: np.ndarray
    """
    24khz pcm ndarray
    """


class CSM:
    model: CSMModel
    config: RQTransformerModelArgs
    codec: MimiModel
    prompt_encoder: CSMPromptEncoder
    kv_cache: Optional[List[any]]

    def __init__(
        self,
        model_id="jkeisling/csm-1b",
        checkpoint_dir: Optional[str] = None,
        quantization: str = "bf16",
        depth=32,
    ):
        checkpoint_dir = Path(
            checkpoint_dir
            if checkpoint_dir is not None
            else snapshot_download(model_id)
        )
        config = RQTransformerModelArgs.from_json_file(
            str(checkpoint_dir / "config.json")
        )
        model_type = ModelType.csm_1b()

        config = RQTransformerModelArgs.from_json_file(
            str(checkpoint_dir / "config.json")
        )
        if depth > config.num_codebooks or depth < 4:
            raise ValueError(
                f"Only depths between 4 and {self.config.num_codebooks} supported; got {depth}"
            )
        self.depth = depth

        tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
        prompt_encoder = CSMPromptEncoder(tokenizer, depth=depth)

        model = CSMModel(config, model_type, depth=depth)
        model_path = str(checkpoint_dir / "model.safetensors")
        model.load_weights(model_path, strict=True)
        if quantization == "bf16":
            model = model.apply(lambda p: p.astype(mx.bfloat16))
        elif quantization == "q8":
            nn.quantize(model, group_size=64, bits=8)
        mx.eval(model.parameters())
        model.eval()

        self.codec = load_mimi()
        self.model = model
        self.prompt_encoder = prompt_encoder
        self.sampling_rate = 24_000

        # State mgmt
        self.kv_cache = None

    def __call__(
        self,
        text: str,
        speaker_id: int,
        context: Optional[List[Segment]] = None,
        use_last_gens: bool = False,
        keep_prompt_only: bool = False,
        temp: float = 0.9,
        fast_temp: float = 0.9,
        top_k: int = 64,
        backbone_min_p: float = 0.05,
    ) -> np.ndarray:
        """
        Blocking E2E audio generation; returns 24khz PCM
        """
        prompt, prompt_mask = self._prompt_encode(
            text, speaker_id, context if context is not None else []
        )
        decode_start_time = time.time()
        prev_kv_cache = (
            self.kv_cache if self.kv_cache is not None and keep_prompt_only else None
        )

        kv_cache = (
            self.kv_cache
            if self.kv_cache is not None and use_last_gens
            else make_prompt_cache(self.model)
        )
        gen = SingleBatchGenerator(
            self.model,
            prompt,
            prompt_mask,
            GenerationSettings(
                default_temp=temp,
                default_fast_temp=fast_temp,
                top_k=top_k,
                min_p=backbone_min_p,
            ),
            kv_cache,
        )
        prefill_start_time = time.time()
        codes = [next(gen)]
        prefill_end_time = time.time()
        prefill_ms = (prefill_end_time - prefill_start_time) * 1000
        print(
            f"{prefill_ms:3f}ms prompt processing: {prompt.shape[1]} tokens ({prompt.shape[-1] / (prefill_end_time - prefill_start_time):3f} tokens/s)"
        )

        # accumulate codes blocking
        for step in tqdm(gen):
            if step is not None:
                codes.append(step)

        mx.eval(codes)

        out_len = len(codes) - 1
        codes = mx.concat(codes, axis=-1)
        decode_end_time = time.time()
        decode_duration = decode_end_time - decode_start_time
        frame_rate = 12.5
        pcm = self.codec.decode(codes)
        print(
            f"Generated in {decode_duration:.2f}s ({(out_len / decode_duration):.2f} tokens/s, {((decode_duration * 1000) / out_len):.2f}ms/token), {(out_len / frame_rate) / decode_duration:.2f}x realtime"
        )
        mx.metal.clear_cache()

        # Persist history in case we need it
        if prev_kv_cache is None:
            self.kv_cache = kv_cache
        else:
            self.kv_cache = prev_kv_cache

        return np.array(pcm).flatten()

    def warmup(self):
        print("Warming up the model")
        self.__call__("This is a test", 0)

    def stream(
        self,
        text: str,
        speaker_id: int,
        context: Optional[List[Segment]] = None,
        use_last_gens: bool = False,
        temp: float = 0.9,
        fast_temp: float = 0.9,
        top_k: int = 64,
        backbone_min_p: float = 0.05,
        keep_prompt_only=False
    ):
        """
        TODO this is very repetitive, I'll refactor this later
        """
        prompt, prompt_mask = self._prompt_encode(
            text, speaker_id, context if context is not None else []
        )
        prev_kv_cache = (
            self.kv_cache if self.kv_cache is not None and keep_prompt_only else None
        )

        kv_cache = (
            self.kv_cache
            if self.kv_cache is not None and use_last_gens
            else make_prompt_cache(self.model)
        )

        mimi_kv_cache = make_prompt_cache(self.codec.decoder_transformer)
        gen = SingleBatchGenerator(
            self.model,
            prompt,
            prompt_mask,
            GenerationSettings(
                default_temp=temp,
                default_fast_temp=fast_temp,
                top_k=top_k,
                min_p=backbone_min_p,
            ),
            kv_cache,
        )
        for frame in tqdm(gen):
            if frame is not None:
                pcm_chunk = self.codec.decode_step(frame, mimi_kv_cache)
                audio_data = np.array(pcm_chunk).flatten()
                yield audio_data

        # Persist history in case we need it
        if prev_kv_cache is None:
            self.kv_cache = kv_cache
        else:
            self.kv_cache = prev_kv_cache

        self.codec.decoder.reset()
        mx.metal.clear_cache()

    def _prompt_encode(
        self, text: str, speaker_id: int, segments: List[Segment]
    ) -> Tuple[mx.array, mx.array]:
        """
        Returns text, text mask
        """
        prompt_segments = []
        mask_segments = []

        # TODO parallelize this if it ever gets slow
        for segment in segments:
            tokens, tokens_mask = self.prompt_encoder.tokenize_text(
                f"[{segment.speaker}]{segment.text}"
            )
            prompt_segments.append(tokens)
            mask_segments.append(tokens_mask)

            # TODO allow for precomputed prompt / cache; just getting it working for now
            # assumes audio is 24khz resampled elsewhere
            mimi_codes = self.codec.encode(
                mx.array(segment.audio)[mx.newaxis, mx.newaxis, :]
            )
            mimi_codes = mimi_codes[:, : self.depth, :]
            audio, audio_mask = self.prompt_encoder.tokenize_audio(mimi_codes)
            prompt_segments.append(audio)
            mask_segments.append(audio_mask)

        curr_tokens, curr_tokens_mask = self.prompt_encoder.tokenize_text(
            f"[{speaker_id}]{text}"
        )
        prompt_segments.append(curr_tokens)
        mask_segments.append(curr_tokens_mask)

        prompt = mx.concat(prompt_segments, axis=0)[mx.newaxis, :, :]
        prompt_mask = mx.concat(mask_segments, axis=0)[mx.newaxis, :, :]
        return prompt, prompt_mask
