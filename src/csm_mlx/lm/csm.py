import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Any, Tuple

from csm_mlx.lm.rq_transformer import (
    RQTransformerModelArgs,
    TransformerBlock,
    create_attention_mask,
)
from csm_mlx.lm.config import ModelType


class CSMModel(nn.Module):
    def __init__(
        self,
        config: RQTransformerModelArgs,
        model_type: ModelType,
        depth: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        if depth is None:
            self.depth = self.config.num_codebooks
        else:
            # Assume validation happened at entry point
            self.depth = depth

        if model_type.family != "csm":
            raise ValueError(f"Cannot load weights for {model_type.family}")

        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        self.codebook0_head = nn.Linear(config.dim, config.codebook_size, bias=False)
        self.audio_head = mx.zeros(
            shape=[config.num_codebooks - 1, config.fast_dim, config.codebook_size]
        )

        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]

        self.fast_project_in = nn.Linear(config.dim, config.fast_dim, bias=False)
        self.fast_layers = [
            TransformerBlock(config, is_fast=True) for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = nn.RMSNorm(config.fast_dim, eps=config.norm_eps)

    def forward_generate(
        self, inputs: mx.array, prompt_masks: mx.array, cache: Optional[List[Any]]
    ) -> Tuple[mx.array, mx.array]:
        x = self._embed_tokens(inputs, prompt_masks)
        mask = create_attention_mask(x, cache) if x.shape[1] > 1 else None

        for layer, layer_cache in zip(self.layers, cache or [None] * len(self.layers)):
            x = layer(x, mask=mask, cache=layer_cache)

        x = self.norm(x)

        x = x[:, -1, :]  # take last token for generation
        c0_logits = self.codebook0_head(x)
        return (c0_logits, x)

    def forward_generate_fast(
        self, x: mx.array, cache: List[Any], codebook_idx: int
    ) -> mx.array:
        mask = create_attention_mask(x, cache) if x.shape[1] >= 1 else None
        x = self.fast_project_in(x)

        for layer, layer_cache in zip(self.fast_layers, cache):
            x = layer(x, mask=mask, cache=layer_cache)

        fast_out = self.fast_norm(x)
        ci_logits = fast_out[:, -1, :] @ self.audio_head[codebook_idx - 1]
        return ci_logits

    def embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        return self.codebook_embeddings(tokens + codebook * self.config.codebook_size)

    def _embed_tokens(self, inputs: mx.array, masks: mx.array) -> mx.array:
        text_embeds = self.embeddings(inputs[:, :, -1])[:, :, mx.newaxis, :]
        audio_tokens = inputs[:, :, :-1] + mx.arange(
            0,
            self.depth * self.config.codebook_size,
            self.config.codebook_size,
        )
        audio_embeds = self.codebook_embeddings(audio_tokens)

        embeds = mx.concat([audio_embeds, text_embeds], axis=-2)
        embeds = embeds * masks[:, :, :, mx.newaxis]
        return mx.sum(embeds, axis=-2)
