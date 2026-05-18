"""Pi0.5 sample_actions composite entry modules."""
from __future__ import annotations

import devproc2 as dp
import devproc2.nn as nn

from .. import ops as pi05_ops
from ._helpers import _grid_1d, _static_dim
from .layers import PI05LanguageEmbedding
from .denoise import PI05DenoiseLoop
from .prefix import PI05PaliGemmaPrefixEncoder
from .vision import PI05VisionEncoder


class PI05SampleActionsFromPrefixEmbeddings(nn.Module):
    """Pi0.5 sample_actions slice from prepared prefix embeddings.

    This is the current single-artifact deploy bridge: token/image embedding
    construction is still supplied by the caller, while the PaliGemma prefix
    transformer, KV materialization, and action denoise loop all run inside one
    DSL/VM graph.
    """

    def __init__(
        self,
        *,
        num_layers: int = 18,
        prefix_hidden_size: int = 2048,
        prefix_intermediate_size: int = 16384,
        decoder_hidden_size: int = 1024,
        decoder_intermediate_size: int = 4096,
        action_horizon: int = 50,
        num_steps: int = 10,
        action_dim: int = 32,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        eps: float = 1.0e-6,
        dtype: str = "bfloat16",
        device: str = "cuda",
        use_static_act_scales: bool = False,
    ) -> None:
        super().__init__()
        self.prefix_encoder = PI05PaliGemmaPrefixEncoder(
            num_layers=num_layers,
            hidden_size=prefix_hidden_size,
            intermediate_size=prefix_intermediate_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
            use_static_act_scales=use_static_act_scales,
        )
        self.denoise_loop = PI05DenoiseLoop(
            num_layers=num_layers,
            hidden_size=decoder_hidden_size,
            intermediate_size=decoder_intermediate_size,
            action_horizon=action_horizon,
            num_steps=num_steps,
            action_dim=action_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
            use_static_act_scales=use_static_act_scales,
        )

    def forward(
        self,
        noise_f32,
        prefix_embs,
        prefix_valid_rows,
        prefix_rope_interleaved,
        suffix_rope_interleaved,
    ):
        prefix_k_cache, prefix_v_cache = self.prefix_encoder.materialize_kv(
            prefix_embs,
            prefix_valid_rows,
            prefix_rope_interleaved,
        )
        return self.denoise_loop(
            noise_f32,
            prefix_k_cache,
            prefix_v_cache,
            prefix_valid_rows,
            suffix_rope_interleaved,
        )

    def forward_fast(
        self,
        noise_f32,
        prefix_embs,
        prefix_valid_rows,
        prefix_rope_interleaved,
        suffix_rope_interleaved,
    ):
        prefix_k_cache, prefix_v_cache = self.prefix_encoder.materialize_kv_fast(
            prefix_embs,
            prefix_valid_rows,
            prefix_rope_interleaved,
        )
        return self.denoise_loop.forward_fast(
            noise_f32,
            prefix_k_cache,
            prefix_v_cache,
            prefix_valid_rows,
            suffix_rope_interleaved,
        )




class PI05SampleActionsFromTokens(nn.Module):
    """Pi0.5 sample_actions slice from images and token ids.

    The caller still supplies token ids, prefix valid row count, and RoPE
    tables. This graph owns the SigLIP vision tower, language embedding,
    prefix embedding concatenation, prefix KV materialization, and denoise
    loop.
    """

    def __init__(
        self,
        *,
        num_layers: int = 18,
        num_views: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        image_channels: int = 3,
        vision_layers: int = 27,
        vision_hidden_size: int = 1152,
        vision_intermediate_size: int = 4304,
        vision_heads: int = 16,
        vocab_size: int = 257152,
        max_prompt_len: int = 200,
        prefix_hidden_size: int = 2048,
        prefix_intermediate_size: int = 16384,
        decoder_hidden_size: int = 1024,
        decoder_intermediate_size: int = 4096,
        action_horizon: int = 50,
        num_steps: int = 10,
        action_dim: int = 32,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        eps: float = 1.0e-6,
        dtype: str = "bfloat16",
        device: str = "cuda",
        use_static_act_scales: bool = False,
    ) -> None:
        super().__init__()
        self.max_prompt_len = int(max_prompt_len)
        self.prefix_hidden_size = int(prefix_hidden_size)
        self.vision = PI05VisionEncoder(
            num_layers=vision_layers,
            num_views=num_views,
            image_size=image_size,
            patch_size=patch_size,
            in_channels=image_channels,
            hidden_size=vision_hidden_size,
            intermediate_size=vision_intermediate_size,
            num_heads=vision_heads,
            output_size=prefix_hidden_size,
            eps=eps,
            dtype=dtype,
            device=device,
            use_static_act_scales=use_static_act_scales,
        )
        self.language = PI05LanguageEmbedding(
            vocab_size=vocab_size,
            hidden_size=prefix_hidden_size,
            dtype=dtype,
            device=device,
        )
        self.prefix_encoder = PI05PaliGemmaPrefixEncoder(
            num_layers=num_layers,
            hidden_size=prefix_hidden_size,
            intermediate_size=prefix_intermediate_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
            use_static_act_scales=use_static_act_scales,
        )
        self.denoise_loop = PI05DenoiseLoop(
            num_layers=num_layers,
            hidden_size=decoder_hidden_size,
            intermediate_size=decoder_intermediate_size,
            action_horizon=action_horizon,
            num_steps=num_steps,
            action_dim=action_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
            use_static_act_scales=use_static_act_scales,
        )

    def forward(
        self,
        noise_f32,
        images_u8,
        token_ids,
        prefix_valid_rows,
        prefix_rope_interleaved,
        suffix_rope_interleaved,
    ):
        image_embs = self.vision(images_u8)
        lang_embs = self.language(token_ids)
        prefix_embs = dp.cat([image_embs, lang_embs], axis=0)
        prefix_k_cache, prefix_v_cache = self.prefix_encoder.materialize_kv(
            prefix_embs,
            prefix_valid_rows,
            prefix_rope_interleaved,
        )
        return self.denoise_loop(
            noise_f32,
            prefix_k_cache,
            prefix_v_cache,
            prefix_valid_rows,
            suffix_rope_interleaved,
        )

    def forward_fast(
        self,
        noise_f32,
        images_u8,
        token_ids,
        prefix_valid_rows,
        prefix_rope_interleaved,
        suffix_rope_interleaved,
    ):
        image_embs = self.vision.forward_fast(images_u8)
        lang_embs = self.language.forward_fast(token_ids)
        image_rows = _static_dim(image_embs, 0)
        lang_rows = _static_dim(lang_embs, 0)
        prefix_rows = image_rows + lang_rows
        prefix_embs = dp.empty((prefix_rows, self.prefix_hidden_size), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_prefix_concat_bf16",
            *[
                image_embs,
                lang_embs,
                image_rows,
                lang_rows,
                self.prefix_hidden_size,
                prefix_embs,
            ],
            launch=_grid_1d(prefix_rows * self.prefix_hidden_size),
        )
        prefix_k_cache, prefix_v_cache = self.prefix_encoder.materialize_kv_fast(
            prefix_embs,
            prefix_valid_rows,
            prefix_rope_interleaved,
        )
        return self.denoise_loop.forward_fast(
            noise_f32,
            prefix_k_cache,
            prefix_v_cache,
            prefix_valid_rows,
            suffix_rope_interleaved,
        )




__all__ = [
    "PI05SampleActionsFromPrefixEmbeddings",
    "PI05SampleActionsFromTokens",
]
