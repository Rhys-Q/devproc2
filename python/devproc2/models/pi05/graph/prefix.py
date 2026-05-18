"""Pi0.5 PaliGemma prefix encoder fragments."""
from __future__ import annotations

import devproc2 as dp
import devproc2.nn as nn

from devproc2.nn.specs import Parameter

from .. import ops as pi05_ops
from ._helpers import (
    _f32_to_i64_bits,
    _fp8_linear_ref,
    _grid_1d,
    _qkv_views,
    _quantize_fp8_maybe_static,
    _rms_norm_unit_fp8_maybe_static,
    _static_dim,
)
from .ffn import PI05FFN
from ..weights import pi05_act_scale_name


class PI05PaliGemmaEncoderLayer(nn.Module):
    """PaliGemma language-model encoder layer for compact prefix tokens."""

    def __init__(
        self,
        layer_idx: int,
        *,
        hidden_size: int = 2048,
        intermediate_size: int = 16384,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        eps: float = 1.0e-6,
        dtype: str = "bfloat16",
        device: str = "cuda",
        use_static_act_scales: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.q_dim = self.num_q_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_dim = self.q_dim + 2 * self.kv_dim
        self.eps = float(eps)
        self.use_static_act_scales = bool(use_static_act_scales)
        self.qkv_w_fp8 = Parameter(
            (self.qkv_dim, hidden_size),
            "fp8_e4m3",
            device=device,
            name=f"fp8.encoder_attn_qkv_w_{layer_idx}.weight",
        )
        self.qkv_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.encoder_attn_qkv_w_{layer_idx}.scale",
        )
        self.o_w_fp8 = Parameter(
            (hidden_size, self.q_dim),
            "fp8_e4m3",
            device=device,
            name=f"fp8.encoder_attn_o_w_{layer_idx}.weight",
        )
        self.o_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.encoder_attn_o_w_{layer_idx}.scale",
        )
        self.qkv_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("encoder_attn_qkv_w", layer_idx),
        )
        self.o_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("encoder_attn_o_w", layer_idx),
        )
        self.ffn = PI05FFN(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
            gate_up_weight_name=f"fp8.encoder_ffn_gate_up_w_{layer_idx}.weight",
            gate_up_scale_name=f"fp8.encoder_ffn_gate_up_w_{layer_idx}.scale",
            down_weight_name=f"fp8.encoder_ffn_down_w_{layer_idx}.weight",
            down_scale_name=f"fp8.encoder_ffn_down_w_{layer_idx}.scale",
            act0_scale_name=pi05_act_scale_name("encoder_ffn_gate_up_w", layer_idx),
            act1_scale_name=pi05_act_scale_name("encoder_ffn_down_w", layer_idx),
        )

    def forward(
        self,
        hidden,
        rope_interleaved,
        k_cache=None,
        v_cache=None,
        prefix_valid_rows=None,
        skip_post_kv: bool = False,
    ):
        rows = _static_dim(hidden, 0)
        normed = dp.rms_norm_unit(hidden, epsilon=self.eps)
        qkv = _fp8_linear_ref(normed, self.qkv_w_fp8, out_features=self.qkv_dim)
        q, k, v = _qkv_views(qkv, rows, self.num_q_heads, self.num_kv_heads, self.head_dim)
        if k_cache is not None and v_cache is not None:
            prefix_rows = _static_dim(k_cache, 1)
            k = dp.select(k_cache, axis=0, index=self.layer_idx)
            v = dp.select(v_cache, axis=0, index=self.layer_idx)
            if skip_post_kv:
                return hidden
        attn = dp.attention(q, k, v, scale=self.head_dim ** -0.5)
        attn_flat = dp.reshape(attn, (rows, self.q_dim))
        attn_out = _fp8_linear_ref(attn_flat, self.o_w_fp8, out_features=self.hidden_size)
        hidden = dp.add(hidden, attn_out)
        ffn_norm = dp.rms_norm_unit(hidden, epsilon=self.eps)
        return dp.add(hidden, self.ffn(ffn_norm))

    def forward_with_kv(self, hidden, rope_interleaved, *, skip_post_kv: bool = False):
        rows = _static_dim(hidden, 0)
        normed = dp.rms_norm_unit(hidden, epsilon=self.eps)
        qkv = _fp8_linear_ref(normed, self.qkv_w_fp8, out_features=self.qkv_dim)
        q, k, v = _qkv_views(qkv, rows, self.num_q_heads, self.num_kv_heads, self.head_dim)
        if skip_post_kv:
            return hidden, k, v
        attn = dp.attention(q, k, v, scale=self.head_dim ** -0.5)
        attn_flat = dp.reshape(attn, (rows, self.q_dim))
        attn_out = _fp8_linear_ref(attn_flat, self.o_w_fp8, out_features=self.hidden_size)
        hidden = dp.add(hidden, attn_out)
        ffn_norm = dp.rms_norm_unit(hidden, epsilon=self.eps)
        return dp.add(hidden, self.ffn(ffn_norm)), k, v

    def forward_fast(
        self,
        hidden,
        rope_interleaved,
        k_cache=None,
        v_cache=None,
        prefix_valid_rows=None,
        skip_post_kv: bool = False,
    ):
        rows = _static_dim(hidden, 0)
        eps_bits = _f32_to_i64_bits(self.eps)
        normed_fp8, normed_scale = _rms_norm_unit_fp8_maybe_static(
            hidden,
            rows,
            self.hidden_size,
            eps_bits,
            self.qkv_act_scale if self.use_static_act_scales else None,
        )
        qkv = pi05_ops.fp8_linear(
            normed_fp8,
            self.qkv_w_fp8,
            rows=rows,
            out_features=self.qkv_dim,
            in_features=self.hidden_size,
            x_scale=normed_scale,
            weight_scale=self.qkv_w_scale,
        )
        prefix_rows = rows
        if k_cache is not None:
            prefix_rows = _static_dim(k_cache, 1)
        if k_cache is not None and v_cache is not None:
            q = dp.empty((rows, self.num_q_heads, self.head_dim), dtype="bfloat16", device="cuda")
            pi05_ops.call_cuda(
                "pi05_qkv_split_rope_cache_bf16",
                *[
                    qkv,
                    rope_interleaved,
                    k_cache,
                    v_cache,
                    self.layer_idx,
                    rows,
                    prefix_rows,
                    self.q_dim,
                    self.kv_dim,
                    self.kv_dim,
                    self.head_dim,
                    q,
                ],
                launch=_grid_1d(
                    max(rows * self.num_q_heads * (self.head_dim // 2), rows * self.kv_dim)
                ),
            )
            k = dp.select(k_cache, axis=0, index=self.layer_idx)
            v = dp.select(v_cache, axis=0, index=self.layer_idx)
            if skip_post_kv:
                return hidden
        else:
            q = dp.empty((rows, self.num_q_heads, self.head_dim), dtype="bfloat16", device="cuda")
            k = dp.empty((rows, self.num_kv_heads, self.head_dim), dtype="bfloat16", device="cuda")
            v = dp.empty((rows, self.num_kv_heads, self.head_dim), dtype="bfloat16", device="cuda")
            pi05_ops.call_cuda(
                "pi05_qkv_split_rope_bf16",
                *[
                    qkv,
                    rope_interleaved,
                    rows,
                    self.q_dim,
                    self.kv_dim,
                    self.kv_dim,
                    self.head_dim,
                    q,
                    k,
                    v,
                ],
                launch=_grid_1d(
                    max(rows * self.num_q_heads * (self.head_dim // 2), rows * self.kv_dim)
                ),
            )
        if prefix_valid_rows is None:
            attn = dp.empty((rows, self.num_q_heads, self.head_dim), dtype="bfloat16", device="cuda")
            launch = dp.KernelLaunchSpec(
                grid=(rows, self.num_q_heads, 1),
                block=(256, 1, 1),
                shared_memory_bytes=rows * 4,
            )
            pi05_ops.call_cuda(
                "pi05_attention_bf16",
                *[
                    q,
                    k,
                    v,
                    rows,
                    rows,
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    _f32_to_i64_bits(self.head_dim ** -0.5),
                    attn,
                ],
                launch=launch,
            )
        else:
            attn = pi05_ops.attention_fa2(
                q,
                k,
                v,
                rows=rows,
                prefix_valid_rows=prefix_valid_rows,
                suffix_rows=0,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                scale_bits=_f32_to_i64_bits(self.head_dim ** -0.5),
            )
        attn_flat = dp.reshape(attn, (rows, self.q_dim))
        attn_fp8, attn_scale = _quantize_fp8_maybe_static(
            attn_flat,
            rows * self.q_dim,
            (rows, self.q_dim),
            self.o_act_scale if self.use_static_act_scales else None,
        )
        pi05_ops.fp8_linear_accum_(
            attn_fp8,
            self.o_w_fp8,
            residual=hidden,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.q_dim,
            x_scale=attn_scale,
            weight_scale=self.o_w_scale,
        )

        ffn_norm_fp8, ffn_norm_scale = _rms_norm_unit_fp8_maybe_static(
            hidden,
            rows,
            self.hidden_size,
            eps_bits,
            self.ffn.act0_scale if self.use_static_act_scales else None,
        )
        if self.use_static_act_scales:
            self.ffn._forward_fast_from_fp8_static_accum(ffn_norm_fp8, ffn_norm_scale, rows, hidden)
        else:
            self.ffn._forward_fast_from_fp8_accum(ffn_norm_fp8, ffn_norm_scale, rows, hidden)
        return hidden




class PI05PaliGemmaPrefixEncoder(nn.Module):
    """Compact PaliGemma prefix transformer slice.

    This executable slice validates the encoder block wiring and can either
    return transformed prefix hidden states or materialize the per-layer KV
    cache tuple used by the action decoder.
    """

    def __init__(
        self,
        *,
        num_layers: int = 18,
        hidden_size: int = 2048,
        intermediate_size: int = 16384,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        eps: float = 1.0e-6,
        dtype: str = "bfloat16",
        device: str = "cuda",
        use_static_act_scales: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.device = device
        self.use_static_act_scales = bool(use_static_act_scales)
        self.layers = nn.ModuleList(
            PI05PaliGemmaEncoderLayer(
                i,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                dtype=dtype,
                device=device,
                use_static_act_scales=use_static_act_scales,
            )
            for i in range(num_layers)
        )

    def forward(self, prefix_embs, rope_interleaved):
        hidden = prefix_embs
        for layer in self.layers:
            hidden = layer(hidden, rope_interleaved)
        return hidden

    def materialize_kv(self, prefix_embs, prefix_valid_rows, rope_interleaved):
        rows = _static_dim(prefix_embs, 0)
        hidden = prefix_embs
        k_layers = []
        v_layers = []
        for i, layer in enumerate(self.layers):
            hidden, k, v = layer.forward_with_kv(
                hidden,
                rope_interleaved,
                skip_post_kv=(i == self.num_layers - 1),
            )
            k_layers.append(dp.reshape(k, (1, rows, self.num_kv_heads, self.head_dim)))
            v_layers.append(dp.reshape(v, (1, rows, self.num_kv_heads, self.head_dim)))
        return dp.cat(k_layers, axis=0), dp.cat(v_layers, axis=0)

    def forward_fast(self, prefix_embs, rope_interleaved):
        hidden = prefix_embs
        for layer in self.layers:
            hidden = layer.forward_fast(hidden, rope_interleaved)
        return hidden

    def materialize_kv_fast(self, prefix_embs, prefix_valid_rows, rope_interleaved):
        rows = _static_dim(prefix_embs, 0)
        k_cache = dp.empty(
            (self.num_layers, rows, self.num_kv_heads, self.head_dim),
            dtype="bfloat16",
            device=self.device,
        )
        v_cache = dp.empty(
            (self.num_layers, rows, self.num_kv_heads, self.head_dim),
            dtype="bfloat16",
            device=self.device,
        )
        hidden = prefix_embs
        for i, layer in enumerate(self.layers):
            hidden = layer.forward_fast(
                hidden,
                rope_interleaved,
                k_cache,
                v_cache,
                prefix_valid_rows,
                skip_post_kv=(i == self.num_layers - 1),
            )
        return k_cache, v_cache




__all__ = [
    "PI05PaliGemmaEncoderLayer",
    "PI05PaliGemmaPrefixEncoder",
]
