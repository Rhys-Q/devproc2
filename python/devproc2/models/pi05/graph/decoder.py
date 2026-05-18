"""Pi0.5 action decoder layer."""
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
    _quantize_fp8_dynamic_parallel,
    _quantize_fp8_maybe_static,
    _static_dim,
)
from .ffn import PI05FFN
from ..weights import pi05_act_scale_name


class PI05DecoderLayer(nn.Module):
    """Pi0.5 action-expert decoder layer wired through DSL-injected kernels.

    This is the dynamic-activation FP8 path used while calibration scales are
    being collected. It consumes precomputed AdaRMSNorm styles from the weight
    artifact and views the per-layer prefix KV cache without copying.
    """

    def __init__(
        self,
        layer_idx: int,
        *,
        num_layers: int = 18,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_q_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        eps: float = 1.0e-6,
        dtype: str = "bfloat16",
        device: str = "cuda",
        adarms_weight: Parameter | None = None,
        use_static_act_scales: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
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
        self.adarms_weight = adarms_weight or Parameter(
            (hidden_size,),
            dtype,
            device=device,
            role="constant_tensor",
            name="constant.decoder_adarms_weight",
        )
        prefix = f"decoder_{layer_idx}"
        self.qkv_w_fp8 = Parameter(
            (self.qkv_dim, hidden_size),
            "fp8_e4m3",
            device=device,
            name=f"fp8.decoder_attn_qkv_w_{layer_idx}.weight",
        )
        self.qkv_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.decoder_attn_qkv_w_{layer_idx}.scale",
        )
        self.o_w_fp8 = Parameter(
            (hidden_size, self.q_dim),
            "fp8_e4m3",
            device=device,
            name=f"fp8.decoder_attn_o_w_{layer_idx}.weight",
        )
        self.o_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.decoder_attn_o_w_{layer_idx}.scale",
        )
        self.qkv_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("decoder_attn_qkv_w", layer_idx),
        )
        self.o_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("decoder_attn_o_w", layer_idx),
        )
        self.ffn = PI05FFN(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
            gate_up_weight_name=f"fp8.decoder_ffn_gate_up_w_{layer_idx}.weight",
            gate_up_scale_name=f"fp8.decoder_ffn_gate_up_w_{layer_idx}.scale",
            down_weight_name=f"fp8.decoder_ffn_down_w_{layer_idx}.weight",
            down_scale_name=f"fp8.decoder_ffn_down_w_{layer_idx}.scale",
            act0_scale_name=pi05_act_scale_name("decoder_ffn_gate_up_w", layer_idx),
            act1_scale_name=pi05_act_scale_name("decoder_ffn_down_w", layer_idx),
        )
        self._debug_name = prefix

    def forward(
        self,
        hidden,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
        style_attn_table,
        style_ffn_table,
        step,
    ):
        rows = _static_dim(hidden, 0)
        prefix_rows = _static_dim(prefix_k_cache, 1)
        prefix_k = dp.select(prefix_k_cache, axis=0, index=self.layer_idx)
        prefix_v = dp.select(prefix_v_cache, axis=0, index=self.layer_idx)
        style_attn = dp.select(style_attn_table, axis=0, index=step)
        style_attn = dp.select(style_attn, axis=0, index=self.layer_idx)
        style_ffn = dp.select(style_ffn_table, axis=0, index=step)
        style_ffn = dp.select(style_ffn, axis=0, index=self.layer_idx)

        normed = dp.adarms_norm(
            hidden,
            self.adarms_weight,
            style_attn,
            axes=(-1,),
            epsilon=self.eps,
        )
        qkv = _fp8_linear_ref(normed, self.qkv_w_fp8, out_features=self.qkv_dim)
        q, suffix_k, suffix_v = _qkv_views(
            qkv,
            rows,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        full_k = dp.cat([prefix_k, suffix_k], axis=0)
        full_v = dp.cat([prefix_v, suffix_v], axis=0)
        attn = dp.attention(q, full_k, full_v, scale=self.head_dim ** -0.5)
        attn_flat = dp.reshape(attn, (rows, self.q_dim))
        attn_out = _fp8_linear_ref(attn_flat, self.o_w_fp8, out_features=self.hidden_size)
        hidden = dp.add(hidden, attn_out)

        ffn_norm = dp.adarms_norm(
            hidden,
            self.adarms_weight,
            style_ffn,
            axes=(-1,),
            epsilon=self.eps,
        )
        return dp.add(hidden, self.ffn(ffn_norm))

    def forward_fast(
        self,
        hidden,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
        style_attn_table,
        style_ffn_table,
        step,
    ):
        rows = _static_dim(hidden, 0)
        prefix_rows = _static_dim(prefix_k_cache, 1)
        eps_bits = _f32_to_i64_bits(self.eps)
        prefix_k = dp.select(prefix_k_cache, axis=0, index=self.layer_idx)
        prefix_v = dp.select(prefix_v_cache, axis=0, index=self.layer_idx)
        style_attn = dp.select(style_attn_table, axis=0, index=step)
        style_attn = dp.select(style_attn, axis=0, index=self.layer_idx)
        style_ffn = dp.select(style_ffn_table, axis=0, index=step)
        style_ffn = dp.select(style_ffn, axis=0, index=self.layer_idx)

        if self.use_static_act_scales:
            normed_fp8 = dp.empty((rows, self.hidden_size), dtype="fp8_e4m3", device="cuda")
            gate_attn = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
            pi05_ops.call_cuda(
                "pi05_ada_rms_norm_style_to_fp8_bf16",
                *[
                    hidden,
                    self.adarms_weight,
                    style_attn,
                    self.qkv_act_scale,
                    rows,
                    self.hidden_size,
                    eps_bits,
                    normed_fp8,
                    gate_attn,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            )
            normed_scale = self.qkv_act_scale
        else:
            normed = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
            gate_attn = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
            pi05_ops.call_cuda(
                "pi05_ada_rms_norm_style_bf16",
                *[
                    hidden,
                    self.adarms_weight,
                    style_attn,
                    rows,
                    self.hidden_size,
                    eps_bits,
                    normed,
                    gate_attn,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            )
            normed_fp8, normed_scale = _quantize_fp8_dynamic_parallel(
                normed,
                rows * self.hidden_size,
                (rows, self.hidden_size),
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
        q = dp.empty((rows, self.num_q_heads, self.head_dim), dtype="bfloat16", device="cuda")
        full_k = dp.empty(
            (prefix_rows + rows, self.num_kv_heads, self.head_dim),
            dtype="bfloat16",
            device="cuda",
        )
        full_v = dp.empty(
            (prefix_rows + rows, self.num_kv_heads, self.head_dim),
            dtype="bfloat16",
            device="cuda",
        )
        pi05_ops.call_cuda(
            "pi05_qkv_split_rope_concat_bf16",
            *[
                qkv,
                rope_interleaved,
                prefix_k,
                prefix_v,
                prefix_valid_rows,
                rows,
                self.q_dim,
                self.kv_dim,
                self.kv_dim,
                self.head_dim,
                q,
                full_k,
                full_v,
            ],
            launch=_grid_1d(
                max(
                    rows * self.num_q_heads * (self.head_dim // 2),
                    (prefix_rows + rows) * self.kv_dim,
                )
            ),
        )
        attn = pi05_ops.attention_fa2(
            q,
            full_k,
            full_v,
            rows=rows,
            prefix_valid_rows=prefix_valid_rows,
            suffix_rows=rows,
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
        attn_out = pi05_ops.fp8_linear(
            attn_fp8,
            self.o_w_fp8,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.q_dim,
            x_scale=attn_scale,
            weight_scale=self.o_w_scale,
        )
        if self.use_static_act_scales:
            ffn_norm_fp8 = dp.empty((rows, self.hidden_size), dtype="fp8_e4m3", device="cuda")
            gate_ffn = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
            pi05_ops.call_cuda(
                "pi05_gate_residual_ada_norm_to_fp8_bf16",
                *[
                    hidden,
                    attn_out,
                    gate_attn,
                    self.adarms_weight,
                    style_ffn,
                    self.ffn.act0_scale,
                    rows,
                    self.hidden_size,
                    eps_bits,
                    ffn_norm_fp8,
                    gate_ffn,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            )
            ffn_norm_scale = self.ffn.act0_scale
            ffn_out = self.ffn._forward_fast_from_fp8_static(ffn_norm_fp8, ffn_norm_scale, rows)
        else:
            pi05_ops.call_cuda(
                "pi05_gate_mul_residual_bf16",
                hidden,
                attn_out,
                gate_attn,
                rows * self.hidden_size,
                launch=_grid_1d(rows * self.hidden_size),
            )

            ffn_norm = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
            gate_ffn = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
            pi05_ops.call_cuda(
                "pi05_ada_rms_norm_style_bf16",
                *[
                    hidden,
                    self.adarms_weight,
                    style_ffn,
                    rows,
                    self.hidden_size,
                    eps_bits,
                    ffn_norm,
                    gate_ffn,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            )
            ffn_norm_fp8, ffn_norm_scale = _quantize_fp8_dynamic_parallel(
                ffn_norm,
                rows * self.hidden_size,
                (rows, self.hidden_size),
            )
            ffn_out = self.ffn._forward_fast_from_fp8(ffn_norm_fp8, ffn_norm_scale, rows)
        pi05_ops.call_cuda(
            "pi05_gate_mul_residual_bf16",
            hidden,
            ffn_out,
            gate_ffn,
            rows * self.hidden_size,
            launch=_grid_1d(rows * self.hidden_size),
        )
        return hidden






__all__ = ["PI05DecoderLayer"]
