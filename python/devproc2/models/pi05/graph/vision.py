"""Pi0.5 SigLIP vision encoder fragments."""
from __future__ import annotations

import devproc2 as dp
import devproc2.nn as nn

from devproc2.nn.specs import Parameter

from .. import ops as pi05_ops
from ._helpers import (
    _add_bias_if_present,
    _f32_to_i64_bits,
    _fp8_linear_ref,
    _grid_1d,
    _layer_norm_fp8_maybe_static,
    _qkv_views,
    _quantize_fp8_maybe_static,
    _static_dim,
    _view_bf16_row,
)
from ..weights import pi05_act_scale_name


class PI05VisionPatchEmbedding(nn.Module):
    """SigLIP patch embedding front slice for the Pi0.5 prefix path.

    ``forward`` keeps the readable projection form over already-im2col patches.
    ``forward_fast`` owns the deploy path from uint8 NHWC images through CUDA
    normalization, patch im2col, BF16 GEMM, bias and position add.
    """

    def __init__(
        self,
        *,
        num_views: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        vision_width: int = 1152,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_views = int(num_views)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.vision_width = int(vision_width)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = self.patch_size * self.patch_size * self.in_channels
        self.patch_weight = Parameter(
            (self.patch_size, self.patch_size, self.in_channels, self.vision_width),
            dtype,
            device=device,
            name="vision_patch_embedding_w",
        )
        self.patch_bias = Parameter(
            (self.vision_width,),
            dtype,
            device=device,
            name="vision_patch_embedding_b",
        )
        self.position_embedding = Parameter(
            (self.num_patches, self.vision_width),
            dtype,
            device=device,
            name="vision_position_embedding",
        )

    def forward(self, patches):
        rows = _static_dim(patches, 0)
        weight = dp.reshape(self.patch_weight, (self.patch_dim, self.vision_width))
        out = dp.matmul(patches, weight)
        out = dp.add(out, self.patch_bias)
        if rows == self.num_patches:
            return dp.add(out, self.position_embedding)
        if rows == self.num_views * self.num_patches:
            tiled_position = dp.cat(
                [self.position_embedding for _ in range(self.num_views)],
                axis=0,
            )
            return dp.add(out, tiled_position)
        return out

    def forward_images(self, images_u8):
        rows = self.num_views * self.num_patches
        image_bf16 = dp.cast(images_u8, dtype="bfloat16")
        patches = dp.image_patch_im2col(
            image_bf16,
            shape=(rows, self.patch_dim),
            patch_size=self.patch_size,
            dtype="bfloat16",
        )
        return self.forward(patches)

    def forward_fast(self, images_u8):
        image_elems = self.num_views * self.image_size * self.image_size * self.in_channels
        rows = self.num_views * self.num_patches
        image_bf16 = dp.empty(
            (self.num_views, self.image_size, self.image_size, self.in_channels),
            dtype="bfloat16",
            device="cuda",
        )
        pi05_ops.call_cuda(
            "pi05_image_u8_to_bf16_norm",
            images_u8,
            image_elems,
            image_bf16,
            launch=_grid_1d(image_elems),
        )
        patches = dp.empty((rows, self.patch_dim), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_patch_im2col_bf16",
            image_bf16,
            self.num_views,
            patches,
            launch=_grid_1d(rows * self.patch_dim),
        )
        out = pi05_ops.bf16_linear(
            patches,
            self.patch_weight,
            rows=rows,
            out_features=self.vision_width,
            in_features=self.patch_dim,
        )
        pi05_ops.call_cuda(
            "pi05_bias_add_bf16",
            out,
            self.patch_bias,
            rows,
            self.vision_width,
            launch=_grid_1d(rows * self.vision_width),
        )
        pi05_ops.call_cuda(
            "pi05_position_add_bf16",
            out,
            self.position_embedding,
            rows,
            self.num_patches,
            self.vision_width,
            launch=_grid_1d(rows * self.vision_width),
        )
        return out




class PI05VisionEncoderLayer(nn.Module):
    """SigLIP vision encoder layer for the Pi0.5 prefix path."""

    def __init__(
        self,
        layer_idx: int,
        *,
        num_layers: int = 27,
        num_views: int = 1,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_heads: int = 16,
        eps: float = 1.0e-6,
        dtype: str = "bfloat16",
        device: str = "cuda",
        use_static_act_scales: bool = False,
        pre_attn_norm_w: Parameter | None = None,
        pre_attn_norm_b: Parameter | None = None,
        pre_ffn_norm_w: Parameter | None = None,
        pre_ffn_norm_b: Parameter | None = None,
        attn_qkv_b: Parameter | None = None,
        attn_o_b: Parameter | None = None,
        ffn_up_b: Parameter | None = None,
        ffn_down_b: Parameter | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
        self.num_views = int(num_views)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_heads = int(num_heads)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_dim = self.hidden_size // self.num_heads
        self.qkv_dim = 3 * self.hidden_size
        self.eps = float(eps)
        self.use_static_act_scales = bool(use_static_act_scales)
        self.pre_attn_norm_w = pre_attn_norm_w or Parameter(
            (num_layers, hidden_size),
            dtype,
            device=device,
            name="vision_pre_attn_norm_w",
        )
        self.pre_attn_norm_b = pre_attn_norm_b or Parameter(
            (num_layers, hidden_size),
            dtype,
            device=device,
            name="vision_pre_attn_norm_b",
        )
        self.pre_ffn_norm_w = pre_ffn_norm_w or Parameter(
            (num_layers, hidden_size),
            dtype,
            device=device,
            name="vision_pre_ffn_norm_w",
        )
        self.pre_ffn_norm_b = pre_ffn_norm_b or Parameter(
            (num_layers, hidden_size),
            dtype,
            device=device,
            name="vision_pre_ffn_norm_b",
        )
        self.attn_qkv_b = attn_qkv_b or Parameter(
            (num_layers, self.qkv_dim),
            dtype,
            device=device,
            name="vision_attn_qkv_b",
        )
        self.attn_o_b = attn_o_b or Parameter(
            (num_layers, hidden_size),
            dtype,
            device=device,
            name="vision_attn_o_b",
        )
        self.ffn_up_b = ffn_up_b or Parameter(
            (num_layers, intermediate_size),
            dtype,
            device=device,
            name="vision_ffn_up_b",
        )
        self.ffn_down_b = ffn_down_b or Parameter(
            (num_layers, hidden_size),
            dtype,
            device=device,
            name="vision_ffn_down_b",
        )
        self.qkv_w_fp8 = Parameter(
            (self.qkv_dim, hidden_size),
            "fp8_e4m3",
            device=device,
            name=f"fp8.vision_attn_qkv_w_{layer_idx}.weight",
        )
        self.qkv_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.vision_attn_qkv_w_{layer_idx}.scale",
        )
        self.o_w_fp8 = Parameter(
            (hidden_size, hidden_size),
            "fp8_e4m3",
            device=device,
            name=f"fp8.vision_attn_o_w_{layer_idx}.weight",
        )
        self.o_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.vision_attn_o_w_{layer_idx}.scale",
        )
        self.up_w_fp8 = Parameter(
            (intermediate_size, hidden_size),
            "fp8_e4m3",
            device=device,
            name=f"fp8.vision_ffn_up_w_{layer_idx}.weight",
        )
        self.up_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.vision_ffn_up_w_{layer_idx}.scale",
        )
        self.down_w_fp8 = Parameter(
            (hidden_size, intermediate_size),
            "fp8_e4m3",
            device=device,
            name=f"fp8.vision_ffn_down_w_{layer_idx}.weight",
        )
        self.down_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"fp8.vision_ffn_down_w_{layer_idx}.scale",
        )
        self.qkv_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("vision_attn_qkv_w", layer_idx),
        )
        self.o_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("vision_attn_o_w", layer_idx),
        )
        self.up_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("vision_ffn_up_w", layer_idx),
        )
        self.down_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("vision_ffn_down_w", layer_idx),
        )

    def forward(self, hidden):
        rows = _static_dim(hidden, 0)
        norm_w = _view_bf16_row(self.pre_attn_norm_w, self.layer_idx, self.hidden_size)
        norm_b = _view_bf16_row(self.pre_attn_norm_b, self.layer_idx, self.hidden_size)
        qkv_b = _view_bf16_row(self.attn_qkv_b, self.layer_idx, self.qkv_dim)
        o_b = _view_bf16_row(self.attn_o_b, self.layer_idx, self.hidden_size)
        ffn_norm_w = _view_bf16_row(self.pre_ffn_norm_w, self.layer_idx, self.hidden_size)
        ffn_norm_b = _view_bf16_row(self.pre_ffn_norm_b, self.layer_idx, self.hidden_size)
        up_b = _view_bf16_row(self.ffn_up_b, self.layer_idx, self.intermediate_size)
        down_b = _view_bf16_row(self.ffn_down_b, self.layer_idx, self.hidden_size)

        attn_norm = dp.layer_norm(
            hidden,
            norm_w,
            norm_b,
            axes=(-1,),
            epsilon=self.eps,
        )
        qkv = _add_bias_if_present(
            _fp8_linear_ref(attn_norm, self.qkv_w_fp8, out_features=self.qkv_dim),
            qkv_b,
        )
        q, k, v = _qkv_views(qkv, rows, self.num_heads, self.num_heads, self.head_dim)
        attn = dp.attention(q, k, v, scale=self.head_dim ** -0.5)
        attn_flat = dp.reshape(attn, (rows, self.hidden_size))
        attn_out = _add_bias_if_present(
            _fp8_linear_ref(attn_flat, self.o_w_fp8, out_features=self.hidden_size),
            o_b,
        )
        hidden = dp.add(hidden, attn_out)

        ffn_norm = dp.layer_norm(
            hidden,
            ffn_norm_w,
            ffn_norm_b,
            axes=(-1,),
            epsilon=self.eps,
        )
        ffn_hidden = _add_bias_if_present(
            _fp8_linear_ref(ffn_norm, self.up_w_fp8, out_features=self.intermediate_size),
            up_b,
        )
        ffn_hidden = dp.gelu(ffn_hidden, approximate="tanh")
        ffn_out = _add_bias_if_present(
            _fp8_linear_ref(ffn_hidden, self.down_w_fp8, out_features=self.hidden_size),
            down_b,
        )
        return dp.add(hidden, ffn_out)

    def forward_fast(self, hidden):
        rows = _static_dim(hidden, 0)
        eps_bits = _f32_to_i64_bits(self.eps)
        norm_w = _view_bf16_row(self.pre_attn_norm_w, self.layer_idx, self.hidden_size)
        norm_b = _view_bf16_row(self.pre_attn_norm_b, self.layer_idx, self.hidden_size)
        qkv_b = _view_bf16_row(self.attn_qkv_b, self.layer_idx, self.qkv_dim)
        o_b = _view_bf16_row(self.attn_o_b, self.layer_idx, self.hidden_size)
        ffn_norm_w = _view_bf16_row(self.pre_ffn_norm_w, self.layer_idx, self.hidden_size)
        ffn_norm_b = _view_bf16_row(self.pre_ffn_norm_b, self.layer_idx, self.hidden_size)
        up_b = _view_bf16_row(self.ffn_up_b, self.layer_idx, self.intermediate_size)
        down_b = _view_bf16_row(self.ffn_down_b, self.layer_idx, self.hidden_size)

        attn_norm_fp8, attn_norm_scale = _layer_norm_fp8_maybe_static(
            hidden,
            norm_w,
            norm_b,
            rows,
            self.hidden_size,
            eps_bits,
            self.qkv_act_scale if self.use_static_act_scales else None,
        )
        qkv = pi05_ops.fp8_linear(
            attn_norm_fp8,
            self.qkv_w_fp8,
            rows=rows,
            out_features=self.qkv_dim,
            in_features=self.hidden_size,
            x_scale=attn_norm_scale,
            weight_scale=self.qkv_w_scale,
        )
        q = dp.empty((rows, self.num_heads, self.head_dim), dtype="bfloat16", device="cuda")
        k = dp.empty((rows, self.num_heads, self.head_dim), dtype="bfloat16", device="cuda")
        v = dp.empty((rows, self.num_heads, self.head_dim), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_qkv_bias_split_bf16",
            *[
                qkv,
                qkv_b,
                rows,
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
                q,
                k,
                v,
            ],
            launch=_grid_1d(rows * self.qkv_dim),
        )
        if rows % self.num_views != 0:
            raise ValueError("vision hidden rows must be divisible by num_views")
        rows_per_view = rows // self.num_views
        attn = pi05_ops.attention_fa2_batched(
            q,
            k,
            v,
            batches=self.num_views,
            query_rows=rows_per_view,
            key_rows=rows_per_view,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            scale_bits=_f32_to_i64_bits(self.head_dim ** -0.5),
        )
        attn_flat = dp.reshape(attn, (rows, self.hidden_size))
        attn_fp8, attn_scale = _quantize_fp8_maybe_static(
            attn_flat,
            rows * self.hidden_size,
            (rows, self.hidden_size),
            self.o_act_scale if self.use_static_act_scales else None,
        )
        attn_out = pi05_ops.fp8_linear(
            attn_fp8,
            self.o_w_fp8,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.hidden_size,
            x_scale=attn_scale,
            weight_scale=self.o_w_scale,
        )
        pi05_ops.call_cuda(
            "pi05_bias_residual_bf16",
            hidden,
            attn_out,
            o_b,
            rows,
            self.hidden_size,
            launch=_grid_1d(rows * self.hidden_size),
        )

        ffn_norm_fp8, ffn_norm_scale = _layer_norm_fp8_maybe_static(
            hidden,
            ffn_norm_w,
            ffn_norm_b,
            rows,
            self.hidden_size,
            eps_bits,
            self.up_act_scale if self.use_static_act_scales else None,
        )
        ffn_hidden = pi05_ops.fp8_linear(
            ffn_norm_fp8,
            self.up_w_fp8,
            rows=rows,
            out_features=self.intermediate_size,
            in_features=self.hidden_size,
            x_scale=ffn_norm_scale,
            weight_scale=self.up_w_scale,
        )
        if self.use_static_act_scales:
            ffn_hidden_fp8 = dp.empty(
                (rows, self.intermediate_size),
                dtype="fp8_e4m3",
                device="cuda",
            )
            pi05_ops.call_cuda(
                "pi05_bias_gelu_to_fp8_bf16",
                ffn_hidden,
                up_b,
                self.down_act_scale,
                rows,
                self.intermediate_size,
                ffn_hidden_fp8,
                launch=_grid_1d(rows * self.intermediate_size),
            )
            ffn_hidden_scale = self.down_act_scale
        else:
            pi05_ops.call_cuda(
                "pi05_bias_add_bf16",
                ffn_hidden,
                up_b,
                rows,
                self.intermediate_size,
                launch=_grid_1d(rows * self.intermediate_size),
            )
            pi05_ops.call_cuda(
                "pi05_gelu_inplace_bf16",
                ffn_hidden,
                rows * self.intermediate_size,
                launch=_grid_1d(rows * self.intermediate_size),
            )
            ffn_hidden_fp8, ffn_hidden_scale = _quantize_fp8_maybe_static(
                ffn_hidden,
                rows * self.intermediate_size,
                (rows, self.intermediate_size),
                None,
            )
        ffn_out = pi05_ops.fp8_linear(
            ffn_hidden_fp8,
            self.down_w_fp8,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.intermediate_size,
            x_scale=ffn_hidden_scale,
            weight_scale=self.down_w_scale,
        )
        pi05_ops.call_cuda(
            "pi05_bias_residual_bf16",
            hidden,
            ffn_out,
            down_b,
            rows,
            self.hidden_size,
            launch=_grid_1d(rows * self.hidden_size),
        )
        return hidden




class PI05VisionEncoder(nn.Module):
    """SigLIP vision tower fast path through final multimodal projection."""

    def __init__(
        self,
        *,
        num_layers: int = 27,
        num_views: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_heads: int = 16,
        output_size: int = 2048,
        eps: float = 1.0e-6,
        dtype: str = "bfloat16",
        device: str = "cuda",
        use_static_act_scales: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_views = int(num_views)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.eps = float(eps)
        self.use_static_act_scales = bool(use_static_act_scales)
        self.patch = PI05VisionPatchEmbedding(
            num_views=num_views,
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            vision_width=hidden_size,
            dtype=dtype,
            device=device,
        )
        self.pre_attn_norm_w = Parameter((num_layers, hidden_size), dtype, device=device, name="vision_pre_attn_norm_w")
        self.pre_attn_norm_b = Parameter((num_layers, hidden_size), dtype, device=device, name="vision_pre_attn_norm_b")
        self.pre_ffn_norm_w = Parameter((num_layers, hidden_size), dtype, device=device, name="vision_pre_ffn_norm_w")
        self.pre_ffn_norm_b = Parameter((num_layers, hidden_size), dtype, device=device, name="vision_pre_ffn_norm_b")
        self.attn_qkv_b = Parameter((num_layers, 3 * hidden_size), dtype, device=device, name="vision_attn_qkv_b")
        self.attn_o_b = Parameter((num_layers, hidden_size), dtype, device=device, name="vision_attn_o_b")
        self.ffn_up_b = Parameter((num_layers, intermediate_size), dtype, device=device, name="vision_ffn_up_b")
        self.ffn_down_b = Parameter((num_layers, hidden_size), dtype, device=device, name="vision_ffn_down_b")
        self.layers = nn.ModuleList(
            PI05VisionEncoderLayer(
                i,
                num_layers=num_layers,
                num_views=num_views,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                eps=eps,
                dtype=dtype,
                device=device,
                use_static_act_scales=use_static_act_scales,
                pre_attn_norm_w=self.pre_attn_norm_w,
                pre_attn_norm_b=self.pre_attn_norm_b,
                pre_ffn_norm_w=self.pre_ffn_norm_w,
                pre_ffn_norm_b=self.pre_ffn_norm_b,
                attn_qkv_b=self.attn_qkv_b,
                attn_o_b=self.attn_o_b,
                ffn_up_b=self.ffn_up_b,
                ffn_down_b=self.ffn_down_b,
            )
            for i in range(num_layers)
        )
        self.final_norm_w = Parameter((hidden_size,), dtype, device=device, name="vision_final_norm_w")
        self.final_norm_b = Parameter((hidden_size,), dtype, device=device, name="vision_final_norm_b")
        self.projector_w_fp8 = Parameter(
            (output_size, hidden_size),
            "fp8_e4m3",
            device=device,
            name="fp8.vision_projector_w.weight",
        )
        self.projector_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name="fp8.vision_projector_w.scale",
        )
        self.projector_b = Parameter((output_size,), dtype, device=device, name="encoder_multi_modal_projector_b")
        self.projector_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=pi05_act_scale_name("vision_projector_w"),
        )

    def forward(self, images_u8):
        hidden = self.patch.forward_images(images_u8)
        rows = _static_dim(hidden, 0)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = dp.layer_norm(
            hidden,
            self.final_norm_w,
            self.final_norm_b,
            axes=(-1,),
            epsilon=self.eps,
        )
        out = _fp8_linear_ref(hidden, self.projector_w_fp8, out_features=self.output_size)
        return dp.add(out, self.projector_b)

    def forward_fast(self, images_u8):
        hidden = self.patch.forward_fast(images_u8)
        rows = _static_dim(hidden, 0)
        for layer in self.layers:
            hidden = layer.forward_fast(hidden)
        hidden_fp8, hidden_scale = _layer_norm_fp8_maybe_static(
            hidden,
            self.final_norm_w,
            self.final_norm_b,
            rows,
            self.hidden_size,
            _f32_to_i64_bits(self.eps),
            self.projector_act_scale if self.use_static_act_scales else None,
        )
        out = pi05_ops.fp8_linear(
            hidden_fp8,
            self.projector_w_fp8,
            rows=rows,
            out_features=self.output_size,
            in_features=self.hidden_size,
            x_scale=hidden_scale,
            weight_scale=self.projector_w_scale,
        )
        pi05_ops.call_cuda(
            "pi05_bias_add_bf16",
            out,
            self.projector_b,
            rows,
            self.output_size,
            launch=_grid_1d(rows * self.output_size),
        )
        return out




__all__ = [
    "PI05VisionEncoder",
    "PI05VisionEncoderLayer",
    "PI05VisionPatchEmbedding",
]
