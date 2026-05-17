"""Pi0.5 nn.Module fragments with standard and fast CUDA paths."""
from __future__ import annotations

import struct

import devproc2 as dp
import devproc2.nn as nn

from devproc2.ir.prim_expr import PrimExpr, ceildiv
from devproc2.nn.specs import Parameter
from devproc2.pi05.kernels import register_pi05_kernels


class PI05Linear(nn.Module):
    """Pi0.5 row-major [K, N] linear projection.

    This matches the converted FlashRT weight layout. The standard forward
    remains a readable matmul/add IR sequence; forward_fast selects the CUDA
    cuBLASLt BF16 packed func.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype: str = "bfloat16",
        device: str = "cuda",
        weight_name: str | None = None,
        bias_name: str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter((in_features, out_features), dtype, device=device, name=weight_name)
        self.bias = (
            Parameter((out_features,), dtype, device=device, name=bias_name)
            if bias
            else None
        )

    def forward(self, x):
        out = dp.matmul(x, self.weight)
        if self.bias is not None:
            out = dp.add(out, self.bias)
        return out

    def forward_fast(self, x):
        register_pi05_kernels(sm_arch=89)
        rows = _static_dim(x, 0)
        out = dp.call_dps_packed(
            "runtime.cuda.bf16_nn_bf16",
            inputs=[
                x,
                self.weight,
                rows,
                self.out_features,
                self.in_features,
            ],
            output_shape=(rows, self.out_features),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        if self.bias is not None:
            dp.call_dps_kernel(
                "pi05_bias_add_bf16",
                inputs=[out, self.bias, rows, self.out_features],
                launch=_grid_1d(rows * self.out_features),
            )
        return out


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

    def forward_fast(self, images_u8):
        register_pi05_kernels(sm_arch=89)
        image_elems = self.num_views * self.image_size * self.image_size * self.in_channels
        rows = self.num_views * self.num_patches
        image_bf16 = dp.call_dps_kernel(
            "pi05_image_u8_to_bf16_norm",
            inputs=[images_u8, image_elems],
            launch=_grid_1d(image_elems),
            output_shape=(self.num_views, self.image_size, self.image_size, self.in_channels),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        patches = dp.call_dps_kernel(
            "pi05_patch_im2col_bf16",
            inputs=[image_bf16, self.num_views],
            launch=_grid_1d(rows * self.patch_dim),
            output_shape=(rows, self.patch_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        out = dp.call_dps_packed(
            "runtime.cuda.bf16_nn_bf16",
            inputs=[
                patches,
                self.patch_weight,
                rows,
                self.vision_width,
                self.patch_dim,
            ],
            output_shape=(rows, self.vision_width),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        dp.call_dps_kernel(
            "pi05_bias_add_bf16",
            inputs=[out, self.patch_bias, rows, self.vision_width],
            launch=_grid_1d(rows * self.vision_width),
        )
        dp.call_dps_kernel(
            "pi05_position_add_bf16",
            inputs=[out, self.position_embedding, rows, self.num_patches, self.vision_width],
            launch=_grid_1d(rows * self.vision_width),
        )
        return out


class PI05LanguageEmbedding(nn.Module):
    """PaliGemma language token embedding used by the Pi0.5 prefix path."""

    def __init__(
        self,
        *,
        vocab_size: int = 257152,
        hidden_size: int = 2048,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.embedding = Parameter(
            (self.vocab_size, self.hidden_size),
            dtype,
            device=device,
            name="embedding_weight",
        )

    def forward(self, token_ids):
        emb = dp.embedding(token_ids, self.embedding)
        return dp.multiply(emb, self.hidden_size ** 0.5)

    def forward_fast(self, token_ids):
        register_pi05_kernels(sm_arch=89)
        num_tokens = _static_dim(token_ids, 0)
        return dp.call_dps_kernel(
            "pi05_embedding_gather_bf16",
            inputs=[token_ids, self.embedding, num_tokens, self.hidden_size],
            launch=_grid_1d(num_tokens * self.hidden_size),
            output_shape=(num_tokens, self.hidden_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )


class PI05Attention(nn.Module):
    """Pi0.5 BF16 attention correctness fallback.

    The performance target path should replace this with FA2 through the same
    DPS boundary. This module gives frontend DSL/VM a concrete attention call
    while the vendored FA2 packed func is integrated.
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.scale = float(scale if scale is not None else self.head_dim ** -0.5)

    def forward_fast(self, q, k, v):
        register_pi05_kernels(sm_arch=89)
        rows_q = _static_dim(q, 0)
        rows_k = _static_dim(k, 0)
        return dp.call_dps_kernel(
            "pi05_attention_bf16",
            inputs=[
                q,
                k,
                v,
                rows_q,
                rows_k,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                _f32_to_i64_bits(self.scale),
            ],
            launch=dp.KernelLaunchSpec(
                grid=(rows_q, self.num_q_heads, 1),
                block=(256, 1, 1),
                shared_memory_bytes=rows_k * 4,
            ),
            output_shape=(rows_q, self.num_q_heads, self.head_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )


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
            name=f"act_scale.vision_attn_qkv_w_{layer_idx}",
        )
        self.o_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"act_scale.vision_attn_o_w_{layer_idx}",
        )
        self.up_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"act_scale.vision_ffn_up_w_{layer_idx}",
        )
        self.down_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"act_scale.vision_ffn_down_w_{layer_idx}",
        )

    def forward_fast_dynamic(self, hidden):
        register_pi05_kernels(sm_arch=89)
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
        qkv = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                attn_norm_fp8,
                self.qkv_w_fp8,
                rows,
                self.qkv_dim,
                self.hidden_size,
                attn_norm_scale,
                self.qkv_w_scale,
            ],
            output_shape=(rows, self.qkv_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        q, k, v = dp.call_dps_kernel(
            "pi05_qkv_bias_split_bf16",
            inputs=[
                qkv,
                qkv_b,
                rows,
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
            ],
            launch=_grid_1d(rows * self.qkv_dim),
            output_specs=[
                nn.TensorSpec((rows, self.num_heads, self.head_dim), "bfloat16"),
                nn.TensorSpec((rows, self.num_heads, self.head_dim), "bfloat16"),
                nn.TensorSpec((rows, self.num_heads, self.head_dim), "bfloat16"),
            ],
        )
        if rows % self.num_views != 0:
            raise ValueError("vision hidden rows must be divisible by num_views")
        rows_per_view = rows // self.num_views
        attn = dp.call_dps_packed(
            "runtime.cuda.pi05_fa2_bf16_batched",
            inputs=[
                q,
                k,
                v,
                self.num_views,
                rows_per_view,
                rows_per_view,
                self.num_heads,
                self.num_heads,
                self.head_dim,
                _f32_to_i64_bits(self.head_dim ** -0.5),
            ],
            output_shape=(rows, self.num_heads, self.head_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        attn_flat = dp.tensor_view(attn, 0, (rows, self.hidden_size))
        attn_fp8, attn_scale = _quantize_fp8_maybe_static(
            attn_flat,
            rows * self.hidden_size,
            (rows, self.hidden_size),
            self.o_act_scale if self.use_static_act_scales else None,
        )
        attn_out = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                attn_fp8,
                self.o_w_fp8,
                rows,
                self.hidden_size,
                self.hidden_size,
                attn_scale,
                self.o_w_scale,
            ],
            output_shape=(rows, self.hidden_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        dp.call_dps_kernel(
            "pi05_bias_residual_bf16",
            inputs=[hidden, attn_out, o_b, rows, self.hidden_size],
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
        ffn_hidden = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                ffn_norm_fp8,
                self.up_w_fp8,
                rows,
                self.intermediate_size,
                self.hidden_size,
                ffn_norm_scale,
                self.up_w_scale,
            ],
            output_shape=(rows, self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        if self.use_static_act_scales:
            ffn_hidden_fp8 = dp.call_dps_kernel(
                "pi05_bias_gelu_to_fp8_bf16",
                inputs=[ffn_hidden, up_b, self.down_act_scale, rows, self.intermediate_size],
                launch=_grid_1d(rows * self.intermediate_size),
                output_shape=(rows, self.intermediate_size),
                output_dtype="fp8_e4m3",
                output_device="cuda",
            )
            ffn_hidden_scale = self.down_act_scale
        else:
            dp.call_dps_kernel(
                "pi05_bias_add_bf16",
                inputs=[ffn_hidden, up_b, rows, self.intermediate_size],
                launch=_grid_1d(rows * self.intermediate_size),
            )
            dp.call_dps_kernel(
                "pi05_gelu_inplace_bf16",
                inputs=[ffn_hidden, rows * self.intermediate_size],
                launch=_grid_1d(rows * self.intermediate_size),
            )
            ffn_hidden_fp8, ffn_hidden_scale = _quantize_fp8_maybe_static(
                ffn_hidden,
                rows * self.intermediate_size,
                (rows, self.intermediate_size),
                None,
            )
        ffn_out = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                ffn_hidden_fp8,
                self.down_w_fp8,
                rows,
                self.hidden_size,
                self.intermediate_size,
                ffn_hidden_scale,
                self.down_w_scale,
            ],
            output_shape=(rows, self.hidden_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        dp.call_dps_kernel(
            "pi05_bias_residual_bf16",
            inputs=[hidden, ffn_out, down_b, rows, self.hidden_size],
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
            name="act_scale.vision_projector_w",
        )

    def forward_fast_dynamic(self, images_u8):
        register_pi05_kernels(sm_arch=89)
        hidden = self.patch.forward_fast(images_u8)
        rows = _static_dim(hidden, 0)
        for layer in self.layers:
            hidden = layer.forward_fast_dynamic(hidden)
        hidden_fp8, hidden_scale = _layer_norm_fp8_maybe_static(
            hidden,
            self.final_norm_w,
            self.final_norm_b,
            rows,
            self.hidden_size,
            _f32_to_i64_bits(self.eps),
            self.projector_act_scale if self.use_static_act_scales else None,
        )
        out = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                hidden_fp8,
                self.projector_w_fp8,
                rows,
                self.output_size,
                self.hidden_size,
                hidden_scale,
                self.projector_w_scale,
            ],
            output_shape=(rows, self.output_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        dp.call_dps_kernel(
            "pi05_bias_add_bf16",
            inputs=[out, self.projector_b, rows, self.output_size],
            launch=_grid_1d(rows * self.output_size),
        )
        return out


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
            name=f"act_scale.encoder_attn_qkv_w_{layer_idx}",
        )
        self.o_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"act_scale.encoder_attn_o_w_{layer_idx}",
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
            act0_scale_name=f"act_scale.encoder_ffn_gate_up_w_{layer_idx}",
            act1_scale_name=f"act_scale.encoder_ffn_down_w_{layer_idx}",
        )

    def forward_fast_dynamic(
        self,
        hidden,
        rope_interleaved,
        k_cache=None,
        v_cache=None,
        prefix_valid_rows=None,
        skip_post_kv: bool = False,
    ):
        register_pi05_kernels(sm_arch=89)
        rows = _static_dim(hidden, 0)
        eps_bits = _f32_to_i64_bits(self.eps)
        normed_fp8, normed_scale = _rms_norm_unit_fp8_maybe_static(
            hidden,
            rows,
            self.hidden_size,
            eps_bits,
            self.qkv_act_scale if self.use_static_act_scales else None,
        )
        qkv = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                normed_fp8,
                self.qkv_w_fp8,
                rows,
                self.qkv_dim,
                self.hidden_size,
                normed_scale,
                self.qkv_w_scale,
            ],
            output_shape=(rows, self.qkv_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        prefix_rows = rows
        prefix_layer_bytes = rows * self.num_kv_heads * self.head_dim * 2
        if k_cache is not None:
            prefix_rows = _static_dim(k_cache, 1)
            prefix_layer_bytes = prefix_rows * self.num_kv_heads * self.head_dim * 2
        if k_cache is not None and v_cache is not None:
            q = dp.call_dps_kernel(
                "pi05_qkv_split_rope_cache_bf16",
                inputs=[
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
                ],
                launch=_grid_1d(max(rows * self.num_q_heads * (self.head_dim // 2), rows * self.kv_dim)),
                output_shape=(rows, self.num_q_heads, self.head_dim),
                output_dtype="bfloat16",
                output_device="cuda",
            )
            k = dp.tensor_view(
                k_cache,
                self.layer_idx,
                (prefix_rows, self.num_kv_heads, self.head_dim),
                byte_stride=prefix_layer_bytes,
            )
            v = dp.tensor_view(
                v_cache,
                self.layer_idx,
                (prefix_rows, self.num_kv_heads, self.head_dim),
                byte_stride=prefix_layer_bytes,
            )
            if skip_post_kv:
                return hidden
        else:
            q, k, v = dp.call_dps_kernel(
                "pi05_qkv_split_rope_bf16",
                inputs=[
                    qkv,
                    rope_interleaved,
                    rows,
                    self.q_dim,
                    self.kv_dim,
                    self.kv_dim,
                    self.head_dim,
                ],
                launch=_grid_1d(max(rows * self.num_q_heads * (self.head_dim // 2), rows * self.kv_dim)),
                output_specs=[
                    nn.TensorSpec((rows, self.num_q_heads, self.head_dim), "bfloat16"),
                    nn.TensorSpec((rows, self.num_kv_heads, self.head_dim), "bfloat16"),
                    nn.TensorSpec((rows, self.num_kv_heads, self.head_dim), "bfloat16"),
                ],
            )
        if prefix_valid_rows is None:
            attn = dp.call_dps_kernel(
                "pi05_attention_bf16",
                inputs=[
                    q,
                    k,
                    v,
                    rows,
                    rows,
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    _f32_to_i64_bits(self.head_dim ** -0.5),
                ],
                launch=dp.KernelLaunchSpec(
                    grid=(rows, self.num_q_heads, 1),
                    block=(256, 1, 1),
                    shared_memory_bytes=rows * 4,
                ),
                output_shape=(rows, self.num_q_heads, self.head_dim),
                output_dtype="bfloat16",
                output_device="cuda",
            )
        else:
            attn = dp.call_dps_packed(
                "runtime.cuda.pi05_fa2_bf16",
                inputs=[
                    q,
                    k,
                    v,
                    rows,
                    prefix_valid_rows,
                    0,
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    _f32_to_i64_bits(self.head_dim ** -0.5),
                ],
                output_shape=(rows, self.num_q_heads, self.head_dim),
                output_dtype="bfloat16",
                output_device="cuda",
            )
        attn_flat = dp.tensor_view(attn, 0, (rows, self.q_dim))
        attn_fp8, attn_scale = _quantize_fp8_maybe_static(
            attn_flat,
            rows * self.q_dim,
            (rows, self.q_dim),
            self.o_act_scale if self.use_static_act_scales else None,
        )
        dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16_accum",
            inputs=[
                attn_fp8,
                self.o_w_fp8,
                hidden,
                rows,
                self.hidden_size,
                self.q_dim,
                attn_scale,
                self.o_w_scale,
            ],
        )

        ffn_norm_fp8, ffn_norm_scale = _rms_norm_unit_fp8_maybe_static(
            hidden,
            rows,
            self.hidden_size,
            eps_bits,
            self.ffn.act0_scale if self.use_static_act_scales else None,
        )
        if self.use_static_act_scales:
            self.ffn.forward_fast_from_fp8_static_accum(ffn_norm_fp8, ffn_norm_scale, rows, hidden)
        else:
            self.ffn.forward_fast_from_fp8_accum(ffn_norm_fp8, ffn_norm_scale, rows, hidden)
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

    def forward_fast_dynamic(self, prefix_embs, rope_interleaved):
        hidden = prefix_embs
        for layer in self.layers:
            hidden = layer.forward_fast_dynamic(hidden, rope_interleaved)
        return hidden

    def forward_fast_kv_dynamic(self, prefix_embs, prefix_valid_rows, rope_interleaved):
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
            hidden = layer.forward_fast_dynamic(
                hidden,
                rope_interleaved,
                k_cache,
                v_cache,
                prefix_valid_rows,
                skip_post_kv=(i == self.num_layers - 1),
            )
        return k_cache, v_cache


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

    def forward_fast_dynamic(
        self,
        noise_f32,
        prefix_embs,
        prefix_valid_rows,
        prefix_rope_interleaved,
        suffix_rope_interleaved,
    ):
        prefix_k_cache, prefix_v_cache = self.prefix_encoder.forward_fast_kv_dynamic(
            prefix_embs,
            prefix_valid_rows,
            prefix_rope_interleaved,
        )
        return self.denoise_loop.forward_fast_dynamic(
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

    def forward_fast_dynamic(
        self,
        noise_f32,
        images_u8,
        token_ids,
        prefix_valid_rows,
        prefix_rope_interleaved,
        suffix_rope_interleaved,
    ):
        register_pi05_kernels(sm_arch=89)
        image_embs = self.vision.forward_fast_dynamic(images_u8)
        lang_embs = self.language.forward_fast(token_ids)
        image_rows = _static_dim(image_embs, 0)
        lang_rows = _static_dim(lang_embs, 0)
        prefix_rows = image_rows + lang_rows
        prefix_embs = dp.call_dps_kernel(
            "pi05_prefix_concat_bf16",
            inputs=[
                image_embs,
                lang_embs,
                image_rows,
                lang_rows,
                self.prefix_hidden_size,
            ],
            launch=_grid_1d(prefix_rows * self.prefix_hidden_size),
            output_shape=(prefix_rows, self.prefix_hidden_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        prefix_k_cache, prefix_v_cache = self.prefix_encoder.forward_fast_kv_dynamic(
            prefix_embs,
            prefix_valid_rows,
            prefix_rope_interleaved,
        )
        return self.denoise_loop.forward_fast_dynamic(
            noise_f32,
            prefix_k_cache,
            prefix_v_cache,
            prefix_valid_rows,
            suffix_rope_interleaved,
        )


class PI05FFN(nn.Module):
    """Pi0.5 FFN block.

    forward() is the readable standard IR path. forward_fast() is the
    opt-in CUDA fused path and consumes pre-quantized FP8 weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        gate_up_weight_name: str | None = None,
        gate_up_scale_name: str | None = None,
        down_weight_name: str | None = None,
        down_scale_name: str | None = None,
        act0_scale_name: str | None = None,
        act1_scale_name: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

        self.gate_up_w_fp8 = Parameter(
            (2 * intermediate_size, hidden_size),
            "fp8_e4m3",
            device=device,
            name=gate_up_weight_name,
        )
        self.gate_up_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=gate_up_scale_name,
        )
        self.down_w_fp8 = Parameter(
            (hidden_size, intermediate_size),
            "fp8_e4m3",
            device=device,
            name=down_weight_name,
        )
        self.down_w_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=down_scale_name,
        )
        self.act0_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=act0_scale_name,
        )
        self.act1_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=act1_scale_name,
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = dp.multiply(dp.gelu(gate, approximate="tanh"), up)
        return self.down_proj(hidden)

    def forward_fast(self, x):
        register_pi05_kernels(sm_arch=89)
        rows = _static_dim(x, 0)
        x_fp8 = dp.call_dps_kernel(
            "pi05_quantize_fp8_static_bf16",
            inputs=[
                x,
                self.act0_scale,
                rows * self.hidden_size,
            ],
            launch=_grid_1d(rows * self.hidden_size),
            output_shape=(rows, self.hidden_size),
            output_dtype="fp8_e4m3",
            output_device="cuda",
        )
        gate_up = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                x_fp8,
                self.gate_up_w_fp8,
                rows,
                2 * self.intermediate_size,
                self.hidden_size,
                self.act0_scale,
                self.gate_up_w_scale,
            ],
            output_shape=(rows, 2 * self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden_fp8 = dp.call_dps_kernel(
            "pi05_geglu_to_fp8_bf16",
            inputs=[
                gate_up,
                self.act1_scale,
                rows,
                self.intermediate_size,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
            output_shape=(rows, self.intermediate_size),
            output_dtype="fp8_e4m3",
            output_device="cuda",
        )
        return dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                hidden_fp8,
                self.down_w_fp8,
                rows,
                self.hidden_size,
                self.intermediate_size,
                self.act1_scale,
                self.down_w_scale,
            ],
            output_like=x,
        )

    def forward_fast_from_fp8(self, x_fp8, act0_scale, rows: int):
        """FFN fast path when caller already produced FP8 normalized input."""

        register_pi05_kernels(sm_arch=89)
        gate_up = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                x_fp8,
                self.gate_up_w_fp8,
                rows,
                2 * self.intermediate_size,
                self.hidden_size,
                act0_scale,
                self.gate_up_w_scale,
            ],
            output_shape=(rows, 2 * self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden = dp.call_dps_kernel(
            "pi05_geglu_bf16",
            inputs=[
                gate_up,
                rows,
                self.intermediate_size,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
            output_shape=(rows, self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden_fp8, act1_scale = _quantize_fp8_dynamic_parallel(
            hidden,
            rows * self.intermediate_size,
            (rows, self.intermediate_size),
        )
        return dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                hidden_fp8,
                self.down_w_fp8,
                rows,
                self.hidden_size,
                self.intermediate_size,
                act1_scale,
                self.down_w_scale,
            ],
            output_shape=(rows, self.hidden_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )

    def forward_fast_from_fp8_accum(self, x_fp8, act0_scale, rows: int, residual):
        """FFN fast path with the down projection accumulated into residual."""

        register_pi05_kernels(sm_arch=89)
        gate_up = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                x_fp8,
                self.gate_up_w_fp8,
                rows,
                2 * self.intermediate_size,
                self.hidden_size,
                act0_scale,
                self.gate_up_w_scale,
            ],
            output_shape=(rows, 2 * self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden = dp.call_dps_kernel(
            "pi05_geglu_bf16",
            inputs=[
                gate_up,
                rows,
                self.intermediate_size,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
            output_shape=(rows, self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden_fp8, act1_scale = _quantize_fp8_dynamic_parallel(
            hidden,
            rows * self.intermediate_size,
            (rows, self.intermediate_size),
        )
        dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16_accum",
            inputs=[
                hidden_fp8,
                self.down_w_fp8,
                residual,
                rows,
                self.hidden_size,
                self.intermediate_size,
                act1_scale,
                self.down_w_scale,
            ],
        )
        return residual

    def forward_fast_from_fp8_static(self, x_fp8, act0_scale, rows: int):
        """FFN fast path with caller-provided input FP8 and calibrated GeGLU scale."""

        register_pi05_kernels(sm_arch=89)
        gate_up = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                x_fp8,
                self.gate_up_w_fp8,
                rows,
                2 * self.intermediate_size,
                self.hidden_size,
                act0_scale,
                self.gate_up_w_scale,
            ],
            output_shape=(rows, 2 * self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden_fp8 = dp.call_dps_kernel(
            "pi05_geglu_to_fp8_bf16",
            inputs=[
                gate_up,
                self.act1_scale,
                rows,
                self.intermediate_size,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
            output_shape=(rows, self.intermediate_size),
            output_dtype="fp8_e4m3",
            output_device="cuda",
        )
        return dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                hidden_fp8,
                self.down_w_fp8,
                rows,
                self.hidden_size,
                self.intermediate_size,
                self.act1_scale,
                self.down_w_scale,
            ],
            output_shape=(rows, self.hidden_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )

    def forward_fast_from_fp8_static_accum(self, x_fp8, act0_scale, rows: int, residual):
        """Static-scale FFN fast path with down projection accumulated in-place."""

        register_pi05_kernels(sm_arch=89)
        gate_up = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                x_fp8,
                self.gate_up_w_fp8,
                rows,
                2 * self.intermediate_size,
                self.hidden_size,
                act0_scale,
                self.gate_up_w_scale,
            ],
            output_shape=(rows, 2 * self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden_fp8 = dp.call_dps_kernel(
            "pi05_geglu_to_fp8_bf16",
            inputs=[
                gate_up,
                self.act1_scale,
                rows,
                self.intermediate_size,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
            output_shape=(rows, self.intermediate_size),
            output_dtype="fp8_e4m3",
            output_device="cuda",
        )
        dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16_accum",
            inputs=[
                hidden_fp8,
                self.down_w_fp8,
                residual,
                rows,
                self.hidden_size,
                self.intermediate_size,
                self.act1_scale,
                self.down_w_scale,
            ],
        )
        return residual

    def forward_fast_dynamic(self, x):
        """Dynamic-activation FP8 path for calibration/correctness fallback."""

        register_pi05_kernels(sm_arch=89)
        rows = _static_dim(x, 0)
        x_fp8, act0_scale = _quantize_fp8_dynamic_parallel(
            x,
            rows * self.hidden_size,
            (rows, self.hidden_size),
        )
        gate_up = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                x_fp8,
                self.gate_up_w_fp8,
                rows,
                2 * self.intermediate_size,
                self.hidden_size,
                act0_scale,
                self.gate_up_w_scale,
            ],
            output_shape=(rows, 2 * self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden = dp.call_dps_kernel(
            "pi05_geglu_bf16",
            inputs=[
                gate_up,
                rows,
                self.intermediate_size,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
            output_shape=(rows, self.intermediate_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden_fp8, act1_scale = _quantize_fp8_dynamic_parallel(
            hidden,
            rows * self.intermediate_size,
            (rows, self.intermediate_size),
        )
        return dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                hidden_fp8,
                self.down_w_fp8,
                rows,
                self.hidden_size,
                self.intermediate_size,
                act1_scale,
                self.down_w_scale,
            ],
            output_like=x,
        )


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
            name=f"act_scale.decoder_attn_qkv_w_{layer_idx}",
        )
        self.o_act_scale = Parameter(
            (1,),
            "float32",
            device=device,
            role="constant_tensor",
            name=f"act_scale.decoder_attn_o_w_{layer_idx}",
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
            act0_scale_name=f"act_scale.decoder_ffn_gate_up_w_{layer_idx}",
            act1_scale_name=f"act_scale.decoder_ffn_down_w_{layer_idx}",
        )
        self._debug_name = prefix

    def forward(self, hidden, cond):
        return dp.adarms_norm(hidden, self.adarms_weight, cond, axes=(-1,), epsilon=self.eps)

    def forward_fast_dynamic(
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
        register_pi05_kernels(sm_arch=89)
        rows = _static_dim(hidden, 0)
        prefix_rows = _static_dim(prefix_k_cache, 1)
        eps_bits = _f32_to_i64_bits(self.eps)
        bf16_bytes = 2

        prefix_layer_bytes = prefix_rows * self.num_kv_heads * self.head_dim * bf16_bytes
        prefix_k = dp.tensor_view(
            prefix_k_cache,
            self.layer_idx,
            (prefix_rows, self.num_kv_heads, self.head_dim),
            byte_stride=prefix_layer_bytes,
        )
        prefix_v = dp.tensor_view(
            prefix_v_cache,
            self.layer_idx,
            (prefix_rows, self.num_kv_heads, self.head_dim),
            byte_stride=prefix_layer_bytes,
        )
        style_step_bytes = self.num_layers * rows * 3 * self.hidden_size * bf16_bytes
        style_layer_offset = self.layer_idx * rows * 3 * self.hidden_size * bf16_bytes
        style_attn = dp.tensor_view(
            style_attn_table,
            step,
            (rows, 3 * self.hidden_size),
            byte_stride=style_step_bytes,
            base_offset=style_layer_offset,
        )
        style_ffn = dp.tensor_view(
            style_ffn_table,
            step,
            (rows, 3 * self.hidden_size),
            byte_stride=style_step_bytes,
            base_offset=style_layer_offset,
        )

        if self.use_static_act_scales:
            normed_fp8, gate_attn = dp.call_dps_kernel(
                "pi05_ada_rms_norm_style_to_fp8_bf16",
                inputs=[
                    hidden,
                    self.adarms_weight,
                    style_attn,
                    self.qkv_act_scale,
                    rows,
                    self.hidden_size,
                    eps_bits,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
                output_specs=[
                    nn.TensorSpec((rows, self.hidden_size), "fp8_e4m3"),
                    nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
                ],
            )
            normed_scale = self.qkv_act_scale
        else:
            normed, gate_attn = dp.call_dps_kernel(
                "pi05_ada_rms_norm_style_bf16",
                inputs=[
                    hidden,
                    self.adarms_weight,
                    style_attn,
                    rows,
                    self.hidden_size,
                    eps_bits,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
                output_specs=[
                    nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
                    nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
                ],
            )
            normed_fp8, normed_scale = _quantize_fp8_dynamic_parallel(
                normed,
                rows * self.hidden_size,
                (rows, self.hidden_size),
            )
        qkv = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                normed_fp8,
                self.qkv_w_fp8,
                rows,
                self.qkv_dim,
                self.hidden_size,
                normed_scale,
                self.qkv_w_scale,
            ],
            output_shape=(rows, self.qkv_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        q, full_k, full_v = dp.call_dps_kernel(
            "pi05_qkv_split_rope_concat_bf16",
            inputs=[
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
            ],
            launch=_grid_1d(
                max(
                    rows * self.num_q_heads * (self.head_dim // 2),
                    (prefix_rows + rows) * self.kv_dim,
                )
            ),
            output_specs=[
                nn.TensorSpec((rows, self.num_q_heads, self.head_dim), "bfloat16"),
                nn.TensorSpec((prefix_rows + rows, self.num_kv_heads, self.head_dim), "bfloat16"),
                nn.TensorSpec((prefix_rows + rows, self.num_kv_heads, self.head_dim), "bfloat16"),
            ],
        )
        attn = dp.call_dps_packed(
            "runtime.cuda.pi05_fa2_bf16",
            inputs=[
                q,
                full_k,
                full_v,
                rows,
                prefix_valid_rows,
                rows,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                _f32_to_i64_bits(self.head_dim ** -0.5),
            ],
            output_shape=(rows, self.num_q_heads, self.head_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        attn_flat = dp.tensor_view(attn, 0, (rows, self.q_dim))
        attn_fp8, attn_scale = _quantize_fp8_maybe_static(
            attn_flat,
            rows * self.q_dim,
            (rows, self.q_dim),
            self.o_act_scale if self.use_static_act_scales else None,
        )
        attn_out = dp.call_dps_packed(
            "runtime.cuda.fp8_nt_bf16",
            inputs=[
                attn_fp8,
                self.o_w_fp8,
                rows,
                self.hidden_size,
                self.q_dim,
                attn_scale,
                self.o_w_scale,
            ],
            output_shape=(rows, self.hidden_size),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        if self.use_static_act_scales:
            ffn_norm_fp8, gate_ffn = dp.call_dps_kernel(
                "pi05_gate_residual_ada_norm_to_fp8_bf16",
                inputs=[
                    hidden,
                    attn_out,
                    gate_attn,
                    self.adarms_weight,
                    style_ffn,
                    self.ffn.act0_scale,
                    rows,
                    self.hidden_size,
                    eps_bits,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
                output_specs=[
                    nn.TensorSpec((rows, self.hidden_size), "fp8_e4m3"),
                    nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
                ],
            )
            ffn_norm_scale = self.ffn.act0_scale
            ffn_out = self.ffn.forward_fast_from_fp8_static(ffn_norm_fp8, ffn_norm_scale, rows)
        else:
            dp.call_dps_kernel(
                "pi05_gate_mul_residual_bf16",
                inputs=[hidden, attn_out, gate_attn, rows * self.hidden_size],
                launch=_grid_1d(rows * self.hidden_size),
            )

            ffn_norm, gate_ffn = dp.call_dps_kernel(
                "pi05_ada_rms_norm_style_bf16",
                inputs=[
                    hidden,
                    self.adarms_weight,
                    style_ffn,
                    rows,
                    self.hidden_size,
                    eps_bits,
                ],
                launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
                output_specs=[
                    nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
                    nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
                ],
            )
            ffn_norm_fp8, ffn_norm_scale = _quantize_fp8_dynamic_parallel(
                ffn_norm,
                rows * self.hidden_size,
                (rows, self.hidden_size),
            )
            ffn_out = self.ffn.forward_fast_from_fp8(ffn_norm_fp8, ffn_norm_scale, rows)
        dp.call_dps_kernel(
            "pi05_gate_mul_residual_bf16",
            inputs=[hidden, ffn_out, gate_ffn, rows * self.hidden_size],
            launch=_grid_1d(rows * self.hidden_size),
        )
        return hidden


class PI05DenoiseStep(nn.Module):
    """Pi0.5 fixed-shape denoise step for the action expert.

    The action output projection weights in the FP8 artifact are pre-scaled by
    ``-1 / num_steps``. Therefore ``forward_fast_dynamic`` returns an action
    delta suitable for adding to ``x_t`` with ``dt=1``.
    """

    def __init__(
        self,
        *,
        num_layers: int = 18,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
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
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.action_horizon = int(action_horizon)
        self.num_steps = int(num_steps)
        self.action_dim = int(action_dim)
        self.eps = float(eps)
        self.use_static_act_scales = bool(use_static_act_scales)
        self.action_in = PI05Linear(
            action_dim,
            hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            weight_name="decoder_action_in_proj_w",
            bias_name="decoder_action_in_proj_b",
        )
        self.adarms_weight = Parameter(
            (hidden_size,),
            dtype,
            device=device,
            role="constant_tensor",
            name="constant.decoder_adarms_weight",
        )
        self.style_attn_table = Parameter(
            (num_steps, num_layers, action_horizon, 3 * hidden_size),
            dtype,
            device=device,
            role="constant_tensor",
            name="precomputed.decoder_style_attn",
        )
        self.style_ffn_table = Parameter(
            (num_steps, num_layers, action_horizon, 3 * hidden_size),
            dtype,
            device=device,
            role="constant_tensor",
            name="precomputed.decoder_style_ffn",
        )
        self.style_final_table = Parameter(
            (num_steps, action_horizon, 3 * hidden_size),
            dtype,
            device=device,
            role="constant_tensor",
            name="precomputed.decoder_style_final",
        )
        self.layers = nn.ModuleList(
            PI05DecoderLayer(
                layer_idx,
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                dtype=dtype,
                device=device,
                adarms_weight=self.adarms_weight,
                use_static_act_scales=use_static_act_scales,
            )
            for layer_idx in range(num_layers)
        )
        self.action_out = PI05Linear(
            hidden_size,
            action_dim,
            bias=True,
            dtype=dtype,
            device=device,
            weight_name="decoder_action_out_proj_w",
            bias_name="decoder_action_out_proj_b",
        )

    def forward(self, action_embs, cond):
        return dp.adarms_norm(action_embs, self.adarms_weight, cond, axes=(-1,), epsilon=self.eps)

    def forward_fast_dynamic(
        self,
        actions_f32,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
        step,
    ):
        register_pi05_kernels(sm_arch=89)
        rows = _static_dim(actions_f32, 0)
        actions_bf16 = dp.call_dps_kernel(
            "pi05_cast_f32_to_bf16",
            inputs=[actions_f32, rows * self.action_dim],
            launch=_grid_1d(rows * self.action_dim),
            output_shape=(rows, self.action_dim),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        hidden = self.action_in.forward_fast(actions_bf16)
        for layer in self.layers:
            hidden = layer.forward_fast_dynamic(
                hidden,
                prefix_k_cache,
                prefix_v_cache,
                prefix_valid_rows,
                rope_interleaved,
                self.style_attn_table,
                self.style_ffn_table,
                step,
            )

        style_step_bytes = self.action_horizon * 3 * self.hidden_size * 2
        style_final = dp.tensor_view(
            self.style_final_table,
            step,
            (rows, 3 * self.hidden_size),
            byte_stride=style_step_bytes,
        )
        hidden, _gate_unused = dp.call_dps_kernel(
            "pi05_ada_rms_norm_style_bf16",
            inputs=[
                hidden,
                self.adarms_weight,
                style_final,
                rows,
                self.hidden_size,
                _f32_to_i64_bits(self.eps),
            ],
            launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            output_specs=[
                nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
                nn.TensorSpec((rows, self.hidden_size), "bfloat16"),
            ],
        )
        return self.action_out.forward_fast(hidden)

    def apply_delta_fast(self, actions_f32, delta_bf16):
        register_pi05_kernels(sm_arch=89)
        rows = _static_dim(actions_f32, 0)
        dp.call_dps_kernel(
            "pi05_euler_update_bf16",
            inputs=[
                actions_f32,
                delta_bf16,
                _f32_to_i64_bits(1.0),
                rows * self.action_dim,
            ],
            launch=_grid_1d(rows * self.action_dim),
        )
        return actions_f32


class PI05DenoiseLoop(nn.Module):
    """Pi0.5 fixed 10-step denoise loop.

    This still consumes precomputed prefix KV/style resources, but it moves the
    Euler loop itself into the DSL/VM graph instead of keeping it in a C++ test
    harness. The input ``actions_f32`` is updated in-place and returned.
    """

    def __init__(
        self,
        *,
        num_layers: int = 18,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
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
        self.num_steps = int(num_steps)
        self.stepper = PI05DenoiseStep(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
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

    def forward_fast_dynamic(
        self,
        actions_f32,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
    ):
        for step in range(self.num_steps):
            delta = self.stepper.forward_fast_dynamic(
                actions_f32,
                prefix_k_cache,
                prefix_v_cache,
                prefix_valid_rows,
                rope_interleaved,
                step,
            )
            actions_f32 = self.stepper.apply_delta_fast(actions_f32, delta)
        return actions_f32


def _static_dim(value, axis: int) -> int:
    ir_value = getattr(value, "value", value)
    si = getattr(ir_value, "struct_info", None)
    shape = getattr(si, "shape", None)
    if shape is None or axis >= len(shape):
        raise ValueError("forward_fast requires static tensor rank information")
    dim = shape[axis]
    return int(getattr(dim, "value", dim))


def _grid_1d(n: int | PrimExpr, block: int = 256) -> dp.KernelLaunchSpec:
    grid_x = max(1, (n + block - 1) // block) if isinstance(n, int) else ceildiv(n, block)
    return dp.KernelLaunchSpec(grid=(grid_x, 1, 1), block=(block, 1, 1))


def _view_bf16_row(table, row: int, width: int):
    return dp.tensor_view(table, row, (width,), byte_stride=width * 2)


def _quantize_fp8_maybe_static(x, n: int, output_shape, scale):
    if scale is None:
        return _quantize_fp8_dynamic_parallel(x, n, output_shape)
    out = dp.call_dps_kernel(
        "pi05_quantize_fp8_static_bf16",
        inputs=[x, scale, n],
        launch=_grid_1d(n),
        output_shape=output_shape,
        output_dtype="fp8_e4m3",
        output_device="cuda",
    )
    return out, scale


def _layer_norm_fp8_maybe_static(x, weight, bias, rows: int, cols: int, eps_bits: int, scale):
    if scale is None:
        normed = dp.call_dps_kernel(
            "pi05_layer_norm_bf16",
            inputs=[x, weight, bias, rows, cols, eps_bits],
            launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            output_shape=(rows, cols),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        return _quantize_fp8_dynamic_parallel(normed, rows * cols, (rows, cols))
    out = dp.call_dps_kernel(
        "pi05_layer_norm_to_fp8_bf16",
        inputs=[x, weight, bias, scale, rows, cols, eps_bits],
        launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
        output_shape=(rows, cols),
        output_dtype="fp8_e4m3",
        output_device="cuda",
    )
    return out, scale


def _rms_norm_unit_fp8_maybe_static(x, rows: int, cols: int, eps_bits: int, scale):
    if scale is None:
        normed = dp.call_dps_kernel(
            "pi05_rms_norm_unit_bf16",
            inputs=[x, rows, cols, eps_bits],
            launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
            output_shape=(rows, cols),
            output_dtype="bfloat16",
            output_device="cuda",
        )
        return _quantize_fp8_dynamic_parallel(normed, rows * cols, (rows, cols))
    out = dp.call_dps_kernel(
        "pi05_rms_norm_unit_to_fp8_bf16",
        inputs=[x, scale, rows, cols, eps_bits],
        launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
        output_shape=(rows, cols),
        output_dtype="fp8_e4m3",
        output_device="cuda",
    )
    return out, scale


def _quantize_fp8_dynamic_parallel(x, n: int, output_shape):
    if n <= 4096:
        out, scale = dp.call_dps_kernel(
            "pi05_quantize_fp8_dynamic_bf16",
            inputs=[x, n],
            launch=dp.KernelLaunchSpec(grid=(1, 1, 1), block=(256, 1, 1)),
            output_specs=[
                nn.TensorSpec(output_shape, "fp8_e4m3"),
                nn.TensorSpec((1,), "float32"),
            ],
        )
        return out, scale

    partial_blocks = 128
    partial_amax = dp.call_dps_kernel(
        "pi05_reduce_amax_bf16",
        inputs=[x, n],
        launch=dp.KernelLaunchSpec(grid=(partial_blocks, 1, 1), block=(256, 1, 1)),
        output_shape=(partial_blocks,),
        output_dtype="float32",
        output_device="cuda",
    )
    scale = dp.call_dps_kernel(
        "pi05_amax_to_scale",
        inputs=[partial_amax, partial_blocks],
        launch=dp.KernelLaunchSpec(grid=(1, 1, 1), block=(256, 1, 1)),
        output_shape=(1,),
        output_dtype="float32",
        output_device="cuda",
    )
    out = dp.call_dps_kernel(
        "pi05_quantize_fp8_static_bf16",
        inputs=[x, scale, n],
        launch=_grid_1d(n),
        output_shape=output_shape,
        output_dtype="fp8_e4m3",
        output_device="cuda",
    )
    return out, scale


def _f32_to_i64_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<f", float(value)) + b"\x00\x00\x00\x00", "little")


__all__ = [
    "PI05Attention",
    "PI05DecoderLayer",
    "PI05DenoiseStep",
    "PI05DenoiseLoop",
    "PI05FFN",
    "PI05LanguageEmbedding",
    "PI05Linear",
    "PI05PaliGemmaEncoderLayer",
    "PI05PaliGemmaPrefixEncoder",
    "PI05SampleActionsFromPrefixEmbeddings",
    "PI05SampleActionsFromTokens",
    "PI05VisionEncoder",
    "PI05VisionEncoderLayer",
    "PI05VisionPatchEmbedding",
]
