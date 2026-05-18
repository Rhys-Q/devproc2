"""Reusable Pi0.5 primitive layers."""
from __future__ import annotations

import devproc2 as dp
import devproc2.nn as nn

from devproc2.nn.specs import Parameter

from .. import ops as pi05_ops
from ._helpers import _f32_to_i64_bits, _grid_1d, _static_dim


class PI05Linear(nn.Module):
    """Pi0.5 row-major [K, N] linear projection.

    This matches the converted Pi0.5 deployment weight layout. The standard forward
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
        rows = _static_dim(x, 0)
        out = pi05_ops.bf16_linear(
            x,
            self.weight,
            rows=rows,
            out_features=self.out_features,
            in_features=self.in_features,
        )
        if self.bias is not None:
            pi05_ops.call_cuda(
                "pi05_bias_add_bf16",
                out,
                self.bias,
                rows,
                self.out_features,
                launch=_grid_1d(rows * self.out_features),
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
        num_tokens = _static_dim(token_ids, 0)
        out = dp.empty((num_tokens, self.hidden_size), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_embedding_gather_bf16",
            token_ids,
            self.embedding,
            num_tokens,
            self.hidden_size,
            out,
            launch=_grid_1d(num_tokens * self.hidden_size),
        )
        return out






class PI05Attention(nn.Module):
    """Pi0.5 BF16 attention correctness fallback.

    The performance target path should replace this with FA2 through the same
    DPS boundary. This module gives frontend DSL/VM a concrete attention call
    while the Pi0.5 FA2 packed func is integrated.
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

    def forward(self, q, k, v):
        return dp.attention(q, k, v, scale=self.scale)

    def forward_fast(self, q, k, v):
        rows_q = _static_dim(q, 0)
        rows_k = _static_dim(k, 0)
        out = dp.empty(
            (rows_q, self.num_q_heads, self.head_dim),
            dtype="bfloat16",
            device="cuda",
        )
        launch = dp.KernelLaunchSpec(
            grid=(rows_q, self.num_q_heads, 1),
            block=(256, 1, 1),
            shared_memory_bytes=rows_k * 4,
        )
        pi05_ops.call_cuda(
            "pi05_attention_bf16",
            *[
                q,
                k,
                v,
                rows_q,
                rows_k,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                _f32_to_i64_bits(self.scale),
                out,
            ],
            launch=launch,
        )
        return out






__all__ = [
    "PI05Attention",
    "PI05LanguageEmbedding",
    "PI05Linear",
]
