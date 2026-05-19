"""Common nn frontend layers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import devproc2 as dp

from devproc2.nn.module import Module
from devproc2.nn.specs import Parameter


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: str = "float16",
        device: str = "cuda",
        padding_idx: Optional[int] = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.scale = scale
        self.weight = Parameter((num_embeddings, embedding_dim), dtype, device=device)

    def forward(self, input):
        output = dp.embedding(
            input,
            self.weight,
            padding_idx=self.padding_idx,
        )
        if self.scale != 1.0:
            output = dp.multiply(output, self.scale)
        return output


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: str = "float16",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter((out_features, in_features), dtype, device=device)
        self.bias = Parameter((out_features,), dtype, device=device) if bias else None

    def forward(self, input):
        weight_t = dp.permute_dims(self.weight, axes=(1, 0))
        output = dp.matmul(input, weight_t)
        if self.bias is not None:
            output = dp.add(output, self.bias)
        return output


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: str = "float16",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(self.normalized_shape, dtype, device=device)
            self.bias = Parameter(self.normalized_shape, dtype, device=device)
        else:
            self.weight = None
            self.bias = None

    def forward(self, input):
        if self.weight is None or self.bias is None:
            raise RuntimeError("LayerNorm without elementwise_affine is not supported yet")
        return dp.layer_norm(
            input,
            self.weight,
            self.bias,
            axes=tuple(range(-len(self.normalized_shape), 0)),
            epsilon=self.eps,
        )


class RMSNorm(Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        use_adarms: bool = False,
        dtype: str = "float16",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_adarms = use_adarms
        self.weight = Parameter((hidden_size,), dtype, device=device)

    def forward(self, input, cond=None):
        if self.use_adarms:
            if cond is None:
                raise RuntimeError("RMSNorm(use_adarms=True) requires cond")
            return dp.adarms_norm(input, self.weight, cond, axes=(-1,), epsilon=self.eps)
        if cond is not None:
            raise RuntimeError("RMSNorm(use_adarms=False) does not accept cond")
        return dp.rms_norm(input, self.weight, axes=(-1,), epsilon=self.eps)


class GELU(Module):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return dp.gelu(input, approximate=self.approximate)


class SiLU(Module):
    def forward(self, input):
        return dp.silu(input)


__all__ = [
    "Embedding",
    "GELU",
    "LayerNorm",
    "Linear",
    "RMSNorm",
    "SiLU",
]
