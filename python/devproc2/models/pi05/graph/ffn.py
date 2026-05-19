"""Pi0.5 feed-forward network block."""
from __future__ import annotations

import devproc2 as dp
import devproc2.nn as nn

from devproc2.nn.specs import Parameter

from .. import ops as pi05_ops
from ._helpers import _grid_1d, _quantize_fp8_dynamic_parallel, _static_dim


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
        rows = _static_dim(x, 0)
        x_fp8 = dp.empty((rows, self.hidden_size), dtype="fp8_e4m3", device="cuda")
        pi05_ops.call_cuda(
            "pi05_quantize_fp8_static_bf16",
            *[
                x,
                self.act0_scale,
                rows * self.hidden_size,
                x_fp8,
            ],
            launch=_grid_1d(rows * self.hidden_size),
        )
        gate_up = pi05_ops.fp8_linear(
            x_fp8,
            self.gate_up_w_fp8,
            rows=rows,
            out_features=2 * self.intermediate_size,
            in_features=self.hidden_size,
            x_scale=self.act0_scale,
            weight_scale=self.gate_up_w_scale,
        )
        hidden_fp8 = dp.empty((rows, self.intermediate_size), dtype="fp8_e4m3", device="cuda")
        pi05_ops.call_cuda(
            "pi05_geglu_to_fp8_bf16",
            *[
                gate_up,
                self.act1_scale,
                rows,
                self.intermediate_size,
                hidden_fp8,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
        )
        return pi05_ops.fp8_linear(
            hidden_fp8,
            self.down_w_fp8,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.intermediate_size,
            x_scale=self.act1_scale,
            weight_scale=self.down_w_scale,
        )

    def _forward_fast_from_fp8(self, x_fp8, act0_scale, rows: int):
        """FFN fast path when caller already produced FP8 normalized input."""
        gate_up = pi05_ops.fp8_linear(
            x_fp8,
            self.gate_up_w_fp8,
            rows=rows,
            out_features=2 * self.intermediate_size,
            in_features=self.hidden_size,
            x_scale=act0_scale,
            weight_scale=self.gate_up_w_scale,
        )
        hidden = dp.empty((rows, self.intermediate_size), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_geglu_bf16",
            *[
                gate_up,
                rows,
                self.intermediate_size,
                hidden,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
        )
        hidden_fp8, act1_scale = _quantize_fp8_dynamic_parallel(
            hidden,
            rows * self.intermediate_size,
            (rows, self.intermediate_size),
        )
        return pi05_ops.fp8_linear(
            hidden_fp8,
            self.down_w_fp8,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.intermediate_size,
            x_scale=act1_scale,
            weight_scale=self.down_w_scale,
        )

    def _forward_fast_from_fp8_accum(self, x_fp8, act0_scale, rows: int, residual):
        """FFN fast path with the down projection accumulated into residual."""
        gate_up = pi05_ops.fp8_linear(
            x_fp8,
            self.gate_up_w_fp8,
            rows=rows,
            out_features=2 * self.intermediate_size,
            in_features=self.hidden_size,
            x_scale=act0_scale,
            weight_scale=self.gate_up_w_scale,
        )
        hidden = dp.empty((rows, self.intermediate_size), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_geglu_bf16",
            *[
                gate_up,
                rows,
                self.intermediate_size,
                hidden,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
        )
        hidden_fp8, act1_scale = _quantize_fp8_dynamic_parallel(
            hidden,
            rows * self.intermediate_size,
            (rows, self.intermediate_size),
        )
        pi05_ops.fp8_linear_accum_(
            hidden_fp8,
            self.down_w_fp8,
            residual=residual,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.intermediate_size,
            x_scale=act1_scale,
            weight_scale=self.down_w_scale,
        )
        return residual

    def _forward_fast_from_fp8_static(self, x_fp8, act0_scale, rows: int):
        """FFN fast path with caller-provided input FP8 and calibrated GeGLU scale."""
        gate_up = pi05_ops.fp8_linear(
            x_fp8,
            self.gate_up_w_fp8,
            rows=rows,
            out_features=2 * self.intermediate_size,
            in_features=self.hidden_size,
            x_scale=act0_scale,
            weight_scale=self.gate_up_w_scale,
        )
        hidden_fp8 = dp.empty((rows, self.intermediate_size), dtype="fp8_e4m3", device="cuda")
        pi05_ops.call_cuda(
            "pi05_geglu_to_fp8_bf16",
            *[
                gate_up,
                self.act1_scale,
                rows,
                self.intermediate_size,
                hidden_fp8,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
        )
        return pi05_ops.fp8_linear(
            hidden_fp8,
            self.down_w_fp8,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.intermediate_size,
            x_scale=self.act1_scale,
            weight_scale=self.down_w_scale,
        )

    def _forward_fast_from_fp8_static_accum(self, x_fp8, act0_scale, rows: int, residual):
        """Static-scale FFN fast path with down projection accumulated in-place."""
        gate_up = pi05_ops.fp8_linear(
            x_fp8,
            self.gate_up_w_fp8,
            rows=rows,
            out_features=2 * self.intermediate_size,
            in_features=self.hidden_size,
            x_scale=act0_scale,
            weight_scale=self.gate_up_w_scale,
        )
        hidden_fp8 = dp.empty((rows, self.intermediate_size), dtype="fp8_e4m3", device="cuda")
        pi05_ops.call_cuda(
            "pi05_geglu_to_fp8_bf16",
            *[
                gate_up,
                self.act1_scale,
                rows,
                self.intermediate_size,
                hidden_fp8,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
        )
        pi05_ops.fp8_linear_accum_(
            hidden_fp8,
            self.down_w_fp8,
            residual=residual,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.intermediate_size,
            x_scale=self.act1_scale,
            weight_scale=self.down_w_scale,
        )
        return residual

    def _forward_fast_dynamic(self, x):
        """Dynamic-activation FP8 path for calibration/correctness fallback."""
        rows = _static_dim(x, 0)
        x_fp8, act0_scale = _quantize_fp8_dynamic_parallel(
            x,
            rows * self.hidden_size,
            (rows, self.hidden_size),
        )
        gate_up = pi05_ops.fp8_linear(
            x_fp8,
            self.gate_up_w_fp8,
            rows=rows,
            out_features=2 * self.intermediate_size,
            in_features=self.hidden_size,
            x_scale=act0_scale,
            weight_scale=self.gate_up_w_scale,
        )
        hidden = dp.empty((rows, self.intermediate_size), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_geglu_bf16",
            *[
                gate_up,
                rows,
                self.intermediate_size,
                hidden,
            ],
            launch=_grid_1d(rows * self.intermediate_size),
        )
        hidden_fp8, act1_scale = _quantize_fp8_dynamic_parallel(
            hidden,
            rows * self.intermediate_size,
            (rows, self.intermediate_size),
        )
        return pi05_ops.fp8_linear(
            hidden_fp8,
            self.down_w_fp8,
            rows=rows,
            out_features=self.hidden_size,
            in_features=self.intermediate_size,
            x_scale=act1_scale,
            weight_scale=self.down_w_scale,
        )






__all__ = ["PI05FFN"]
