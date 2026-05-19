"""Pi0.5 denoise step and fixed-step loop."""
from __future__ import annotations

import devproc2 as dp
import devproc2.nn as nn

from devproc2.nn.specs import Parameter

from .. import ops as pi05_ops
from ._helpers import _f32_to_i64_bits, _grid_1d, _static_dim
from .decoder import PI05DecoderLayer
from .layers import PI05Linear


class PI05DenoiseStep(nn.Module):
    """Pi0.5 fixed-shape denoise step for the action expert.

    The action output projection weights in the FP8 artifact are pre-scaled by
    ``-1 / num_steps``. Therefore ``forward_fast`` returns an action
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

    def forward(
        self,
        actions_f32,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
        step,
    ):
        rows = _static_dim(actions_f32, 0)
        hidden = self.action_in(actions_f32)
        for layer in self.layers:
            hidden = layer(
                hidden,
                prefix_k_cache,
                prefix_v_cache,
                prefix_valid_rows,
                rope_interleaved,
                self.style_attn_table,
                self.style_ffn_table,
                step,
            )

        style_final = dp.select(self.style_final_table, axis=0, index=step)
        hidden_out = dp.adarms_norm(
            hidden,
            self.adarms_weight,
            style_final,
            axes=(-1,),
            epsilon=self.eps,
        )
        return self.action_out(hidden_out)

    def forward_fast(
        self,
        actions_f32,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
        step,
    ):
        rows = _static_dim(actions_f32, 0)
        actions_bf16 = dp.empty((rows, self.action_dim), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_cast_f32_to_bf16",
            actions_f32,
            rows * self.action_dim,
            actions_bf16,
            launch=_grid_1d(rows * self.action_dim),
        )
        hidden = self.action_in.forward_fast(actions_bf16)
        for layer in self.layers:
            hidden = layer.forward_fast(
                hidden,
                prefix_k_cache,
                prefix_v_cache,
                prefix_valid_rows,
                rope_interleaved,
                self.style_attn_table,
                self.style_ffn_table,
                step,
            )

        style_final = dp.select(self.style_final_table, axis=0, index=step)
        hidden_out = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
        gate_unused = dp.empty((rows, self.hidden_size), dtype="bfloat16", device="cuda")
        pi05_ops.call_cuda(
            "pi05_ada_rms_norm_style_bf16",
            *[
                hidden,
                self.adarms_weight,
                style_final,
                rows,
                self.hidden_size,
                _f32_to_i64_bits(self.eps),
                hidden_out,
                gate_unused,
            ],
            launch=dp.KernelLaunchSpec(grid=(rows, 1, 1), block=(256, 1, 1)),
        )
        return self.action_out.forward_fast(hidden_out)

    def _apply_delta_fast(self, actions_f32, delta_bf16):
        rows = _static_dim(actions_f32, 0)
        pi05_ops.call_cuda(
            "pi05_euler_update_bf16",
            actions_f32,
            delta_bf16,
            _f32_to_i64_bits(1.0),
            rows * self.action_dim,
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

    def forward(
        self,
        actions_f32,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
    ):
        for step in range(self.num_steps):
            delta = self.stepper(
                actions_f32,
                prefix_k_cache,
                prefix_v_cache,
                prefix_valid_rows,
                rope_interleaved,
                step,
            )
            actions_f32 = dp.add(actions_f32, delta)
        return actions_f32

    def forward_fast(
        self,
        actions_f32,
        prefix_k_cache,
        prefix_v_cache,
        prefix_valid_rows,
        rope_interleaved,
    ):
        for step in range(self.num_steps):
            delta = self.stepper.forward_fast(
                actions_f32,
                prefix_k_cache,
                prefix_v_cache,
                prefix_valid_rows,
                rope_interleaved,
                step,
            )
            actions_f32 = self.stepper._apply_delta_fast(actions_f32, delta)
        return actions_f32






__all__ = [
    "PI05DenoiseLoop",
    "PI05DenoiseStep",
]
