"""Pi0.5 executable/artifact export helpers."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any

import devproc2 as dp
import devproc2.frontend.dsl as dsl

from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_abi import EmitABIPass
from devproc2.compiler.passes.emit_executable import EmitExecutablePass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir.nodes import Function, IRModule
from devproc2.ir.ops import ReturnOp
from devproc2.nn import GraphBuilder, ScalarSpec, TensorSpec
from devproc2.models.pi05.artifact import Pi05ArtifactSummary, prepare_pi05_artifact
from devproc2.models.pi05.modules import (
    PI05DenoiseLoop,
    PI05DenoiseStep,
    PI05PaliGemmaPrefixEncoder,
    PI05SampleActionsFromPrefixEmbeddings,
    PI05SampleActionsFromTokens,
    PI05VisionEncoder,
)
from devproc2.vm.executable import Executable


DEFAULT_PREFIX_ROWS = 968
DEFAULT_ACTION_HORIZON = 50
DEFAULT_ACTION_DIM = 32
DEFAULT_NUM_STEPS = 10
DEFAULT_NUM_LAYERS = 18
DEFAULT_HIDDEN_SIZE = 1024
DEFAULT_INTERMEDIATE_SIZE = 4096
DEFAULT_NUM_Q_HEADS = 8
DEFAULT_NUM_KV_HEADS = 1
DEFAULT_HEAD_DIM = 256
DEFAULT_PREFIX_HIDDEN_SIZE = 2048
DEFAULT_PREFIX_INTERMEDIATE_SIZE = 16384
DEFAULT_MAX_PROMPT_LEN = 200
DEFAULT_VOCAB_SIZE = 257152
DEFAULT_VISION_LAYERS = 27
DEFAULT_VISION_VIEWS = 3
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 14
DEFAULT_IMAGE_CHANNELS = 3
DEFAULT_VISION_HIDDEN_SIZE = 1152
DEFAULT_VISION_INTERMEDIATE_SIZE = 4304
DEFAULT_VISION_HEADS = 16
DEFAULT_VISION_OUTPUT_SIZE = 2048


@dataclass(frozen=True)
class Pi05DenoiseCompileResult:
    module: IRModule
    lowered_module: IRModule
    executable: Executable
    context: PassContext
    num_user_inputs: int


@dataclass(frozen=True)
class Pi05DenoiseExportSummary:
    artifact_dir: Path
    function_name: str
    num_user_inputs: int
    num_weight_params: int
    vm_functions: int
    instructions: int
    storage_bytes: int
    resource_summary: Pi05ArtifactSummary | None = None

    def to_json_obj(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_dir": str(self.artifact_dir),
            "function_name": self.function_name,
            "num_user_inputs": self.num_user_inputs,
            "num_weight_params": self.num_weight_params,
            "vm_functions": self.vm_functions,
            "instructions": self.instructions,
            "storage_bytes": self.storage_bytes,
        }
        if self.resource_summary is not None:
            payload["weights_entries"] = self.resource_summary.weights_entries
            payload["kernels"] = self.resource_summary.kernels
            payload["tokenizer"] = self.resource_summary.tokenizer
            payload["fp8_layout"] = self.resource_summary.fp8_layout
        return payload


def pi05_denoise_input_specs(
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
) -> dict[str, object]:
    """Return fixed-shape denoise-step ABI specs.

    This is one denoise step for one sample. The full sample_actions runner will
    wrap it in the 10-step Euler loop and provide prefix KV/style tables.
    """

    return {
        "actions_f32": TensorSpec((action_horizon, action_dim), "float32", device=device),
        "prefix_k_cache": TensorSpec(
            (num_layers, prefix_rows, num_kv_heads, head_dim),
            "bfloat16",
            device=device,
        ),
        "prefix_v_cache": TensorSpec(
            (num_layers, prefix_rows, num_kv_heads, head_dim),
            "bfloat16",
            device=device,
        ),
        "prefix_valid_rows": ScalarSpec("int64"),
        "rope_interleaved": TensorSpec((action_horizon, head_dim), "bfloat16", device=device),
        "step": ScalarSpec("int64"),
    }


def pi05_denoise_loop_input_specs(
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
) -> dict[str, object]:
    """Return fixed-shape ABI specs for the unrolled denoise loop."""

    specs = pi05_denoise_input_specs(
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_rows=prefix_rows,
        num_steps=num_steps,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )
    specs.pop("step")
    return specs


def pi05_sample_actions_precomputed_prefix_input_specs(
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
) -> dict[str, object]:
    """ABI for sample_actions when prefix KV/RoPE are supplied by a caller.

    This is the deployable back half of Pi0.5 ``sample_actions``: it accepts
    initial noise plus precomputed prefix resources and returns final actions.
    The vision/text prefix path remains a separate, not-yet-compiled graph.
    """

    return {
        "noise_f32": TensorSpec((action_horizon, action_dim), "float32", device=device),
        "prefix_k_cache": TensorSpec(
            (num_layers, prefix_rows, num_kv_heads, head_dim),
            "bfloat16",
            device=device,
        ),
        "prefix_v_cache": TensorSpec(
            (num_layers, prefix_rows, num_kv_heads, head_dim),
            "bfloat16",
            device=device,
        ),
        "prefix_valid_rows": ScalarSpec("int64"),
        "rope_interleaved": TensorSpec((action_horizon, head_dim), "bfloat16", device=device),
    }


def pi05_sample_actions_precomputed_prefix_embs_input_specs(
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    prefix_hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
) -> dict[str, object]:
    """ABI for sample_actions from caller-supplied prefix embeddings.

    The caller still prepares multimodal prefix embeddings and RoPE tables, but
    prefix transformer KV materialization and the denoise loop run in one VM
    invocation.
    """

    return {
        "noise_f32": TensorSpec((action_horizon, action_dim), "float32", device=device),
        "prefix_embs": TensorSpec((prefix_rows, prefix_hidden_size), "bfloat16", device=device),
        "prefix_valid_rows": ScalarSpec("int64"),
        "prefix_rope_interleaved": TensorSpec((prefix_rows, head_dim), "bfloat16", device=device),
        "suffix_rope_interleaved": TensorSpec((action_horizon, head_dim), "bfloat16", device=device),
    }


def pi05_sample_actions_tokens_input_specs(
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    num_views: int = DEFAULT_VISION_VIEWS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    image_channels: int = DEFAULT_IMAGE_CHANNELS,
    max_prompt_len: int = DEFAULT_MAX_PROMPT_LEN,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
) -> dict[str, object]:
    """ABI for sample_actions from raw image tensor plus token ids."""

    return {
        "noise_f32": TensorSpec((action_horizon, action_dim), "float32", device=device),
        "images_u8": TensorSpec(
            (num_views, image_size, image_size, image_channels),
            "uint8",
            device=device,
        ),
        "token_ids": TensorSpec((max_prompt_len,), "int32", device=device),
        "prefix_valid_rows": ScalarSpec("int64"),
        "prefix_rope_interleaved": TensorSpec((prefix_rows, head_dim), "bfloat16", device=device),
        "suffix_rope_interleaved": TensorSpec((action_horizon, head_dim), "bfloat16", device=device),
    }


def pi05_vision_encoder_input_specs(
    *,
    num_views: int = DEFAULT_VISION_VIEWS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    image_channels: int = DEFAULT_IMAGE_CHANNELS,
    device: str = "cuda",
) -> dict[str, object]:
    """Return fixed-shape ABI specs for the SigLIP vision encoder slice."""

    return {
        "images_u8": TensorSpec(
            (num_views, image_size, image_size, image_channels),
            "uint8",
            device=device,
        ),
    }


def pi05_paligemma_prefix_encoder_input_specs(
    *,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
) -> dict[str, object]:
    """Return fixed-shape ABI specs for the compact PaliGemma prefix encoder."""

    return {
        "prefix_embs": TensorSpec((prefix_rows, hidden_size), "bfloat16", device=device),
        "rope_interleaved": TensorSpec((prefix_rows, head_dim), "bfloat16", device=device),
    }


def pi05_paligemma_prefix_kv_encoder_input_specs(
    *,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
) -> dict[str, object]:
    """Return ABI specs for the compact PaliGemma prefix KV encoder."""

    return {
        "prefix_embs": TensorSpec((prefix_rows, hidden_size), "bfloat16", device=device),
        "prefix_valid_rows": ScalarSpec("int64"),
        "rope_interleaved": TensorSpec((prefix_rows, head_dim), "bfloat16", device=device),
    }


def build_pi05_denoise_module(
    *,
    function_name: str = "main",
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_q_heads: int = DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
    use_static_act_scales: bool = False,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build the Pi0.5 denoise fast-path IR module."""

    if reset_dsl:
        dp.reset_module()
    denoise = PI05DenoiseStep(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        action_horizon=action_horizon,
        num_steps=num_steps,
        action_dim=action_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )
    input_specs = pi05_denoise_input_specs(
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_rows=prefix_rows,
        num_steps=num_steps,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )
    module = GraphBuilder().build(denoise.forward_fast_dynamic, input_specs)
    fn = module.functions["forward_fast_dynamic"]
    if function_name != "forward_fast_dynamic":
        module = IRModule({function_name: fn})
    return module, len(input_specs)


def _build_pi05_denoise_loop_module_with_specs(
    *,
    input_specs: dict[str, object],
    function_name: str,
    action_horizon: int,
    action_dim: int,
    num_steps: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
    use_static_act_scales: bool = False,
) -> tuple[IRModule, int]:
    denoise_loop = PI05DenoiseLoop(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        action_horizon=action_horizon,
        num_steps=num_steps,
        action_dim=action_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )
    module = GraphBuilder().build(denoise_loop.forward_fast_dynamic, input_specs)
    fn = module.functions["forward_fast_dynamic"]
    if function_name != "forward_fast_dynamic":
        module = IRModule({function_name: fn})
    return module, len(input_specs)


def build_pi05_denoise_loop_module(
    *,
    function_name: str = "main",
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_q_heads: int = DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
    use_static_act_scales: bool = False,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build the Pi0.5 fixed-step denoise loop IR module."""

    if reset_dsl:
        dp.reset_module()
    input_specs = pi05_denoise_loop_input_specs(
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_rows=prefix_rows,
        num_steps=num_steps,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )
    return _build_pi05_denoise_loop_module_with_specs(
        input_specs=input_specs,
        function_name=function_name,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_steps=num_steps,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )


def build_pi05_sample_actions_precomputed_prefix_module(
    *,
    function_name: str = "main",
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_q_heads: int = DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
    use_static_act_scales: bool = False,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build sample_actions back-half IR with caller-supplied prefix KV."""

    if reset_dsl:
        dp.reset_module()
    input_specs = pi05_sample_actions_precomputed_prefix_input_specs(
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_rows=prefix_rows,
        num_steps=num_steps,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )
    return _build_pi05_denoise_loop_module_with_specs(
        input_specs=input_specs,
        function_name=function_name,
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_steps=num_steps,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )


def build_pi05_sample_actions_precomputed_prefix_embs_module(
    *,
    function_name: str = "main",
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    prefix_hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
    prefix_intermediate_size: int = DEFAULT_PREFIX_INTERMEDIATE_SIZE,
    decoder_hidden_size: int = DEFAULT_HIDDEN_SIZE,
    decoder_intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_q_heads: int = DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
    use_static_act_scales: bool = False,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build sample_actions IR from caller-supplied prefix embeddings."""

    if reset_dsl:
        dp.reset_module()
    sample = PI05SampleActionsFromPrefixEmbeddings(
        num_layers=num_layers,
        prefix_hidden_size=prefix_hidden_size,
        prefix_intermediate_size=prefix_intermediate_size,
        decoder_hidden_size=decoder_hidden_size,
        decoder_intermediate_size=decoder_intermediate_size,
        action_horizon=action_horizon,
        num_steps=num_steps,
        action_dim=action_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )
    input_specs = pi05_sample_actions_precomputed_prefix_embs_input_specs(
        action_horizon=action_horizon,
        action_dim=action_dim,
        prefix_rows=prefix_rows,
        prefix_hidden_size=prefix_hidden_size,
        head_dim=head_dim,
        device=device,
    )
    module = GraphBuilder().build(sample.forward_fast_dynamic, input_specs)
    fn = module.functions["forward_fast_dynamic"]
    if function_name != "forward_fast_dynamic":
        module = IRModule({function_name: fn})
    return module, len(input_specs)


def build_pi05_sample_actions_tokens_module(
    *,
    function_name: str = "main",
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_dim: int = DEFAULT_ACTION_DIM,
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_steps: int = DEFAULT_NUM_STEPS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_views: int = DEFAULT_VISION_VIEWS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    image_channels: int = DEFAULT_IMAGE_CHANNELS,
    vision_layers: int = DEFAULT_VISION_LAYERS,
    vision_hidden_size: int = DEFAULT_VISION_HIDDEN_SIZE,
    vision_intermediate_size: int = DEFAULT_VISION_INTERMEDIATE_SIZE,
    vision_heads: int = DEFAULT_VISION_HEADS,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_prompt_len: int = DEFAULT_MAX_PROMPT_LEN,
    prefix_hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
    prefix_intermediate_size: int = DEFAULT_PREFIX_INTERMEDIATE_SIZE,
    decoder_hidden_size: int = DEFAULT_HIDDEN_SIZE,
    decoder_intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
    num_q_heads: int = DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
    use_static_act_scales: bool = False,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build sample_actions IR from images and token ids."""

    if reset_dsl:
        dp.reset_module()
    sample = PI05SampleActionsFromTokens(
        num_layers=num_layers,
        num_views=num_views,
        image_size=image_size,
        patch_size=patch_size,
        image_channels=image_channels,
        vision_layers=vision_layers,
        vision_hidden_size=vision_hidden_size,
        vision_intermediate_size=vision_intermediate_size,
        vision_heads=vision_heads,
        vocab_size=vocab_size,
        max_prompt_len=max_prompt_len,
        prefix_hidden_size=prefix_hidden_size,
        prefix_intermediate_size=prefix_intermediate_size,
        decoder_hidden_size=decoder_hidden_size,
        decoder_intermediate_size=decoder_intermediate_size,
        action_horizon=action_horizon,
        num_steps=num_steps,
        action_dim=action_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )
    input_specs = pi05_sample_actions_tokens_input_specs(
        action_horizon=action_horizon,
        action_dim=action_dim,
        num_views=num_views,
        image_size=image_size,
        image_channels=image_channels,
        max_prompt_len=max_prompt_len,
        prefix_rows=prefix_rows,
        head_dim=head_dim,
        device=device,
    )
    module = GraphBuilder().build(sample.forward_fast_dynamic, input_specs)
    fn = module.functions["forward_fast_dynamic"]
    if function_name != "forward_fast_dynamic":
        module = IRModule({function_name: fn})
    return module, len(input_specs)


def build_pi05_vision_encoder_module(
    *,
    function_name: str = "main",
    num_layers: int = DEFAULT_VISION_LAYERS,
    num_views: int = DEFAULT_VISION_VIEWS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    image_channels: int = DEFAULT_IMAGE_CHANNELS,
    hidden_size: int = DEFAULT_VISION_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_VISION_INTERMEDIATE_SIZE,
    num_heads: int = DEFAULT_VISION_HEADS,
    output_size: int = DEFAULT_VISION_OUTPUT_SIZE,
    device: str = "cuda",
    use_static_act_scales: bool = False,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build the SigLIP vision encoder IR module for the prefix path."""

    if reset_dsl:
        dp.reset_module()
    encoder = PI05VisionEncoder(
        num_layers=num_layers,
        num_views=num_views,
        image_size=image_size,
        patch_size=patch_size,
        in_channels=image_channels,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        output_size=output_size,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )
    input_specs = pi05_vision_encoder_input_specs(
        num_views=num_views,
        image_size=image_size,
        image_channels=image_channels,
        device=device,
    )
    module = GraphBuilder().build(encoder.forward_fast_dynamic, input_specs)
    fn = module.functions["forward_fast_dynamic"]
    if function_name != "forward_fast_dynamic":
        module = IRModule({function_name: fn})
    return module, len(input_specs)


def build_pi05_paligemma_prefix_encoder_module(
    *,
    function_name: str = "main",
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_PREFIX_INTERMEDIATE_SIZE,
    num_q_heads: int = DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    device: str = "cuda",
    use_static_act_scales: bool = False,
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build the compact PaliGemma prefix transformer IR module."""

    if reset_dsl:
        dp.reset_module()
    encoder = PI05PaliGemmaPrefixEncoder(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
        use_static_act_scales=use_static_act_scales,
    )
    input_specs = pi05_paligemma_prefix_encoder_input_specs(
        prefix_rows=prefix_rows,
        hidden_size=hidden_size,
        head_dim=head_dim,
        device=device,
    )
    module = GraphBuilder().build(encoder.forward_fast_dynamic, input_specs)
    fn = module.functions["forward_fast_dynamic"]
    if function_name != "forward_fast_dynamic":
        module = IRModule({function_name: fn})
    return module, len(input_specs)


def build_pi05_paligemma_prefix_kv_encoder_module(
    *,
    function_name: str = "main",
    prefix_rows: int = DEFAULT_PREFIX_ROWS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_PREFIX_INTERMEDIATE_SIZE,
    num_q_heads: int = DEFAULT_NUM_Q_HEADS,
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    use_static_act_scales: bool = False,
    device: str = "cuda",
    reset_dsl: bool = True,
) -> tuple[IRModule, int]:
    """Build the compact PaliGemma prefix transformer plus KV-cache outputs."""

    if reset_dsl:
        dp.reset_module()
    encoder = PI05PaliGemmaPrefixEncoder(
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        use_static_act_scales=use_static_act_scales,
        device=device,
    )
    input_specs = pi05_paligemma_prefix_kv_encoder_input_specs(
        prefix_rows=prefix_rows,
        hidden_size=hidden_size,
        head_dim=head_dim,
        device=device,
    )
    module = GraphBuilder().build(encoder.forward_fast_kv_dynamic, input_specs)
    fn = module.functions["forward_fast_kv_dynamic"]
    if function_name != "forward_fast_kv_dynamic":
        module = IRModule({function_name: fn})
    return module, len(input_specs)


def compile_pi05_denoise_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile the denoise fast path to VM bytecode structures."""

    module, num_user_inputs = build_pi05_denoise_module(**module_kwargs)
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def compile_pi05_denoise_loop_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile the unrolled denoise loop to VM bytecode structures."""

    module, num_user_inputs = build_pi05_denoise_loop_module(**module_kwargs)
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def compile_pi05_sample_actions_precomputed_prefix_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile sample_actions back half with precomputed prefix resources."""

    module, num_user_inputs = build_pi05_sample_actions_precomputed_prefix_module(
        **module_kwargs
    )
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def compile_pi05_sample_actions_precomputed_prefix_embs_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile sample_actions from prepared prefix embeddings."""

    module, num_user_inputs = build_pi05_sample_actions_precomputed_prefix_embs_module(
        **module_kwargs
    )
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def compile_pi05_sample_actions_tokens_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile sample_actions from raw images and token ids."""

    module, num_user_inputs = build_pi05_sample_actions_tokens_module(
        **module_kwargs
    )
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def compile_pi05_vision_encoder_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile the SigLIP vision encoder prefix slice to VM bytecode."""

    module, num_user_inputs = build_pi05_vision_encoder_module(**module_kwargs)
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def compile_pi05_paligemma_prefix_encoder_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile the compact PaliGemma prefix encoder to VM bytecode."""

    module, num_user_inputs = build_pi05_paligemma_prefix_encoder_module(**module_kwargs)
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def compile_pi05_paligemma_prefix_kv_encoder_executable(
    *,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseCompileResult:
    """Compile the compact PaliGemma prefix KV-cache encoder to VM bytecode."""

    module, num_user_inputs = build_pi05_paligemma_prefix_kv_encoder_module(
        **module_kwargs
    )
    module = InferStructInfoPass().run(module)
    module = _stamp_single_return_struct_info(module)
    abi_module = module
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=sm_arch).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return Pi05DenoiseCompileResult(
        module=abi_module,
        lowered_module=module,
        executable=exe,
        context=ctx,
        num_user_inputs=num_user_inputs,
    )


def emit_pi05_denoise_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-denoise-fast",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm, abi.json and metadata for the denoise fast path."""

    output_dir = Path(output_dir)
    result = compile_pi05_denoise_executable(sm_arch=sm_arch, **module_kwargs)
    EmitExecutablePass().run(result.executable, str(output_dir))
    EmitABIPass().run(
        result.module,
        result.executable,
        result.context,
        str(output_dir),
        model_name=model_name,
        target="cuda",
        target_arch=target_arch,
    )
    main_fn = result.executable.function_table[-1]
    storage_plan = result.context.get("storage_plan")
    storage_bytes = 0
    if storage_plan is not None:
        storage_bytes = sum(int(entry.size_bytes) for entry in storage_plan.entries)
    return Pi05DenoiseExportSummary(
        artifact_dir=output_dir,
        function_name=main_fn.name,
        num_user_inputs=result.num_user_inputs,
        num_weight_params=max(0, main_fn.num_args - result.num_user_inputs),
        vm_functions=len(result.executable.function_table),
        instructions=len(result.executable.instructions),
        storage_bytes=storage_bytes,
    )


def _emit_compile_result(
    output_dir: Path,
    result: Pi05DenoiseCompileResult,
    *,
    model_name: str,
    target_arch: str,
) -> Pi05DenoiseExportSummary:
    EmitExecutablePass().run(result.executable, str(output_dir))
    EmitABIPass().run(
        result.module,
        result.executable,
        result.context,
        str(output_dir),
        model_name=model_name,
        target="cuda",
        target_arch=target_arch,
    )
    main_fn = result.executable.function_table[-1]
    storage_plan = result.context.get("storage_plan")
    storage_bytes = 0
    if storage_plan is not None:
        storage_bytes = sum(int(entry.size_bytes) for entry in storage_plan.entries)
    return Pi05DenoiseExportSummary(
        artifact_dir=output_dir,
        function_name=main_fn.name,
        num_user_inputs=result.num_user_inputs,
        num_weight_params=max(0, main_fn.num_args - result.num_user_inputs),
        vm_functions=len(result.executable.function_table),
        instructions=len(result.executable.instructions),
        storage_bytes=storage_bytes,
    )


def emit_pi05_denoise_loop_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-denoise-loop-fast",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm, abi.json and metadata for the unrolled denoise loop."""

    output_dir = Path(output_dir)
    result = compile_pi05_denoise_loop_executable(sm_arch=sm_arch, **module_kwargs)
    EmitExecutablePass().run(result.executable, str(output_dir))
    EmitABIPass().run(
        result.module,
        result.executable,
        result.context,
        str(output_dir),
        model_name=model_name,
        target="cuda",
        target_arch=target_arch,
    )
    main_fn = result.executable.function_table[-1]
    storage_plan = result.context.get("storage_plan")
    storage_bytes = 0
    if storage_plan is not None:
        storage_bytes = sum(int(entry.size_bytes) for entry in storage_plan.entries)
    return Pi05DenoiseExportSummary(
        artifact_dir=output_dir,
        function_name=main_fn.name,
        num_user_inputs=result.num_user_inputs,
        num_weight_params=max(0, main_fn.num_args - result.num_user_inputs),
        vm_functions=len(result.executable.function_table),
        instructions=len(result.executable.instructions),
        storage_bytes=storage_bytes,
    )


def emit_pi05_sample_actions_precomputed_prefix_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-sample-actions-precomputed-prefix",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm/ABI for sample_actions with precomputed prefix KV."""

    output_dir = Path(output_dir)
    result = compile_pi05_sample_actions_precomputed_prefix_executable(
        sm_arch=sm_arch,
        **module_kwargs,
    )
    return _emit_compile_result(
        output_dir,
        result,
        model_name=model_name,
        target_arch=target_arch,
    )


def emit_pi05_sample_actions_precomputed_prefix_embs_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-sample-actions-precomputed-prefix-embs",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm/ABI for sample_actions from prefix embeddings."""

    output_dir = Path(output_dir)
    result = compile_pi05_sample_actions_precomputed_prefix_embs_executable(
        sm_arch=sm_arch,
        **module_kwargs,
    )
    return _emit_compile_result(
        output_dir,
        result,
        model_name=model_name,
        target_arch=target_arch,
    )


def emit_pi05_sample_actions_tokens_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-sample-actions-tokens",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm/ABI for sample_actions from images and token ids."""

    output_dir = Path(output_dir)
    result = compile_pi05_sample_actions_tokens_executable(
        sm_arch=sm_arch,
        **module_kwargs,
    )
    return _emit_compile_result(
        output_dir,
        result,
        model_name=model_name,
        target_arch=target_arch,
    )


def emit_pi05_vision_encoder_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-vision-encoder-fast",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm/ABI for the SigLIP vision encoder prefix slice."""

    output_dir = Path(output_dir)
    result = compile_pi05_vision_encoder_executable(
        sm_arch=sm_arch,
        **module_kwargs,
    )
    return _emit_compile_result(
        output_dir,
        result,
        model_name=model_name,
        target_arch=target_arch,
    )


def emit_pi05_paligemma_prefix_encoder_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-paligemma-prefix-encoder-fast",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm/ABI for the compact PaliGemma prefix encoder."""

    output_dir = Path(output_dir)
    result = compile_pi05_paligemma_prefix_encoder_executable(
        sm_arch=sm_arch,
        **module_kwargs,
    )
    return _emit_compile_result(
        output_dir,
        result,
        model_name=model_name,
        target_arch=target_arch,
    )


def emit_pi05_paligemma_prefix_kv_encoder_executable(
    output_dir: str | Path,
    *,
    model_name: str = "openpi0.5-paligemma-prefix-kv-encoder-fast",
    target_arch: str = "sm89",
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Write executable.vm/ABI for the compact PaliGemma prefix KV encoder."""

    output_dir = Path(output_dir)
    result = compile_pi05_paligemma_prefix_kv_encoder_executable(
        sm_arch=sm_arch,
        **module_kwargs,
    )
    return _emit_compile_result(
        output_dir,
        result,
        model_name=model_name,
        target_arch=target_arch,
    )


def export_pi05_denoise_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit denoise bytecode/ABI and optionally install Pi0.5 resources."""

    summary = emit_pi05_denoise_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def export_pi05_denoise_loop_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit unrolled denoise-loop bytecode/ABI and optionally install resources."""

    summary = emit_pi05_denoise_loop_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def export_pi05_sample_actions_precomputed_prefix_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit sample_actions precomputed-prefix bytecode/ABI and resources."""

    summary = emit_pi05_sample_actions_precomputed_prefix_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def export_pi05_sample_actions_precomputed_prefix_embs_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit sample_actions-from-prefix-embs bytecode/ABI and resources."""

    summary = emit_pi05_sample_actions_precomputed_prefix_embs_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def export_pi05_sample_actions_tokens_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit sample_actions-from-images/tokens bytecode/ABI and resources."""

    summary = emit_pi05_sample_actions_tokens_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def export_pi05_vision_encoder_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit the SigLIP vision encoder bytecode/ABI and optional resources."""

    summary = emit_pi05_vision_encoder_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def export_pi05_paligemma_prefix_encoder_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit compact PaliGemma prefix encoder bytecode/ABI and resources."""

    summary = emit_pi05_paligemma_prefix_encoder_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def export_pi05_paligemma_prefix_kv_encoder_artifact(
    *,
    artifact_dir: str | Path,
    weight_package_dir: str | Path | None = None,
    tokenizer_model_path: str | Path | None = None,
    compile_kernels: bool = True,
    nvcc: str | None = None,
    sm_arch: int = 89,
    **module_kwargs: Any,
) -> Pi05DenoiseExportSummary:
    """Emit compact PaliGemma prefix KV encoder bytecode/ABI and resources."""

    summary = emit_pi05_paligemma_prefix_kv_encoder_executable(
        artifact_dir,
        target_arch=f"sm{sm_arch}",
        sm_arch=sm_arch,
        **module_kwargs,
    )
    resource_summary = None
    if weight_package_dir is not None:
        resource_summary = prepare_pi05_artifact(
            weight_package_dir=weight_package_dir,
            artifact_dir=artifact_dir,
            tokenizer_model_path=tokenizer_model_path,
            sm_arch=sm_arch,
            compile_kernels=compile_kernels,
            nvcc=nvcc,
        )
    return Pi05DenoiseExportSummary(
        artifact_dir=summary.artifact_dir,
        function_name=summary.function_name,
        num_user_inputs=summary.num_user_inputs,
        num_weight_params=summary.num_weight_params,
        vm_functions=summary.vm_functions,
        instructions=summary.instructions,
        storage_bytes=summary.storage_bytes,
        resource_summary=resource_summary,
    )


def _stamp_single_return_struct_info(module: IRModule) -> IRModule:
    functions: dict[str, Function] = {}
    for name, fn in module.functions.items():
        ret_si = fn.ret_struct_info
        term = fn.body.entry_block.ops[-1]
        if ret_si is None and isinstance(term, ReturnOp) and len(term.values) == 1:
            ret_si = getattr(term.values[0], "struct_info", None)
        functions[name] = Function(fn.body, ret_si)
    return IRModule(functions)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the Pi0.5 denoise fast-path VM artifact.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("build/pi05_fp8_artifact"))
    parser.add_argument(
        "--entry-kind",
        choices=(
            "step",
            "loop",
            "sample_precomputed_prefix",
            "sample_precomputed_prefix_embs",
            "sample_tokens",
            "vision_encoder",
            "paligemma_prefix_encoder",
            "paligemma_prefix_kv_encoder",
        ),
        default="step",
    )
    parser.add_argument("--weight-package-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-model-path", type=Path, default=None)
    parser.add_argument("--prefix-rows", type=int, default=DEFAULT_PREFIX_ROWS)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument("--max-prompt-len", type=int, default=DEFAULT_MAX_PROMPT_LEN)
    parser.add_argument("--num-views", type=int, default=DEFAULT_VISION_VIEWS)
    parser.add_argument("--sm-arch", type=int, default=89)
    parser.add_argument("--use-static-act-scales", action="store_true")
    parser.add_argument("--no-compile-kernels", action="store_true")
    args = parser.parse_args(argv)

    if args.entry_kind == "loop":
        exporter = export_pi05_denoise_loop_artifact
    elif args.entry_kind == "sample_precomputed_prefix":
        exporter = export_pi05_sample_actions_precomputed_prefix_artifact
    elif args.entry_kind == "sample_precomputed_prefix_embs":
        exporter = export_pi05_sample_actions_precomputed_prefix_embs_artifact
    elif args.entry_kind == "sample_tokens":
        exporter = export_pi05_sample_actions_tokens_artifact
    elif args.entry_kind == "vision_encoder":
        exporter = export_pi05_vision_encoder_artifact
    elif args.entry_kind == "paligemma_prefix_encoder":
        exporter = export_pi05_paligemma_prefix_encoder_artifact
    elif args.entry_kind == "paligemma_prefix_kv_encoder":
        exporter = export_pi05_paligemma_prefix_kv_encoder_artifact
    else:
        exporter = export_pi05_denoise_artifact
    kwargs: dict[str, Any] = {}
    if args.entry_kind != "vision_encoder":
        kwargs["prefix_rows"] = args.prefix_rows
    if args.entry_kind not in (
        "vision_encoder",
        "paligemma_prefix_encoder",
        "paligemma_prefix_kv_encoder",
    ):
        kwargs["action_horizon"] = args.action_horizon
    if args.entry_kind == "sample_tokens":
        kwargs["max_prompt_len"] = args.max_prompt_len
        kwargs["num_views"] = args.num_views
    if args.entry_kind == "vision_encoder":
        kwargs["num_views"] = args.num_views
    kwargs["use_static_act_scales"] = args.use_static_act_scales
    summary = exporter(
        artifact_dir=args.artifact_dir,
        weight_package_dir=args.weight_package_dir,
        tokenizer_model_path=args.tokenizer_model_path,
        compile_kernels=not args.no_compile_kernels,
        sm_arch=args.sm_arch,
        **kwargs,
    )
    print(json.dumps(summary.to_json_obj(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
