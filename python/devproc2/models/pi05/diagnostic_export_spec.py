"""Diagnostic Pi0.5 export declarations.

These entrypoints are useful for oracle checks, perf bisection, and staged
bring-up. Product builds should use ``model.pi05_recipe``.
"""
from __future__ import annotations

from typing import Any

from devproc2.export.recipe import CompileRecipe, EntrypointRecipe
from devproc2.models.pi05.config import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INTERMEDIATE_SIZE,
    DEFAULT_NUM_KV_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_Q_HEADS,
    DEFAULT_NUM_STEPS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PREFIX_HIDDEN_SIZE,
    DEFAULT_PREFIX_INTERMEDIATE_SIZE,
    DEFAULT_PREFIX_ROWS,
    DEFAULT_VISION_HEADS,
    DEFAULT_VISION_HIDDEN_SIZE,
    DEFAULT_VISION_INTERMEDIATE_SIZE,
    DEFAULT_VISION_LAYERS,
    DEFAULT_VISION_OUTPUT_SIZE,
    DEFAULT_VISION_VIEWS,
)
from devproc2.models.pi05.graph import (
    PI05DenoiseLoop,
    PI05DenoiseStep,
    PI05PaliGemmaPrefixEncoder,
    PI05SampleActionsFromPrefixEmbeddings,
    PI05VisionEncoder,
)
from devproc2.models.pi05.model import PI05_MODEL_ID, pi05_cuda_backend
from devproc2.nn import Module, ScalarSpec, TensorSpec


def _opt(options: dict[str, Any] | object, name: str, default: Any) -> Any:
    if isinstance(options, dict):
        return options.get(name, default)
    return default


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
    """Fixed-shape ABI specs for one diagnostic denoise step."""

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
    """Fixed-shape ABI specs for the diagnostic unrolled denoise loop."""

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
    """Diagnostic ABI for sample_actions with caller-supplied prefix KV."""

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
    """Diagnostic ABI for sample_actions with caller-supplied prefix embeddings."""

    return {
        "noise_f32": TensorSpec((action_horizon, action_dim), "float32", device=device),
        "prefix_embs": TensorSpec((prefix_rows, prefix_hidden_size), "bfloat16", device=device),
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
    """Diagnostic ABI specs for the SigLIP vision encoder slice."""

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
    """Diagnostic ABI specs for the compact PaliGemma prefix encoder."""

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
    """Diagnostic ABI specs for compact PaliGemma prefix KV materialization."""

    return {
        "prefix_embs": TensorSpec((prefix_rows, hidden_size), "bfloat16", device=device),
        "prefix_valid_rows": ScalarSpec("int64"),
        "rope_interleaved": TensorSpec((prefix_rows, head_dim), "bfloat16", device=device),
    }


def _denoise_step_module(options: dict[str, Any]) -> Module:
    return PI05DenoiseStep(
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_HIDDEN_SIZE)),
        intermediate_size=int(_opt(options, "intermediate_size", DEFAULT_INTERMEDIATE_SIZE)),
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        num_steps=int(_opt(options, "num_steps", DEFAULT_NUM_STEPS)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        num_q_heads=int(_opt(options, "num_q_heads", DEFAULT_NUM_Q_HEADS)),
        num_kv_heads=int(_opt(options, "num_kv_heads", DEFAULT_NUM_KV_HEADS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
        use_static_act_scales=bool(_opt(options, "use_static_act_scales", False)),
    )


def _denoise_loop_module(options: dict[str, Any]) -> Module:
    return PI05DenoiseLoop(
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_HIDDEN_SIZE)),
        intermediate_size=int(_opt(options, "intermediate_size", DEFAULT_INTERMEDIATE_SIZE)),
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        num_steps=int(_opt(options, "num_steps", DEFAULT_NUM_STEPS)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        num_q_heads=int(_opt(options, "num_q_heads", DEFAULT_NUM_Q_HEADS)),
        num_kv_heads=int(_opt(options, "num_kv_heads", DEFAULT_NUM_KV_HEADS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
        use_static_act_scales=bool(_opt(options, "use_static_act_scales", False)),
    )


def _sample_prefix_embs_module(options: dict[str, Any]) -> Module:
    return PI05SampleActionsFromPrefixEmbeddings(
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        prefix_hidden_size=int(_opt(options, "prefix_hidden_size", DEFAULT_PREFIX_HIDDEN_SIZE)),
        prefix_intermediate_size=int(
            _opt(options, "prefix_intermediate_size", DEFAULT_PREFIX_INTERMEDIATE_SIZE)
        ),
        decoder_hidden_size=int(_opt(options, "decoder_hidden_size", DEFAULT_HIDDEN_SIZE)),
        decoder_intermediate_size=int(
            _opt(options, "decoder_intermediate_size", DEFAULT_INTERMEDIATE_SIZE)
        ),
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        num_steps=int(_opt(options, "num_steps", DEFAULT_NUM_STEPS)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        num_q_heads=int(_opt(options, "num_q_heads", DEFAULT_NUM_Q_HEADS)),
        num_kv_heads=int(_opt(options, "num_kv_heads", DEFAULT_NUM_KV_HEADS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
        use_static_act_scales=bool(_opt(options, "use_static_act_scales", False)),
    )


def _vision_encoder_module(options: dict[str, Any]) -> Module:
    return PI05VisionEncoder(
        num_layers=int(_opt(options, "num_layers", DEFAULT_VISION_LAYERS)),
        num_views=int(_opt(options, "num_views", DEFAULT_VISION_VIEWS)),
        image_size=int(_opt(options, "image_size", DEFAULT_IMAGE_SIZE)),
        patch_size=int(_opt(options, "patch_size", DEFAULT_PATCH_SIZE)),
        in_channels=int(_opt(options, "image_channels", DEFAULT_IMAGE_CHANNELS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_VISION_HIDDEN_SIZE)),
        intermediate_size=int(_opt(options, "intermediate_size", DEFAULT_VISION_INTERMEDIATE_SIZE)),
        num_heads=int(_opt(options, "num_heads", DEFAULT_VISION_HEADS)),
        output_size=int(_opt(options, "output_size", DEFAULT_VISION_OUTPUT_SIZE)),
        device=str(_opt(options, "device", "cuda")),
        use_static_act_scales=bool(_opt(options, "use_static_act_scales", False)),
    )


def _prefix_encoder_module(options: dict[str, Any]) -> Module:
    return PI05PaliGemmaPrefixEncoder(
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_PREFIX_HIDDEN_SIZE)),
        intermediate_size=int(_opt(options, "intermediate_size", DEFAULT_PREFIX_INTERMEDIATE_SIZE)),
        num_q_heads=int(_opt(options, "num_q_heads", DEFAULT_NUM_Q_HEADS)),
        num_kv_heads=int(_opt(options, "num_kv_heads", DEFAULT_NUM_KV_HEADS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
        use_static_act_scales=bool(_opt(options, "use_static_act_scales", False)),
    )


def _denoise_step_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_denoise_input_specs(
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        prefix_rows=int(_opt(options, "prefix_rows", DEFAULT_PREFIX_ROWS)),
        num_steps=int(_opt(options, "num_steps", DEFAULT_NUM_STEPS)),
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_HIDDEN_SIZE)),
        num_kv_heads=int(_opt(options, "num_kv_heads", DEFAULT_NUM_KV_HEADS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
    )


def _denoise_loop_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_denoise_loop_input_specs(
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        prefix_rows=int(_opt(options, "prefix_rows", DEFAULT_PREFIX_ROWS)),
        num_steps=int(_opt(options, "num_steps", DEFAULT_NUM_STEPS)),
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_HIDDEN_SIZE)),
        num_kv_heads=int(_opt(options, "num_kv_heads", DEFAULT_NUM_KV_HEADS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
    )


def _sample_prefix_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_sample_actions_precomputed_prefix_input_specs(
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        prefix_rows=int(_opt(options, "prefix_rows", DEFAULT_PREFIX_ROWS)),
        num_steps=int(_opt(options, "num_steps", DEFAULT_NUM_STEPS)),
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_HIDDEN_SIZE)),
        num_kv_heads=int(_opt(options, "num_kv_heads", DEFAULT_NUM_KV_HEADS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
    )


def _sample_prefix_embs_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_sample_actions_precomputed_prefix_embs_input_specs(
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        prefix_rows=int(_opt(options, "prefix_rows", DEFAULT_PREFIX_ROWS)),
        prefix_hidden_size=int(_opt(options, "prefix_hidden_size", DEFAULT_PREFIX_HIDDEN_SIZE)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
    )


def _vision_encoder_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_vision_encoder_input_specs(
        num_views=int(_opt(options, "num_views", DEFAULT_VISION_VIEWS)),
        image_size=int(_opt(options, "image_size", DEFAULT_IMAGE_SIZE)),
        image_channels=int(_opt(options, "image_channels", DEFAULT_IMAGE_CHANNELS)),
        device=str(_opt(options, "device", "cuda")),
    )


def _prefix_encoder_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_paligemma_prefix_encoder_input_specs(
        prefix_rows=int(_opt(options, "prefix_rows", DEFAULT_PREFIX_ROWS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_PREFIX_HIDDEN_SIZE)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
    )


def _prefix_kv_encoder_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_paligemma_prefix_kv_encoder_input_specs(
        prefix_rows=int(_opt(options, "prefix_rows", DEFAULT_PREFIX_ROWS)),
        hidden_size=int(_opt(options, "hidden_size", DEFAULT_PREFIX_HIDDEN_SIZE)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
    )


step = EntrypointRecipe(
    name="step",
    model_id=PI05_MODEL_ID,
    build_module=_denoise_step_module,
    input_specs=_denoise_step_specs,
    model_name="openpi0.5-denoise-fast",
    packed_backends=(pi05_cuda_backend,),
)
loop = EntrypointRecipe(
    name="loop",
    model_id=PI05_MODEL_ID,
    build_module=_denoise_loop_module,
    input_specs=_denoise_loop_specs,
    model_name="openpi0.5-denoise-loop-fast",
    packed_backends=(pi05_cuda_backend,),
)
sample_precomputed_prefix = EntrypointRecipe(
    name="sample_precomputed_prefix",
    model_id=PI05_MODEL_ID,
    build_module=_denoise_loop_module,
    input_specs=_sample_prefix_specs,
    model_name="openpi0.5-sample-actions-precomputed-prefix",
    packed_backends=(pi05_cuda_backend,),
)
sample_precomputed_prefix_embs = EntrypointRecipe(
    name="sample_precomputed_prefix_embs",
    model_id=PI05_MODEL_ID,
    build_module=_sample_prefix_embs_module,
    input_specs=_sample_prefix_embs_specs,
    model_name="openpi0.5-sample-actions-precomputed-prefix-embs",
    packed_backends=(pi05_cuda_backend,),
)
vision_encoder = EntrypointRecipe(
    name="vision_encoder",
    model_id=PI05_MODEL_ID,
    build_module=_vision_encoder_module,
    input_specs=_vision_encoder_specs,
    model_name="openpi0.5-vision-encoder-fast",
    packed_backends=(pi05_cuda_backend,),
)
paligemma_prefix_encoder = EntrypointRecipe(
    name="paligemma_prefix_encoder",
    model_id=PI05_MODEL_ID,
    build_module=_prefix_encoder_module,
    input_specs=_prefix_encoder_specs,
    model_name="openpi0.5-paligemma-prefix-encoder-fast",
    packed_backends=(pi05_cuda_backend,),
)
paligemma_prefix_kv_encoder = EntrypointRecipe(
    name="paligemma_prefix_kv_encoder",
    model_id=PI05_MODEL_ID,
    build_module=_prefix_encoder_module,
    input_specs=_prefix_kv_encoder_specs,
    normal_method="materialize_kv",
    fast_method="materialize_kv_fast",
    model_name="openpi0.5-paligemma-prefix-kv-encoder-fast",
    packed_backends=(pi05_cuda_backend,),
)

pi05_diagnostic_recipe = CompileRecipe(
    model_id=PI05_MODEL_ID,
    entrypoints={
        item.name: item
        for item in (
            step,
            loop,
            sample_precomputed_prefix,
            sample_precomputed_prefix_embs,
            vision_encoder,
            paligemma_prefix_encoder,
            paligemma_prefix_kv_encoder,
        )
    },
)


__all__ = [
    "loop",
    "paligemma_prefix_encoder",
    "paligemma_prefix_kv_encoder",
    "pi05_denoise_input_specs",
    "pi05_denoise_loop_input_specs",
    "pi05_diagnostic_recipe",
    "pi05_paligemma_prefix_encoder_input_specs",
    "pi05_paligemma_prefix_kv_encoder_input_specs",
    "pi05_sample_actions_precomputed_prefix_embs_input_specs",
    "pi05_sample_actions_precomputed_prefix_input_specs",
    "pi05_vision_encoder_input_specs",
    "sample_precomputed_prefix",
    "sample_precomputed_prefix_embs",
    "step",
    "vision_encoder",
]
