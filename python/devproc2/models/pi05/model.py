"""Public Pi0.5 model exports and product compile declaration."""
from __future__ import annotations

from typing import Any

from devproc2.artifact.manifest import PackedBackendRecipe, PackedFuncSpec
from devproc2.export.recipe import CompileRecipe, EntrypointRecipe
from devproc2.models.pi05.config import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INTERMEDIATE_SIZE,
    DEFAULT_MAX_PROMPT_LEN,
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
    DEFAULT_VOCAB_SIZE,
)
from devproc2.models.pi05.graph import (
    PI05FFN,
    PI05Attention,
    PI05DecoderLayer,
    PI05DenoiseLoop,
    PI05DenoiseStep,
    PI05LanguageEmbedding,
    PI05Linear,
    PI05PaliGemmaEncoderLayer,
    PI05PaliGemmaPrefixEncoder,
    PI05SampleActionsFromPrefixEmbeddings,
    PI05SampleActionsFromTokens,
    PI05VisionEncoder,
    PI05VisionEncoderLayer,
    PI05VisionPatchEmbedding,
)
from devproc2.nn import Module, ScalarSpec, TensorSpec

PI05_MODEL_ID = "openpi0.5"


def _opt(options: dict[str, Any] | object, name: str, default: Any) -> Any:
    if isinstance(options, dict):
        return options.get(name, default)
    return default


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
    """Runtime ABI for full Pi0.5 sample_actions from images and token ids."""

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


def _sample_tokens_module(options: dict[str, Any]) -> Module:
    return PI05SampleActionsFromTokens(
        num_layers=int(_opt(options, "num_layers", DEFAULT_NUM_LAYERS)),
        num_views=int(_opt(options, "num_views", DEFAULT_VISION_VIEWS)),
        image_size=int(_opt(options, "image_size", DEFAULT_IMAGE_SIZE)),
        patch_size=int(_opt(options, "patch_size", DEFAULT_PATCH_SIZE)),
        image_channels=int(_opt(options, "image_channels", DEFAULT_IMAGE_CHANNELS)),
        vision_layers=int(_opt(options, "vision_layers", DEFAULT_VISION_LAYERS)),
        vision_hidden_size=int(_opt(options, "vision_hidden_size", DEFAULT_VISION_HIDDEN_SIZE)),
        vision_intermediate_size=int(
            _opt(options, "vision_intermediate_size", DEFAULT_VISION_INTERMEDIATE_SIZE)
        ),
        vision_heads=int(_opt(options, "vision_heads", DEFAULT_VISION_HEADS)),
        vocab_size=int(_opt(options, "vocab_size", DEFAULT_VOCAB_SIZE)),
        max_prompt_len=int(_opt(options, "max_prompt_len", DEFAULT_MAX_PROMPT_LEN)),
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


def _sample_tokens_specs(options: dict[str, Any]) -> dict[str, object]:
    return pi05_sample_actions_tokens_input_specs(
        action_horizon=int(_opt(options, "action_horizon", DEFAULT_ACTION_HORIZON)),
        action_dim=int(_opt(options, "action_dim", DEFAULT_ACTION_DIM)),
        num_views=int(_opt(options, "num_views", DEFAULT_VISION_VIEWS)),
        image_size=int(_opt(options, "image_size", DEFAULT_IMAGE_SIZE)),
        image_channels=int(_opt(options, "image_channels", DEFAULT_IMAGE_CHANNELS)),
        max_prompt_len=int(_opt(options, "max_prompt_len", DEFAULT_MAX_PROMPT_LEN)),
        prefix_rows=int(_opt(options, "prefix_rows", DEFAULT_PREFIX_ROWS)),
        head_dim=int(_opt(options, "head_dim", DEFAULT_HEAD_DIM)),
        device=str(_opt(options, "device", "cuda")),
    )


pi05_cuda_backend = PackedBackendRecipe(
    name="pi05.cuda",
    kind="compiled_packed_backend",
    sources=(
        "python/devproc2/models/pi05/cuda/backends/fp8_gemm/cublaslt_runner.cc",
        "python/devproc2/models/pi05/cuda/backends/fp8_gemm/cutlass_fp8_gemm_sm89.h",
        "python/devproc2/models/pi05/cuda/backends/fp8_gemm/cutlass_fp8_gemm_sm89.cu",
        "python/devproc2/models/pi05/cuda/fa2/fa2_wrapper.cu",
    ),
    include_dirs=(
        "runtime/include",
        "python/devproc2/models/pi05/cuda/backends/fp8_gemm",
        "python/devproc2/models/pi05/cuda/fa2",
        "python/devproc2/models/pi05/cuda/fa2/flash_attn_2_src",
    ),
    compile_definitions=(
        "DEVPROC2_WITH_CUDA",
        "DEVPROC2_WITH_CUTLASS",
        "DEVPROC2_WITH_PI05_FA2",
    ),
    link_libraries=("CUDA::cudart", "CUDA::cublasLt"),
    targets=("sm89",),
    register_symbol="devproc2_register_pi05_cuda_backend",
    packed_funcs=(
        PackedFuncSpec("pi05.cuda.fp8_nn_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.fp8_nt_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.fp8_nn_bf16_accum", device="cuda"),
        PackedFuncSpec("pi05.cuda.fp8_nt_bf16_accum", device="cuda"),
        PackedFuncSpec("pi05.cuda.bf16_nn_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.bf16_nt_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.fa2_bf16", device="cuda"),
        PackedFuncSpec("pi05.cuda.fa2_bf16_batched", device="cuda"),
    ),
)


sample_tokens = EntrypointRecipe(
    name="sample_tokens",
    model_id=PI05_MODEL_ID,
    build_module=_sample_tokens_module,
    input_specs=_sample_tokens_specs,
    model_name="openpi0.5-sample-actions-tokens",
    packed_backends=(pi05_cuda_backend,),
)

pi05_recipe = CompileRecipe(
    model_id=PI05_MODEL_ID,
    entrypoints={sample_tokens.name: sample_tokens},
)


PI05_MODEL = pi05_recipe


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
    "PI05_MODEL",
    "PI05_MODEL_ID",
    "DEFAULT_PREFIX_ROWS",
    "DEFAULT_ACTION_HORIZON",
    "DEFAULT_ACTION_DIM",
    "DEFAULT_NUM_STEPS",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_INTERMEDIATE_SIZE",
    "DEFAULT_NUM_Q_HEADS",
    "DEFAULT_NUM_KV_HEADS",
    "DEFAULT_HEAD_DIM",
    "DEFAULT_PREFIX_HIDDEN_SIZE",
    "DEFAULT_PREFIX_INTERMEDIATE_SIZE",
    "DEFAULT_MAX_PROMPT_LEN",
    "DEFAULT_VOCAB_SIZE",
    "DEFAULT_VISION_LAYERS",
    "DEFAULT_VISION_VIEWS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_IMAGE_CHANNELS",
    "DEFAULT_VISION_HIDDEN_SIZE",
    "DEFAULT_VISION_INTERMEDIATE_SIZE",
    "DEFAULT_VISION_HEADS",
    "DEFAULT_VISION_OUTPUT_SIZE",
    "pi05_cuda_backend",
    "pi05_recipe",
    "pi05_sample_actions_tokens_input_specs",
    "sample_tokens",
]
