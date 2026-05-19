"""Typed Pi0.5 configuration defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
DEFAULT_TOKENIZER_MODEL = None
DEFAULT_PROFILE = "pi05_libero_base_3v200"


@dataclass(frozen=True)
class PI05ShapeConfig:
    prefix_rows: int = DEFAULT_PREFIX_ROWS
    action_horizon: int = DEFAULT_ACTION_HORIZON
    action_dim: int = DEFAULT_ACTION_DIM
    num_steps: int = DEFAULT_NUM_STEPS
    num_layers: int = DEFAULT_NUM_LAYERS
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE
    num_q_heads: int = DEFAULT_NUM_Q_HEADS
    num_kv_heads: int = DEFAULT_NUM_KV_HEADS
    head_dim: int = DEFAULT_HEAD_DIM
    prefix_hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE
    prefix_intermediate_size: int = DEFAULT_PREFIX_INTERMEDIATE_SIZE
    max_prompt_len: int = DEFAULT_MAX_PROMPT_LEN
    vocab_size: int = DEFAULT_VOCAB_SIZE
    vision_layers: int = DEFAULT_VISION_LAYERS
    vision_views: int = DEFAULT_VISION_VIEWS
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    image_channels: int = DEFAULT_IMAGE_CHANNELS
    vision_hidden_size: int = DEFAULT_VISION_HIDDEN_SIZE
    vision_intermediate_size: int = DEFAULT_VISION_INTERMEDIATE_SIZE
    vision_heads: int = DEFAULT_VISION_HEADS
    vision_output_size: int = DEFAULT_VISION_OUTPUT_SIZE


@dataclass(frozen=True)
class PI05LayoutConfig:
    fp8_layout: str = "nk"
    weight_layout: str = "row_major"
    qkv_layout: str = "q_k_v"


@dataclass(frozen=True)
class PI05EntrypointConfig:
    name: str = "sample_tokens"
    compile_mode: str = "fast"
    normal_method: str = "forward"
    fast_method: str = "forward_fast"


@dataclass(frozen=True)
class PI05KernelConfig:
    source_file: str = "cuda/pi05_kernels.cu"
    sm_arch: int = 89
    extra_nvcc_flags: tuple[str, ...] = ("--std=c++17",)


@dataclass(frozen=True)
class PI05ArtifactRecipeConfig:
    tokenizer_model_path: str | None = DEFAULT_TOKENIZER_MODEL
    compile_kernels: bool = True
    weight_package_dir: str | None = None


@dataclass(frozen=True)
class PI05Config:
    """Single Pi0.5 model configuration entrypoint.

    Precision changes may select different DSL graph variants. Target changes
    must stay behind `ops.py`, lowering, or runtime registry boundaries.
    """

    shape: PI05ShapeConfig = field(default_factory=PI05ShapeConfig)
    precision: str = "fp8"
    target: str = "rtx4090_sm89"
    layout: PI05LayoutConfig = field(default_factory=PI05LayoutConfig)
    entrypoint: PI05EntrypointConfig = field(default_factory=PI05EntrypointConfig)
    kernel: PI05KernelConfig = field(default_factory=PI05KernelConfig)
    artifact: PI05ArtifactRecipeConfig = field(default_factory=PI05ArtifactRecipeConfig)

    def __post_init__(self) -> None:
        if self.precision not in {"fp8", "bf16", "fp16"}:
            raise ValueError("PI05Config.precision must be one of: fp8, bf16, fp16")
        if self.entrypoint.compile_mode not in {"fast", "normal"}:
            raise ValueError("PI05Config.entrypoint.compile_mode must be 'fast' or 'normal'")
        if self.layout.fp8_layout not in {"nk", "kn"}:
            raise ValueError("PI05Config.layout.fp8_layout must be 'nk' or 'kn'")

    @classmethod
    def default_profile(cls) -> str:
        return DEFAULT_PROFILE

    @classmethod
    def profiles(cls) -> dict[str, PI05Config]:
        return {
            DEFAULT_PROFILE: cls(),
        }

    @classmethod
    def for_profile(cls, name: str) -> PI05Config:
        profiles = cls.profiles()
        try:
            return profiles[name]
        except KeyError as exc:
            available = ", ".join(sorted(profiles))
            raise KeyError(f"unknown Pi0.5 profile {name!r}; available: {available}") from exc

    def to_options(self) -> dict[str, object]:
        shape = self.shape
        return {
            "prefix_rows": shape.prefix_rows,
            "action_horizon": shape.action_horizon,
            "action_dim": shape.action_dim,
            "num_steps": shape.num_steps,
            "num_layers": shape.num_layers,
            "hidden_size": shape.hidden_size,
            "intermediate_size": shape.intermediate_size,
            "num_q_heads": shape.num_q_heads,
            "num_kv_heads": shape.num_kv_heads,
            "head_dim": shape.head_dim,
            "prefix_hidden_size": shape.prefix_hidden_size,
            "prefix_intermediate_size": shape.prefix_intermediate_size,
            "max_prompt_len": shape.max_prompt_len,
            "vocab_size": shape.vocab_size,
            "vision_layers": shape.vision_layers,
            "num_views": shape.vision_views,
            "image_size": shape.image_size,
            "patch_size": shape.patch_size,
            "image_channels": shape.image_channels,
            "vision_hidden_size": shape.vision_hidden_size,
            "vision_intermediate_size": shape.vision_intermediate_size,
            "vision_heads": shape.vision_heads,
            "output_size": shape.vision_output_size,
            "compile_mode": self.entrypoint.compile_mode,
        }

    def to_json_obj(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "PI05ArtifactRecipeConfig",
    "PI05Config",
    "PI05EntrypointConfig",
    "PI05KernelConfig",
    "PI05LayoutConfig",
    "PI05ShapeConfig",
    "DEFAULT_PROFILE",
]
