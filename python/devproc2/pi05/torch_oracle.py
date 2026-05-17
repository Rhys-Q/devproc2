"""Dump PyTorch Pi0.5 denoise-step oracle tensors for devproc2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


DEFAULT_OPENPI_ROOT = Path("/root/autodl-tmp/openpi")
DEFAULT_CKPT = Path("/root/autodl-tmp/tools/pi05-pytorch-base")
DEFAULT_DUMP_DIR = Path("/root/autodl-tmp/openpi/outputs/pi05_torch_infer")
DEFAULT_OUT_DIR = Path("build/pi05_torch_denoise_oracle")


def dump_torch_denoise_oracle(
    *,
    openpi_root: str | Path = DEFAULT_OPENPI_ROOT,
    ckpt_dir: str | Path = DEFAULT_CKPT,
    input_dump_dir: str | Path = DEFAULT_DUMP_DIR,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    precision: str = "bfloat16",
    device: str = "cuda",
    example_index: int = 0,
    num_steps: int = 10,
    steps: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    """Run the PyTorch policy and dump one-example denoise-step oracle data.

    The prefix cache and RoPE table are written once to ``prefix_inputs.npz``.
    Each requested denoise step is written to ``step_XXX.npz``.
    BF16 tensors are stored as uint16 bit patterns because NumPy has no native
    bfloat16 dtype.
    """

    _install_openpi_paths(Path(openpi_root))

    import torch
    from openpi.models import model as _model
    from openpi.models import pi0_config
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from scripts import dump_pi05_torch_infer

    if precision not in {"bfloat16", "float16"}:
        raise ValueError("precision must be 'bfloat16' or 'float16'")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if any(step < 0 or step >= num_steps for step in steps):
        raise ValueError(f"steps must be in [0, {num_steps})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt_dir = Path(ckpt_dir)
    input_dump_dir = Path(input_dump_dir)

    model_config = pi0_config.Pi0Config(pi05=True, dtype=precision, pytorch_compile_mode=None)
    input_transform = dump_pi05_torch_infer._make_transform(model_config)  # noqa: SLF001
    inputs = dump_pi05_torch_infer._load_inputs(input_dump_dir / "inputs.npz")  # noqa: SLF001
    model = dump_pi05_torch_infer._make_model(ckpt_dir, precision, device_obj)  # noqa: SLF001

    try:
        transformed = input_transform(
            dump_pi05_torch_infer._example_from_inputs(inputs, example_index)  # noqa: SLF001
        )
        batch = dump_pi05_torch_infer._to_torch_batch(transformed, device_obj)  # noqa: SLF001
        observation = _model.Observation.from_dict(batch)

        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
            observation,
            train=False,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
        )
        prefix_valid_rows = int(prefix_pad_masks[0].sum().item())
        prefix_seq_len = int(prefix_pad_masks.shape[1])
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        prefix_embs_bf16 = _to_bf16_bits(prefix_embs[0])
        prefix_rope_interleaved = _build_prefix_rope_interleaved_bf16(
            model,
            position_ids=prefix_position_ids,
            head_dim=model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.head_dim,
        )

        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_k_cache, prefix_v_cache = _pack_prefix_cache_bf16(past_key_values, prefix_seq_len)
        rope_interleaved = _build_rope_interleaved_bf16(
            model,
            prefix_valid_rows=prefix_valid_rows,
            action_horizon=model_config.action_horizon,
            head_dim=prefix_k_cache.shape[-1],
        )

        np.savez_compressed(
            out_dir / "prefix_inputs.npz",
            prefix_k_cache_bf16=prefix_k_cache,
            prefix_v_cache_bf16=prefix_v_cache,
            prefix_embs_bf16=prefix_embs_bf16,
            prefix_rope_interleaved_bf16=prefix_rope_interleaved,
            rope_interleaved_bf16=rope_interleaved,
            prefix_valid_rows=np.asarray(prefix_valid_rows, dtype=np.int64),
            prefix_seq_len=np.asarray(prefix_seq_len, dtype=np.int64),
            prefix_pad_mask=prefix_pad_masks[0].detach().cpu().numpy().astype(np.bool_),
        )
        _write_raw(raw_dir / "prefix_embs_bf16.bin", prefix_embs_bf16)
        _write_raw(raw_dir / "prefix_rope_interleaved_bf16.bin", prefix_rope_interleaved)
        _write_raw(raw_dir / "prefix_k_cache_bf16.bin", prefix_k_cache)
        _write_raw(raw_dir / "prefix_v_cache_bf16.bin", prefix_v_cache)
        _write_raw(raw_dir / "rope_interleaved_bf16.bin", rope_interleaved)

        wanted_steps = set(steps)
        noise = torch.from_numpy(inputs["noise"][example_index]).to(device_obj)[None, ...]
        x_t = noise.to(torch.float32)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device_obj)
        time = torch.tensor(1.0, dtype=torch.float32, device=device_obj)
        dumped_steps: list[int] = []
        for step in range(num_steps):
            expanded_time = time.expand(x_t.shape[0])
            v_t = model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            delta = dt * v_t
            x_next = x_t + delta
            if step in wanted_steps:
                actions_f32 = _to_f32_numpy(x_t[0])
                target_v_t_f32 = _to_f32_numpy(v_t[0])
                target_delta_f32 = _to_f32_numpy(delta[0])
                target_delta_bf16 = _to_bf16_bits(delta[0])
                x_next_f32 = _to_f32_numpy(x_next[0])
                np.savez_compressed(
                    out_dir / f"step_{step:03d}.npz",
                    actions_f32=actions_f32,
                    step=np.asarray(step, dtype=np.int64),
                    time=np.asarray(float(time.detach().cpu().item()), dtype=np.float32),
                    target_v_t_f32=target_v_t_f32,
                    target_delta_f32=target_delta_f32,
                    target_delta_bf16=target_delta_bf16,
                    x_next_f32=x_next_f32,
                )
                step_raw_dir = raw_dir / f"step_{step:03d}"
                step_raw_dir.mkdir(exist_ok=True)
                _write_raw(step_raw_dir / "actions_f32.bin", actions_f32)
                _write_raw(step_raw_dir / "target_v_t_f32.bin", target_v_t_f32)
                _write_raw(step_raw_dir / "target_delta_f32.bin", target_delta_f32)
                _write_raw(step_raw_dir / "target_delta_bf16.bin", target_delta_bf16)
                _write_raw(step_raw_dir / "x_next_f32.bin", x_next_f32)
                dumped_steps.append(step)
            x_t = x_next
            time = time + dt

        metadata = {
            "format": "devproc2.pi05.torch_denoise_oracle",
            "format_version": 1,
            "openpi_root": str(openpi_root),
            "ckpt_dir": str(ckpt_dir),
            "input_dump_dir": str(input_dump_dir),
            "precision": precision,
            "device": str(device_obj),
            "example_index": example_index,
            "num_steps": num_steps,
            "dumped_steps": dumped_steps,
            "prefix_valid_rows": prefix_valid_rows,
            "prefix_seq_len": prefix_seq_len,
            "prefix_embs_shape": list(prefix_embs_bf16.shape),
            "prefix_k_cache_shape": list(prefix_k_cache.shape),
            "prefix_rope_interleaved_shape": list(prefix_rope_interleaved.shape),
            "rope_interleaved_shape": list(rope_interleaved.shape),
            "raw_dir": "raw",
            "raw_files": {
                "prefix_embs_bf16": "raw/prefix_embs_bf16.bin",
                "prefix_rope_interleaved_bf16": "raw/prefix_rope_interleaved_bf16.bin",
                "prefix_k_cache_bf16": "raw/prefix_k_cache_bf16.bin",
                "prefix_v_cache_bf16": "raw/prefix_v_cache_bf16.bin",
                "rope_interleaved_bf16": "raw/rope_interleaved_bf16.bin",
            },
            "action_horizon": model_config.action_horizon,
            "action_dim": model_config.action_dim,
            "torch": torch.__version__,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    finally:
        del model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()


def _install_openpi_paths(openpi_root: Path) -> None:
    root = openpi_root.resolve()
    for path in (root, root / "src"):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def _pack_prefix_cache_bf16(past_key_values: Any, prefix_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not hasattr(past_key_values, "key_cache") or not hasattr(past_key_values, "value_cache"):
        raise TypeError("past_key_values must expose key_cache/value_cache lists")
    k_layers = []
    v_layers = []
    for key, value in zip(past_key_values.key_cache, past_key_values.value_cache, strict=True):
        # Torch cache layout is [B, num_kv_heads, S, head_dim]. devproc2 uses
        # [layers, S, num_kv_heads, head_dim].
        k = key[0, :, :prefix_seq_len, :].permute(1, 0, 2).contiguous()
        v = value[0, :, :prefix_seq_len, :].permute(1, 0, 2).contiguous()
        k_layers.append(_interleave_rope_pairs(k))
        v_layers.append(v)
    k_cache = torch.stack(k_layers, dim=0)
    v_cache = torch.stack(v_layers, dim=0)
    return _to_bf16_bits(k_cache), _to_bf16_bits(v_cache)


def _interleave_rope_pairs(tensor: Any) -> Any:
    """Convert rotate-half `[lo..., hi...]` head_dim layout to FlashRT pairs."""

    import torch

    if tensor.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even for RoPE pair interleave")
    half = tensor.shape[-1] // 2
    return torch.stack((tensor[..., :half], tensor[..., half:]), dim=-1).reshape(*tensor.shape).contiguous()


def _build_rope_interleaved_bf16(
    model: Any,
    *,
    prefix_valid_rows: int,
    action_horizon: int,
    head_dim: int,
) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    position_ids = torch.arange(
        prefix_valid_rows,
        prefix_valid_rows + action_horizon,
        dtype=torch.long,
        device=device,
    )[None, :]
    dummy = torch.zeros((1, action_horizon, head_dim), dtype=dtype, device=device)
    cos, sin = model.paligemma_with_expert.gemma_expert.model.rotary_emb(dummy, position_ids)
    half = head_dim // 2
    rope = torch.empty((action_horizon, head_dim), dtype=cos.dtype, device=device)
    rope[:, 0::2] = cos[0, :, :half]
    rope[:, 1::2] = sin[0, :, :half]
    return _to_bf16_bits(rope)


def _build_prefix_rope_interleaved_bf16(
    model: Any,
    *,
    position_ids: Any,
    head_dim: int,
) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    seq_len = int(position_ids.shape[1])
    dummy = torch.zeros((1, seq_len, head_dim), dtype=dtype, device=device)
    rotary = model.paligemma_with_expert.paligemma.model.language_model.rotary_emb
    cos, sin = rotary(dummy, position_ids)
    half = head_dim // 2
    rope = torch.empty((seq_len, head_dim), dtype=cos.dtype, device=device)
    rope[:, 0::2] = cos[0, :, :half]
    rope[:, 1::2] = sin[0, :, :half]
    return _to_bf16_bits(rope)


def _to_f32_numpy(tensor: Any) -> np.ndarray:
    import torch

    return tensor.detach().to(dtype=torch.float32).cpu().numpy().copy()


def _to_bf16_bits(tensor: Any) -> np.ndarray:
    import torch

    return tensor.detach().contiguous().to(torch.bfloat16).view(torch.uint16).cpu().numpy().copy()


def _write_raw(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array)
    array.tofile(path)


def _parse_steps(value: str, num_steps: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(num_steps))
    steps = tuple(int(part) for part in value.split(",") if part != "")
    if not steps:
        raise ValueError("steps must not be empty")
    return steps


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dump Pi0.5 PyTorch denoise-step oracle tensors.")
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--input-dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--steps", default="0")
    args = parser.parse_args(argv)

    steps = _parse_steps(args.steps, args.num_steps)
    metadata = dump_torch_denoise_oracle(
        openpi_root=args.openpi_root,
        ckpt_dir=args.ckpt,
        input_dump_dir=args.input_dump_dir,
        out_dir=args.out_dir,
        precision=args.precision,
        device=args.device,
        example_index=args.example_index,
        num_steps=args.num_steps,
        steps=steps,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
