"""Pi0.5 weight package conversion and FP8 quantization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from devproc2.quantization import common_fp8_scale


BF16 = "bfloat16"
FP16 = "float16"
FP32 = "float32"
FP8_E4M3 = "fp8_e4m3"

VIS_L = 27
VIS_D = 1152
VIS_H = 4304
ENC_L = 18
ENC_D = 2048
ENC_H = 16384
DEC_L = 18
DEC_D = 1024
DEC_H = 4096
ACTION_DIM = 32
ACTION_HORIZON = 50
NUM_STEPS_DEFAULT = 10


def pi05_act_scale_name(logical_name: str, layer_idx: int | None = None) -> str:
    suffix = f"_{layer_idx}" if layer_idx is not None else ""
    return f"act_scale.{logical_name}{suffix}"


@dataclass(frozen=True)
class QuantSpec:
    scheme: str
    storage_dtype: str
    compute_dtype: str
    scale_name: str | None
    zero_point_name: str | None = None
    group_size: int | None = None
    axis: int | None = None
    packed_layout: str | None = None


@dataclass(frozen=True)
class WeightEntry:
    name: str
    kind: Literal["weight", "constant_tensor", "scale"]
    shape: tuple[int, ...]
    dtype: str
    layout: str
    offset: int
    nbytes: int
    alignment: int = 256
    transform: str | None = None
    tied_to: str | None = None
    quant: QuantSpec | None = None

    def to_weight_map_obj(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["quant"] = None if self.quant is None else asdict(self.quant)
        payload.pop("offset")
        payload.pop("nbytes")
        payload.pop("alignment")
        return payload

    def to_index_obj(self) -> dict[str, object]:
        return {
            "name": self.name,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "alignment": self.alignment,
        }


def select_fp8_layout(hardware: str | None = None, fp8_layout: str | None = None) -> str:
    if fp8_layout is not None:
        if fp8_layout not in ("kn", "nk"):
            raise ValueError(f"fp8_layout must be 'kn' or 'nk', got {fp8_layout!r}")
        return fp8_layout
    if hardware == "rtx_sm89":
        return "nk"
    if hardware == "rtx_sm120":
        return "kn"
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 8 and minor == 9:
                return "nk"
    except Exception:
        pass
    return "kn"


class WeightPackageWriter:
    """Write a devproc2 self-contained weight package."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model: str = "openpi0.5",
        precision: str = BF16,
        alignment: int = 256,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model = model
        self.precision = precision
        self.alignment = int(alignment)
        self._data = bytearray()
        self.entries: list[WeightEntry] = []

    def add_tensor(
        self,
        name: str,
        tensor: Any,
        *,
        dtype: str | None = None,
        kind: Literal["weight", "constant_tensor", "scale"] = "weight",
        layout: str = "row_major",
        transform: str | None = None,
        tied_to: str | None = None,
        quant: QuantSpec | None = None,
    ) -> WeightEntry:
        raw, shape, resolved_dtype = _tensor_to_bytes(tensor, dtype)
        offset = self._align(len(self._data))
        if offset > len(self._data):
            self._data.extend(b"\x00" * (offset - len(self._data)))
        self._data.extend(raw)
        entry = WeightEntry(
            name=name,
            kind=kind,
            shape=tuple(int(s) for s in shape),
            dtype=resolved_dtype,
            layout=layout,
            offset=offset,
            nbytes=len(raw),
            alignment=self.alignment,
            transform=transform,
            tied_to=tied_to,
            quant=quant,
        )
        self.entries.append(entry)
        return entry

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "weights.bin").write_bytes(bytes(self._data))
        _write_json(self.output_dir / "manifest.json", {
            "format": "devproc2.weights",
            "format_version": 1,
            "model": self.model,
            "precision": self.precision,
            "data_file": "weights.bin",
            "index_file": "weights.index.json",
            "weight_map_file": "weight_map.json",
        })
        _write_json(self.output_dir / "weights.index.json", {
            "format_version": 1,
            "data_file": "weights.bin",
            "entries": [entry.to_index_obj() for entry in self.entries],
        })
        _write_json(self.output_dir / "weight_map.json", {
            "format_version": 1,
            "weights": [entry.to_weight_map_obj() for entry in self.entries],
        })
        _write_json(self.output_dir / "quantization.json", {
            "format_version": 1,
            "entries": [
                {
                    "name": entry.name,
                    "shape": list(entry.shape),
                    "dtype": entry.dtype,
                    "quant": asdict(entry.quant),
                }
                for entry in self.entries
                if entry.quant is not None
            ],
        })

    def _align(self, value: int) -> int:
        rem = value % self.alignment
        return value if rem == 0 else value + (self.alignment - rem)


def convert_pi05_weights(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    hardware: str | None = "rtx_sm89",
    fp8_layout: str | None = None,
    include_bf16: bool = True,
    include_support_bf16: bool = True,
    include_fp8: bool = True,
    include_precomputed_styles: bool = True,
    action_horizon: int = ACTION_HORIZON,
    device: str = "cuda",
) -> None:
    """Convert the local Pi0.5 safetensors checkpoint to a devproc2 package."""

    checkpoint_dir = Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(safetensors_path)

    layout = select_fp8_layout(hardware, fp8_layout)
    weights = _convert_pi05_safetensors(safetensors_path)
    if include_bf16 and include_fp8:
        precision = "bf16+fp8"
    elif include_fp8:
        precision = "fp8"
    else:
        precision = BF16
    writer = WeightPackageWriter(output_dir, precision=precision)

    if include_bf16:
        for name, tensor in weights.items():
            writer.add_tensor(name, tensor, dtype=BF16, layout="row_major")
    elif include_support_bf16:
        for name, tensor in _iter_bf16_support_tensors(weights):
            writer.add_tensor(
                name,
                tensor,
                dtype=BF16,
                kind="constant_tensor" if name.startswith("constant.") else "weight",
                layout="row_major",
            )

    if include_precomputed_styles:
        styles = _precompute_decoder_styles(
            weights,
            chunk_size=action_horizon,
            num_steps=NUM_STEPS_DEFAULT,
            device=device,
        )
        for name, tensor in styles.items():
            writer.add_tensor(
                f"precomputed.{name}",
                tensor,
                dtype=BF16,
                kind="constant_tensor",
                layout="row_major",
            )

    if include_fp8:
        for name, tensor in _iter_fp8_targets(weights):
            q_bytes, scale, q_shape = _quantize_fp8_e4m3(tensor, layout=layout, device=device)
            scale_name = f"fp8.{name}.scale"
            weight_name = f"fp8.{name}.weight"
            writer.add_tensor(scale_name, scale, dtype=FP32, kind="scale", layout="scalar")
            writer.add_tensor(
                weight_name,
                q_bytes.reshape(q_shape),
                dtype=FP8_E4M3,
                layout=layout,
                quant=QuantSpec(
                    scheme="fp8_e4m3_per_tensor",
                    storage_dtype=FP8_E4M3,
                    compute_dtype=BF16,
                    scale_name=scale_name,
                    packed_layout=layout,
                ),
            )

    writer.write()
    _write_json(Path(output_dir) / "convert_report.json", {
        "source": {"type": "safetensors", "path": str(safetensors_path)},
        "ruleset": "openpi05_hf_to_devproc2_flashrt_v1",
        "fp8_layout": layout,
        "include_bf16": include_bf16,
        "include_support_bf16": include_support_bf16,
        "include_fp8": include_fp8,
        "include_precomputed_styles": include_precomputed_styles,
        "action_horizon": int(action_horizon),
        "num_entries": len(writer.entries),
    })


def _convert_pi05_safetensors(safetensors_path: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    bf16 = torch.bfloat16
    f = safe_open(str(safetensors_path), framework="pt")
    keys = set(f.keys())
    strip = "model." if all(k.startswith("model.") for k in keys) else ""

    def g(key: str):
        return f.get_tensor(strip + key).to(bf16).contiguous()

    def g_raw(key: str):
        return f.get_tensor(strip + key)

    ckpt: dict[str, Any] = {}

    vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    ckpt["vision_patch_embedding_w"] = (
        g(f"{vp}.embeddings.patch_embedding.weight").permute(2, 3, 1, 0).contiguous()
    )
    ckpt["vision_patch_embedding_b"] = g(f"{vp}.embeddings.patch_embedding.bias")
    ckpt["vision_position_embedding"] = g(f"{vp}.embeddings.position_embedding.weight")

    qkv_w, qkv_b, o_w, o_b = [], [], [], []
    up_w, up_b, down_w, down_b = [], [], [], []
    ln1_w, ln1_b, ln2_w, ln2_b = [], [], [], []
    for i in range(VIS_L):
        lp = f"{vp}.encoder.layers.{i}"
        q = g(f"{lp}.self_attn.q_proj.weight")
        k = g(f"{lp}.self_attn.k_proj.weight")
        v = g(f"{lp}.self_attn.v_proj.weight")
        qkv_w.append(torch.cat([q, k, v], dim=0).t().contiguous())
        qkv_b.append(torch.cat([
            g(f"{lp}.self_attn.q_proj.bias"),
            g(f"{lp}.self_attn.k_proj.bias"),
            g(f"{lp}.self_attn.v_proj.bias"),
        ]).contiguous())
        o_w.append(g(f"{lp}.self_attn.out_proj.weight").t().contiguous())
        o_b.append(g(f"{lp}.self_attn.out_proj.bias"))
        up_w.append(g(f"{lp}.mlp.fc1.weight").t().contiguous())
        up_b.append(g(f"{lp}.mlp.fc1.bias"))
        down_w.append(g(f"{lp}.mlp.fc2.weight").t().contiguous())
        down_b.append(g(f"{lp}.mlp.fc2.bias"))
        ln1_w.append(g(f"{lp}.layer_norm1.weight"))
        ln1_b.append(g(f"{lp}.layer_norm1.bias"))
        ln2_w.append(g(f"{lp}.layer_norm2.weight"))
        ln2_b.append(g(f"{lp}.layer_norm2.bias"))

    ckpt["vision_attn_qkv_w"] = torch.stack(qkv_w)
    ckpt["vision_attn_qkv_b"] = torch.stack(qkv_b)
    ckpt["vision_attn_o_w"] = torch.stack(o_w)
    ckpt["vision_attn_o_b"] = torch.stack(o_b)
    ckpt["vision_ffn_up_w"] = torch.stack(up_w)
    ckpt["vision_ffn_up_b"] = torch.stack(up_b)
    ckpt["vision_ffn_down_w"] = torch.stack(down_w)
    ckpt["vision_ffn_down_b"] = torch.stack(down_b)
    ckpt["vision_pre_attn_norm_w"] = torch.stack(ln1_w)
    ckpt["vision_pre_attn_norm_b"] = torch.stack(ln1_b)
    ckpt["vision_pre_ffn_norm_w"] = torch.stack(ln2_w)
    ckpt["vision_pre_ffn_norm_b"] = torch.stack(ln2_b)
    ckpt["vision_final_norm_w"] = g(f"{vp}.post_layernorm.weight")
    ckpt["vision_final_norm_b"] = g(f"{vp}.post_layernorm.bias")

    mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    ckpt["encoder_multi_modal_projector_w"] = g(f"{mp}.weight").t().contiguous()
    ckpt["encoder_multi_modal_projector_b"] = g(f"{mp}.bias")

    ep = "paligemma_with_expert.paligemma.model.language_model.layers"
    enc_qkv, enc_o, enc_gate, enc_up, enc_down = [], [], [], [], []
    for i in range(ENC_L):
        attn_scale = g_raw(f"{ep}.{i}.input_layernorm.weight").float()
        fuse_attn = 1.0 + attn_scale
        q = _interleave_qk(g_raw(f"{ep}.{i}.self_attn.q_proj.weight").float(), 8)
        k = _interleave_qk(g_raw(f"{ep}.{i}.self_attn.k_proj.weight").float(), 1)
        v = g_raw(f"{ep}.{i}.self_attn.v_proj.weight").float()
        enc_qkv.append(torch.cat([
            q * fuse_attn.unsqueeze(0),
            k * fuse_attn.unsqueeze(0),
            v * fuse_attn.unsqueeze(0),
        ], dim=0).t().to(bf16).contiguous())
        enc_o.append(g(f"{ep}.{i}.self_attn.o_proj.weight").t().contiguous())
        ffn_scale = g_raw(f"{ep}.{i}.post_attention_layernorm.weight").float()
        fuse_ffn = 1.0 + ffn_scale
        enc_gate.append((g_raw(f"{ep}.{i}.mlp.gate_proj.weight").float() * fuse_ffn.unsqueeze(0)).t().to(bf16).contiguous())
        enc_up.append((g_raw(f"{ep}.{i}.mlp.up_proj.weight").float() * fuse_ffn.unsqueeze(0)).t().to(bf16).contiguous())
        enc_down.append(g(f"{ep}.{i}.mlp.down_proj.weight").t().contiguous())
    ckpt["encoder_attn_qkv_w"] = torch.stack(enc_qkv)
    ckpt["encoder_attn_o_w"] = torch.stack(enc_o)
    ckpt["encoder_ffn_gate_w"] = torch.stack(enc_gate)
    ckpt["encoder_ffn_up_w"] = torch.stack(enc_up)
    ckpt["encoder_ffn_down_w"] = torch.stack(enc_down)

    dp = "paligemma_with_expert.gemma_expert.model.layers"
    dec_qkv, dec_o, dec_gate, dec_up, dec_down = [], [], [], [], []
    dec_attn_mod_w, dec_attn_mod_b, dec_ffn_mod_w, dec_ffn_mod_b = [], [], [], []
    for i in range(DEC_L):
        dec_attn_mod_w.append(g(f"{dp}.{i}.input_layernorm.dense.weight").t().contiguous())
        dec_attn_mod_b.append(g(f"{dp}.{i}.input_layernorm.dense.bias"))
        q = _interleave_qk(g(f"{dp}.{i}.self_attn.q_proj.weight").float(), 8).to(bf16)
        k = _interleave_qk(g(f"{dp}.{i}.self_attn.k_proj.weight").float(), 1).to(bf16)
        v = g(f"{dp}.{i}.self_attn.v_proj.weight")
        dec_qkv.append(torch.cat([q, k, v], dim=0).t().contiguous())
        dec_o.append(g(f"{dp}.{i}.self_attn.o_proj.weight").t().contiguous())
        dec_ffn_mod_w.append(g(f"{dp}.{i}.post_attention_layernorm.dense.weight").t().contiguous())
        dec_ffn_mod_b.append(g(f"{dp}.{i}.post_attention_layernorm.dense.bias"))
        dec_gate.append(g(f"{dp}.{i}.mlp.gate_proj.weight").t().contiguous())
        dec_up.append(g(f"{dp}.{i}.mlp.up_proj.weight").t().contiguous())
        dec_down.append(g(f"{dp}.{i}.mlp.down_proj.weight").t().contiguous())
    ckpt["decoder_attn_qkv_w"] = torch.stack(dec_qkv)
    ckpt["decoder_attn_o_w"] = torch.stack(dec_o)
    ckpt["decoder_ffn_gate_w"] = torch.stack(dec_gate)
    ckpt["decoder_ffn_up_w"] = torch.stack(dec_up)
    ckpt["decoder_ffn_down_w"] = torch.stack(dec_down)
    ckpt["decoder_pre_attn_norm_mod_w"] = torch.stack(dec_attn_mod_w)
    ckpt["decoder_pre_attn_norm_mod_b"] = torch.stack(dec_attn_mod_b)
    ckpt["decoder_pre_ffn_norm_mod_w"] = torch.stack(dec_ffn_mod_w)
    ckpt["decoder_pre_ffn_norm_mod_b"] = torch.stack(dec_ffn_mod_b)
    ckpt["decoder_final_norm_mod_w"] = g("paligemma_with_expert.gemma_expert.model.norm.dense.weight").t().contiguous()
    ckpt["decoder_final_norm_mod_b"] = g("paligemma_with_expert.gemma_expert.model.norm.dense.bias")

    ckpt["decoder_time_mlp_in_w"] = g("time_mlp_in.weight").t().contiguous()
    ckpt["decoder_time_mlp_in_b"] = g("time_mlp_in.bias")
    ckpt["decoder_time_mlp_out_w"] = g("time_mlp_out.weight").t().contiguous()
    ckpt["decoder_time_mlp_out_b"] = g("time_mlp_out.bias")
    ckpt["decoder_time_embeds"] = _build_time_embeddings()
    ckpt["decoder_action_in_proj_w"] = g("action_in_proj.weight").t().contiguous()
    ckpt["decoder_action_in_proj_b"] = g("action_in_proj.bias")
    ckpt["decoder_action_out_proj_w"] = (g("action_out_proj.weight").t().contiguous() * (-1.0 / NUM_STEPS_DEFAULT))
    ckpt["decoder_action_out_proj_b"] = g("action_out_proj.bias") * (-1.0 / NUM_STEPS_DEFAULT)
    ckpt["embedding_weight"] = g("paligemma_with_expert.paligemma.lm_head.weight")
    return ckpt


def _iter_fp8_targets(weights: dict[str, Any]):
    for i in range(VIS_L):
        yield f"vision_attn_qkv_w_{i}", weights["vision_attn_qkv_w"][i]
        yield f"vision_attn_o_w_{i}", weights["vision_attn_o_w"][i]
        yield f"vision_ffn_up_w_{i}", weights["vision_ffn_up_w"][i]
        yield f"vision_ffn_down_w_{i}", weights["vision_ffn_down_w"][i]
    yield "vision_projector_w", weights["encoder_multi_modal_projector_w"]
    for i in range(ENC_L):
        yield f"encoder_attn_qkv_w_{i}", weights["encoder_attn_qkv_w"][i]
        yield f"encoder_attn_o_w_{i}", weights["encoder_attn_o_w"][i]
        yield f"encoder_ffn_gate_up_w_{i}", _torch_cat(
            weights["encoder_ffn_gate_w"][i], weights["encoder_ffn_up_w"][i], dim=1
        )
        yield f"encoder_ffn_down_w_{i}", weights["encoder_ffn_down_w"][i]
    for i in range(DEC_L):
        yield f"decoder_attn_qkv_w_{i}", weights["decoder_attn_qkv_w"][i]
        yield f"decoder_attn_o_w_{i}", weights["decoder_attn_o_w"][i]
        yield f"decoder_ffn_gate_up_w_{i}", _torch_cat(
            weights["decoder_ffn_gate_w"][i], weights["decoder_ffn_up_w"][i], dim=1
        )
        yield f"decoder_ffn_down_w_{i}", weights["decoder_ffn_down_w"][i]


def _iter_bf16_support_tensors(weights: dict[str, Any]):
    """Tensors still needed by the FP8 fast path.

    These are intentionally not a full BF16 duplicate of quantized matmul
    weights. They cover embeddings, biases, norms, small action projections and
    constants that are not represented by the FP8 weight set.
    """

    support_names = (
        "vision_patch_embedding_w",
        "vision_patch_embedding_b",
        "vision_position_embedding",
        "vision_attn_qkv_b",
        "vision_attn_o_b",
        "vision_ffn_up_b",
        "vision_ffn_down_b",
        "vision_pre_attn_norm_w",
        "vision_pre_attn_norm_b",
        "vision_pre_ffn_norm_w",
        "vision_pre_ffn_norm_b",
        "vision_final_norm_w",
        "vision_final_norm_b",
        "encoder_multi_modal_projector_b",
        "embedding_weight",
        "decoder_action_in_proj_w",
        "decoder_action_in_proj_b",
        "decoder_action_out_proj_w",
        "decoder_action_out_proj_b",
    )
    for name in support_names:
        yield name, weights[name]
    yield "constant.decoder_adarms_weight", np.full((DEC_D,), 0x3F80, dtype=np.uint16)


def _quantize_fp8_e4m3(tensor: Any, *, layout: str, device: str):
    import torch

    w = tensor
    if layout == "nk" and getattr(w, "ndim", 0) == 2:
        w = w.t().contiguous()
    else:
        w = w.contiguous()
    w_dev = w.to(device=device)
    amax = w_dev.float().abs().max().item()
    scale = common_fp8_scale([amax])
    q = (w_dev.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q_u8 = q.contiguous().view(torch.uint8).cpu().numpy().copy()
    scale_np = np.asarray([scale], dtype=np.float32)
    del w_dev, q
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return q_u8, scale_np, tuple(int(s) for s in w.shape)


def _interleave_qk(w: Any, num_heads: int):
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
        .reshape(num_heads, 2, head_dim // 2, in_dim)
        .permute(0, 2, 1, 3)
        .reshape(out_dim, in_dim)
    )


def _build_time_embeddings():
    import torch

    t = torch.tensor(1.0, dtype=torch.float32)
    fraction = torch.linspace(0.0, 1.0, DEC_D // 2)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    items = []
    for _ in range(NUM_STEPS_DEFAULT):
        sinusoid_input = t.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        items.append(torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(torch.bfloat16))
        t = t - 1.0 / NUM_STEPS_DEFAULT
    return torch.cat(items, dim=0).contiguous()


def _precompute_decoder_styles(
    weights: dict[str, Any],
    *,
    chunk_size: int,
    num_steps: int,
    device: str,
) -> dict[str, np.ndarray]:
    import torch

    target_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    bf16 = torch.bfloat16

    def w(name: str) -> torch.Tensor:
        return weights[name].to(device=target_device, dtype=bf16)

    time_emb_schedule = w("decoder_time_embeds")
    t_in_w = w("decoder_time_mlp_in_w")
    t_in_b = w("decoder_time_mlp_in_b")
    t_out_w = w("decoder_time_mlp_out_w")
    t_out_b = w("decoder_time_mlp_out_b")
    attn_mod_w = w("decoder_pre_attn_norm_mod_w")
    attn_mod_b = w("decoder_pre_attn_norm_mod_b")
    ffn_mod_w = w("decoder_pre_ffn_norm_mod_w")
    ffn_mod_b = w("decoder_pre_ffn_norm_mod_b")
    final_mod_w = w("decoder_final_norm_mod_w")
    final_mod_b = w("decoder_final_norm_mod_b")

    time_emb_out = torch.empty(num_steps, chunk_size, DEC_D, dtype=bf16, device=target_device)
    style_attn = torch.empty(num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device=target_device)
    style_ffn = torch.empty(num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device=target_device)
    style_final = torch.empty(num_steps, chunk_size, 3 * DEC_D, dtype=bf16, device=target_device)

    for step in range(num_steps):
        te = time_emb_schedule[step:step + 1]
        tmp = te @ t_in_w + t_in_b[None, :]
        tmp = (tmp.float() * torch.sigmoid(tmp.float())).to(bf16)
        tmp = tmp @ t_out_w + t_out_b[None, :]
        tmp = (tmp.float() * torch.sigmoid(tmp.float())).to(bf16)
        te_expanded = tmp.expand(chunk_size, -1).contiguous()
        time_emb_out[step] = te_expanded

        for layer in range(DEC_L):
            style_attn[step, layer] = te_expanded @ attn_mod_w[layer] + attn_mod_b[layer][None, :]
            style_ffn[step, layer] = te_expanded @ ffn_mod_w[layer] + ffn_mod_b[layer][None, :]
        style_final[step] = te_expanded @ final_mod_w + final_mod_b[None, :]

    def to_bf16_bits(t: torch.Tensor) -> np.ndarray:
        return t.contiguous().view(torch.uint16).cpu().numpy().copy()

    out = {
        "decoder_time_emb": to_bf16_bits(time_emb_out),
        "decoder_style_attn": to_bf16_bits(style_attn),
        "decoder_style_ffn": to_bf16_bits(style_ffn),
        "decoder_style_final": to_bf16_bits(style_final),
    }
    if target_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def _torch_cat(a: Any, b: Any, *, dim: int):
    import torch

    return torch.cat([a, b], dim=dim).contiguous()


def _tensor_to_bytes(tensor: Any, dtype: str | None) -> tuple[bytes, tuple[int, ...], str]:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(tensor, torch.Tensor):
        t = tensor.detach().contiguous()
        resolved = dtype or _torch_dtype_name(t)
        if resolved == BF16:
            raw = t.to(torch.bfloat16).view(torch.uint16).cpu().numpy().tobytes()
        elif resolved == FP16:
            raw = t.to(torch.float16).cpu().numpy().tobytes()
        elif resolved == FP32:
            raw = t.to(torch.float32).cpu().numpy().tobytes()
        elif resolved == FP8_E4M3:
            raw = t.view(torch.uint8).cpu().numpy().tobytes() if t.dtype == torch.float8_e4m3fn else t.to(torch.uint8).cpu().numpy().tobytes()
        else:
            raw = t.cpu().numpy().tobytes()
        return raw, tuple(int(s) for s in t.shape), resolved

    arr = np.asarray(tensor)
    resolved = dtype or str(arr.dtype)
    if resolved == BF16:
        if arr.dtype != np.uint16:
            raise TypeError("numpy bfloat16 payload must be provided as uint16 bit pattern")
        raw = np.ascontiguousarray(arr).tobytes()
    elif resolved == FP8_E4M3:
        raw = np.ascontiguousarray(arr.astype(np.uint8, copy=False)).tobytes()
    else:
        raw = np.ascontiguousarray(arr).tobytes()
    return raw, tuple(int(s) for s in arr.shape), resolved


def _torch_dtype_name(tensor: Any) -> str:
    import torch

    if tensor.dtype == torch.bfloat16:
        return BF16
    if tensor.dtype == torch.float16:
        return FP16
    if tensor.dtype == torch.float32:
        return FP32
    if tensor.dtype == torch.float8_e4m3fn:
        return FP8_E4M3
    return str(tensor.dtype).removeprefix("torch.")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
