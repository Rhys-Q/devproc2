from __future__ import annotations

import json
from pathlib import Path

import pytest

import devproc2 as dp
import devproc2.frontend.dsl as dsl
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir.op_ref import KernelRef, PackedFuncRef
from devproc2.ir.ops import (
    CallDPSOp,
    CallOp,
    CudaCallOp,
    ReturnOp,
    TensorCreateOp,
    TensorViewOp,
    TupleOp,
)
from devproc2.nn import GraphBuilder, ScalarSpec, TensorSpec
from devproc2.models.pi05.modules import (
    PI05Attention,
    PI05DecoderLayer,
    PI05DenoiseLoop,
    PI05DenoiseStep,
    PI05FFN,
    PI05LanguageEmbedding,
    PI05Linear,
    PI05PaliGemmaEncoderLayer,
    PI05PaliGemmaPrefixEncoder,
    PI05SampleActionsFromPrefixEmbeddings,
    PI05VisionEncoder,
    PI05VisionEncoderLayer,
    PI05VisionPatchEmbedding,
)
from devproc2.models.pi05.export import (
    emit_pi05_denoise_executable,
    emit_pi05_denoise_loop_executable,
    emit_pi05_paligemma_prefix_encoder_executable,
    emit_pi05_paligemma_prefix_kv_encoder_executable,
    emit_pi05_sample_actions_precomputed_prefix_embs_executable,
    emit_pi05_sample_actions_precomputed_prefix_executable,
    emit_pi05_vision_encoder_executable,
    compile_pi05_sample_actions_tokens_executable,
    pi05_denoise_input_specs,
    pi05_denoise_loop_input_specs,
    pi05_paligemma_prefix_encoder_input_specs,
    pi05_paligemma_prefix_kv_encoder_input_specs,
    pi05_sample_actions_precomputed_prefix_embs_input_specs,
    pi05_sample_actions_precomputed_prefix_input_specs,
    pi05_vision_encoder_input_specs,
)


@pytest.fixture(autouse=True)
def reset_dsl():
    dp.reset_module()
    yield
    dp.reset_module()


def _call_names(module):
    fn = next(iter(module.functions.values()))
    return [
        op.op_ref.name
        for op in fn.body.blocks[0].ops
        if isinstance(op, CallOp)
    ]


def _lowered_ops(module, fn_name: str):
    lowered = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=89).run(module)
    return lowered.functions[fn_name].body.blocks[0].ops


def test_pi05_ffn_forward_uses_standard_ops():
    ffn = PI05FFN(8, 16)
    module = GraphBuilder().build(ffn.forward, {"x": TensorSpec((2, 8), "bfloat16")})
    names = _call_names(module)
    assert "matmul" in names
    assert "gelu" in names
    assert "multiply" in names
    assert not any(name.startswith("pi05.") for name in names)


def test_pi05_ffn_forward_fast_uses_fused_op():
    ffn = PI05FFN(8, 16)
    module = GraphBuilder().build(ffn.forward_fast, {"x": TensorSpec((33, 8), "bfloat16")})
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    create_ops = [op for op in ops if isinstance(op, TensorCreateOp)]
    assert len(dps_ops) == 4
    assert len(create_ops) == 4
    targets = [op.target_ref.name for op in dps_ops]
    assert targets == [
        "kernel.pi05_quantize_fp8_static_bf16",
        "runtime.cuda.fp8_nt_bf16",
        "kernel.pi05_geglu_to_fp8_bf16",
        "runtime.cuda.fp8_nt_bf16",
    ]
    assert isinstance(dps_ops[0].target_ref, KernelRef)
    assert dps_ops[0].target_ref.spec is not None
    assert dps_ops[0].target_ref.spec.launch.grid == (2, 1, 1)
    assert isinstance(dps_ops[1].target_ref, PackedFuncRef)
    assert isinstance(dps_ops[2].target_ref, KernelRef)
    assert dps_ops[2].target_ref.spec is not None
    assert dps_ops[2].target_ref.spec.launch.grid == (3, 1, 1)
    assert isinstance(dps_ops[3].target_ref, PackedFuncRef)
    assert create_ops[0].results[0].struct_info.dtype == "fp8_e4m3"
    assert create_ops[1].results[0].struct_info.shape[1].value == 32
    assert create_ops[2].results[0].struct_info.dtype == "fp8_e4m3"
    params = list(module.functions["forward_fast"].params)
    param_names = [p.name for p in params]
    assert "gate_up_w_fp8" in ".".join(param_names)
    assert "down_w_fp8" in ".".join(param_names)


def test_pi05_ffn_forward_fast_uses_dynamic_quant_scales():
    ffn = PI05FFN(8, 16)
    module = GraphBuilder().build(ffn._forward_fast_dynamic, {"x": TensorSpec((33, 8), "bfloat16")})
    ops = _lowered_ops(module, "_forward_fast_dynamic")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    assert [op.target_ref.name for op in dps_ops] == [
        "kernel.pi05_quantize_fp8_dynamic_bf16",
        "runtime.cuda.fp8_nt_bf16",
        "kernel.pi05_geglu_bf16",
        "kernel.pi05_quantize_fp8_dynamic_bf16",
        "runtime.cuda.fp8_nt_bf16",
    ]
    first_quant = dps_ops[0]
    second_quant = dps_ops[3]
    assert first_quant.target_ref.spec is not None
    assert first_quant.target_ref.spec.launch.grid == (1, 1, 1)
    assert second_quant.target_ref.spec is not None
    assert second_quant.target_ref.spec.launch.grid == (1, 1, 1)
    assert first_quant.effect.writes[0].struct_info.dtype == "fp8_e4m3"
    assert second_quant.effect.writes[0].struct_info.dtype == "fp8_e4m3"
    assert first_quant.effect.writes[1].struct_info.shape[0].value == 1
    assert second_quant.effect.writes[1].struct_info.shape[0].value == 1


def test_pi05_linear_forward_fast_uses_bf16_packed_gemm():
    linear = PI05Linear(8, 16, bias=False)
    module = GraphBuilder().build(linear.forward_fast, {"x": TensorSpec((33, 8), "bfloat16")})
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    assert len(dps_ops) == 1
    assert dps_ops[0].target_ref.name == "runtime.cuda.bf16_nn_bf16"
    assert isinstance(dps_ops[0].target_ref, PackedFuncRef)
    create = next(op for op in ops if isinstance(op, TensorCreateOp))
    assert create.results[0].struct_info.shape[1].value == 16


def test_pi05_linear_forward_fast_bias_uses_inplace_cuda_bias_kernel():
    linear = PI05Linear(8, 16, bias=True)
    module = GraphBuilder().build(linear.forward_fast, {"x": TensorSpec((33, 8), "bfloat16")})
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [
        op for op in ops
        if isinstance(op, CallDPSOp)
    ]
    assert [op.target_ref.name for op in dps_ops] == [
        "runtime.cuda.bf16_nn_bf16",
        "kernel.pi05_bias_add_bf16",
    ]
    assert not dps_ops[1].outputs


def test_pi05_vision_patch_embedding_forward_uses_standard_ops():
    embed = PI05VisionPatchEmbedding(
        num_views=2,
        image_size=4,
        patch_size=2,
        in_channels=3,
        vision_width=8,
    )
    module = GraphBuilder().build(
        embed.forward,
        {"patches": TensorSpec((8, 12), "bfloat16")},
    )
    names = _call_names(module)
    assert names == ["reshape", "matmul", "add", "cat", "add"]
    params = [p.name for p in module.functions["forward"].params]
    assert "vision_patch_embedding_w" in params
    assert "vision_patch_embedding_b" in params
    assert "vision_position_embedding" in params


def test_pi05_vision_patch_embedding_forward_fast_wires_prefix_kernels():
    embed = PI05VisionPatchEmbedding()
    module = GraphBuilder().build(
        embed.forward_fast,
        {"images_u8": TensorSpec((3, 224, 224, 3), "uint8")},
    )
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]
    assert targets == [
        "kernel.pi05_image_u8_to_bf16_norm",
        "kernel.pi05_patch_im2col_bf16",
        "runtime.cuda.bf16_nn_bf16",
        "kernel.pi05_bias_add_bf16",
        "kernel.pi05_position_add_bf16",
    ]
    assert dps_ops[0].target_ref.spec is not None
    assert dps_ops[0].target_ref.spec.launch.grid == (1764, 1, 1)
    assert dps_ops[1].target_ref.spec is not None
    assert dps_ops[1].target_ref.spec.launch.grid == (1764, 1, 1)
    assert not dps_ops[-1].outputs
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "vision_patch_embedding_w" in names
    assert "vision_patch_embedding_b" in names
    assert "vision_position_embedding" in names


def test_pi05_language_embedding_forward_fast_uses_cuda_gather():
    embed = PI05LanguageEmbedding(vocab_size=1024, hidden_size=16)
    module = GraphBuilder().build(
        embed.forward_fast,
        {"token_ids": TensorSpec((12,), "int32")},
    )
    ops = _lowered_ops(module, "forward_fast")
    dps = next(op for op in ops if isinstance(op, CallDPSOp))
    create = next(op for op in ops if isinstance(op, TensorCreateOp))
    assert dps.target_ref.name == "kernel.pi05_embedding_gather_bf16"
    assert dps.target_ref.spec is not None
    assert dps.target_ref.spec.launch.grid == (1, 1, 1)
    assert create.results[0].struct_info.shape[0].value == 12
    assert create.results[0].struct_info.shape[1].value == 16
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "embedding_weight" in names


def test_pi05_vision_encoder_layer_fast_dynamic_wires_siglip_block():
    layer = PI05VisionEncoderLayer(
        1,
        num_layers=3,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
    )
    module = GraphBuilder().build(
        layer.forward_fast,
        {"hidden": TensorSpec((4, 8), "bfloat16")},
    )
    ops = _lowered_ops(module, "forward_fast")
    views = [op for op in ops if isinstance(op, TensorViewOp)]
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]

    assert len(views) >= 8
    assert views[0].byte_stride == 8 * 2
    assert views[2].byte_stride == 24 * 2
    assert targets[:6] == [
        "kernel.pi05_layer_norm_bf16",
        "kernel.pi05_quantize_fp8_dynamic_bf16",
        "runtime.cuda.fp8_nt_bf16",
        "kernel.pi05_qkv_bias_split_bf16",
        "runtime.cuda.pi05_fa2_bf16_batched",
        "kernel.pi05_quantize_fp8_dynamic_bf16",
    ]
    assert targets.count("kernel.pi05_layer_norm_bf16") == 2
    assert targets.count("runtime.cuda.fp8_nt_bf16") == 4
    assert targets.count("runtime.cuda.fp8_nt_bf16_accum") == 0
    assert targets.count("kernel.pi05_bias_residual_bf16") == 2
    assert "kernel.pi05_qkv_bias_split_bf16" in targets
    assert "kernel.pi05_gelu_inplace_bf16" in targets
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "vision_pre_attn_norm_w" in names
    assert "vision_attn_qkv_b" in names
    assert "fp8.vision_attn_qkv_w_1.weight" in names
    assert "fp8.vision_ffn_down_w_1.scale" in names


def test_pi05_vision_encoder_layer_static_scales_fuse_layer_norm_quant():
    layer = PI05VisionEncoderLayer(
        1,
        num_layers=3,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        use_static_act_scales=True,
    )
    module = GraphBuilder().build(
        layer.forward_fast,
        {"hidden": TensorSpec((4, 8), "bfloat16")},
    )
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [
        op for op in ops
        if isinstance(op, CallDPSOp)
    ]
    targets = [op.target_ref.name for op in dps_ops]

    assert targets[0] == "kernel.pi05_layer_norm_to_fp8_bf16"
    assert targets.count("kernel.pi05_layer_norm_to_fp8_bf16") == 2
    assert targets.count("kernel.pi05_layer_norm_bf16") == 0
    assert "kernel.pi05_qkv_bias_split_bf16" in targets
    assert "kernel.pi05_bias_gelu_to_fp8_bf16" in targets
    assert "kernel.pi05_gelu_inplace_bf16" not in targets
    assert "kernel.pi05_quantize_fp8_dynamic_bf16" not in targets


def test_pi05_vision_encoder_unrolls_layers_and_projects_image_tokens():
    encoder = PI05VisionEncoder(
        num_layers=2,
        num_views=1,
        image_size=4,
        patch_size=2,
        in_channels=3,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        output_size=12,
    )
    module = GraphBuilder().build(
        encoder.forward_fast,
        {"images_u8": TensorSpec((1, 4, 4, 3), "uint8")},
    )
    fn = module.functions["forward_fast"]
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]
    ret = ops[-1]

    assert targets[:5] == [
        "kernel.pi05_image_u8_to_bf16_norm",
        "kernel.pi05_patch_im2col_bf16",
        "runtime.cuda.bf16_nn_bf16",
        "kernel.pi05_bias_add_bf16",
        "kernel.pi05_position_add_bf16",
    ]
    assert targets.count("kernel.pi05_layer_norm_bf16") == 5
    assert targets.count("runtime.cuda.fp8_nt_bf16") == 9
    assert targets.count("runtime.cuda.fp8_nt_bf16_accum") == 0
    assert targets[-2:] == [
        "runtime.cuda.fp8_nt_bf16",
        "kernel.pi05_bias_add_bf16",
    ]
    assert isinstance(ret, ReturnOp)
    assert ret.values[0].struct_info.shape[0].value == 4
    assert ret.values[0].struct_info.shape[1].value == 12
    names = [p.name for p in fn.params]
    assert names.count("vision_pre_attn_norm_w") == 1
    assert "fp8.vision_attn_qkv_w_0.weight" in names
    assert "fp8.vision_attn_qkv_w_1.weight" in names
    assert "fp8.vision_projector_w.weight" in names
    assert "encoder_multi_modal_projector_b" in names


def test_pi05_vision_encoder_fast_dynamic_lowers_to_vm():
    encoder = PI05VisionEncoder(
        num_layers=1,
        num_views=1,
        image_size=4,
        patch_size=2,
        in_channels=3,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        output_size=12,
    )
    module = GraphBuilder().build(
        encoder.forward_fast,
        {"images_u8": TensorSpec((1, 4, 4, 3), "uint8")},
    )
    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=89).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    names = [entry.name for entry in exe.function_table]
    assert "forward_fast" in names
    assert "vm.builtin.tensor_view" in names
    assert "runtime.cuda.pi05_fa2_bf16_batched" in names
    assert "runtime.cuda.fp8_nt_bf16" in names


def test_pi05_paligemma_encoder_layer_fast_dynamic_wires_rope_attention_and_ffn():
    layer = PI05PaliGemmaEncoderLayer(
        1,
        hidden_size=8,
        intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        layer.forward_fast,
        {
            "hidden": TensorSpec((5, 8), "bfloat16"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
        },
    )
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]

    assert targets[:5] == [
        "kernel.pi05_rms_norm_unit_bf16",
        "kernel.pi05_quantize_fp8_dynamic_bf16",
        "runtime.cuda.fp8_nt_bf16",
        "kernel.pi05_qkv_split_rope_bf16",
        "kernel.pi05_attention_bf16",
    ]
    assert targets.count("kernel.pi05_rms_norm_unit_bf16") == 2
    assert targets.count("runtime.cuda.fp8_nt_bf16") == 2
    assert targets.count("runtime.cuda.fp8_nt_bf16_accum") == 2
    assert targets.count("kernel.pi05_residual_add_bf16") == 0
    assert "kernel.pi05_geglu_bf16" in targets
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "fp8.encoder_attn_qkv_w_1.weight" in names
    assert "fp8.encoder_ffn_gate_up_w_1.weight" in names
    assert "fp8.encoder_ffn_down_w_1.scale" in names


def test_pi05_paligemma_encoder_layer_static_scales_fuse_rms_quant():
    layer = PI05PaliGemmaEncoderLayer(
        1,
        hidden_size=8,
        intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        use_static_act_scales=True,
    )
    module = GraphBuilder().build(
        layer.forward_fast,
        {
            "hidden": TensorSpec((5, 8), "bfloat16"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
        },
    )
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [
        op for op in ops
        if isinstance(op, CallDPSOp)
    ]
    targets = [op.target_ref.name for op in dps_ops]

    assert targets[0] == "kernel.pi05_rms_norm_unit_to_fp8_bf16"
    assert targets.count("kernel.pi05_rms_norm_unit_to_fp8_bf16") == 2
    assert targets.count("kernel.pi05_rms_norm_unit_bf16") == 0
    assert "kernel.pi05_reduce_amax_bf16" not in targets


def test_pi05_paligemma_prefix_encoder_unrolls_and_lowers_to_vm():
    encoder = PI05PaliGemmaPrefixEncoder(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        encoder.forward_fast,
        {
            "prefix_embs": TensorSpec((5, 8), "bfloat16"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
        },
    )
    fn = module.functions["forward_fast"]
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]
    ret = ops[-1]

    assert targets.count("kernel.pi05_rms_norm_unit_bf16") == 4
    assert targets.count("kernel.pi05_qkv_split_rope_bf16") == 2
    assert targets.count("kernel.pi05_attention_bf16") == 2
    assert targets.count("runtime.cuda.fp8_nt_bf16") == 4
    assert targets.count("runtime.cuda.fp8_nt_bf16_accum") == 4
    assert isinstance(ret, ReturnOp)
    assert ret.values[0].struct_info.shape[0].value == 5
    names = [p.name for p in fn.params]
    assert "fp8.encoder_attn_qkv_w_0.weight" in names
    assert "fp8.encoder_attn_qkv_w_1.weight" in names

    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=89).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    lowered_names = [entry.name for entry in exe.function_table]
    assert "forward_fast" in lowered_names
    assert "kernel.pi05_rms_norm_unit_bf16" in lowered_names
    assert "kernel.pi05_qkv_split_rope_bf16" in lowered_names


def test_pi05_paligemma_prefix_kv_encoder_materializes_cache_tuple():
    encoder = PI05PaliGemmaPrefixEncoder(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        encoder.forward_fast,
        {
            "prefix_embs": TensorSpec((5, 8), "bfloat16"),
            "prefix_valid_rows": ScalarSpec("int64"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
        },
    )
    fn = module.functions["forward_fast"]
    ops = _lowered_ops(module, "forward_fast")
    creates = [op for op in ops if isinstance(op, TensorCreateOp)]
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]
    ret = ops[-1]

    def shape_tuple(shape):
        return tuple(getattr(dim, "value", dim) for dim in shape)

    assert shape_tuple(creates[0].shape) == (2, 5, 1, 4)
    assert shape_tuple(creates[1].shape) == (2, 5, 1, 4)
    assert targets.count("kernel.pi05_qkv_split_rope_cache_bf16") == 2
    assert targets.count("kernel.pi05_copy_kv_cache_layer_bf16") == 0
    assert targets.count("runtime.cuda.pi05_fa2_bf16") == 1
    assert isinstance(ret, ReturnOp)
    assert isinstance(ret.values[0].op, TupleOp)

    module = InferStructInfoPass().run(module)
    tuple_si = module.functions["forward_fast"].ret_struct_info
    assert tuple_si is None
    ret_si = module.functions["forward_fast"].body.blocks[0].ops[-1].values[0].struct_info
    assert ret_si.fields[0].shape[0].value == 2
    assert ret_si.fields[0].shape[1].value == 5
    assert ret_si.fields[1].shape[3].value == 4

    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=89).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    lowered_names = [entry.name for entry in exe.function_table]
    assert "forward_fast" in lowered_names
    assert "kernel.pi05_qkv_split_rope_cache_bf16" in lowered_names
    assert "vm.builtin.make_tuple" in lowered_names


def test_pi05_sample_actions_from_prefix_embeddings_chains_prefix_kv_and_denoise():
    sample = PI05SampleActionsFromPrefixEmbeddings(
        num_layers=1,
        prefix_hidden_size=8,
        prefix_intermediate_size=16,
        decoder_hidden_size=8,
        decoder_intermediate_size=16,
        action_horizon=5,
        num_steps=2,
        action_dim=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        sample.forward_fast,
        {
            "noise_f32": TensorSpec((5, 4), "float32"),
            "prefix_embs": TensorSpec((3, 8), "bfloat16"),
            "prefix_valid_rows": ScalarSpec("int64"),
            "prefix_rope_interleaved": TensorSpec((3, 4), "bfloat16"),
            "suffix_rope_interleaved": TensorSpec((5, 4), "bfloat16"),
        },
    )
    fn = module.functions["forward_fast"]
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]
    ret = ops[-1]

    assert [p.name for p in fn.params][:5] == [
        "noise_f32",
        "prefix_embs",
        "prefix_valid_rows",
        "prefix_rope_interleaved",
        "suffix_rope_interleaved",
    ]
    assert targets.count("kernel.pi05_qkv_split_rope_cache_bf16") == 1
    assert targets.count("kernel.pi05_copy_kv_cache_layer_bf16") == 0
    assert targets.count("runtime.cuda.pi05_fa2_bf16") == 2
    assert targets.count("kernel.pi05_euler_update_bf16") == 2
    assert isinstance(ret, ReturnOp)
    assert ret.values[0].struct_info.dtype == "float32"
    assert ret.values[0].struct_info.shape[0].value == 5


def test_pi05_fast_modules_can_use_artifact_parameter_names():
    ffn = PI05FFN(
        8,
        16,
        gate_up_weight_name="fp8.encoder_ffn_gate_up_w_0.weight",
        gate_up_scale_name="fp8.encoder_ffn_gate_up_w_0.scale",
        down_weight_name="fp8.encoder_ffn_down_w_0.weight",
        down_scale_name="fp8.encoder_ffn_down_w_0.scale",
        act0_scale_name="act.encoder_ffn_gate_up_w_0.scale",
        act1_scale_name="act.encoder_ffn_down_w_0.scale",
    )
    module = GraphBuilder().build(ffn.forward_fast, {"x": TensorSpec((2, 8), "bfloat16")})
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "fp8.encoder_ffn_gate_up_w_0.weight" in names
    assert "fp8.encoder_ffn_down_w_0.scale" in names
    assert "act.encoder_ffn_down_w_0.scale" in names

    linear = PI05Linear(
        8,
        16,
        bias=True,
        weight_name="decoder_action_in_proj_w",
        bias_name="decoder_action_in_proj_b",
    )
    module = GraphBuilder().build(linear.forward_fast, {"x": TensorSpec((2, 8), "bfloat16")})
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "decoder_action_in_proj_w" in names
    assert "decoder_action_in_proj_b" in names


def test_pi05_attention_forward_fast_uses_cuda_kernel():
    attn = PI05Attention(num_q_heads=2, num_kv_heads=1, head_dim=4)
    module = GraphBuilder().build(
        attn.forward_fast,
        {
            "q": TensorSpec((3, 2, 4), "bfloat16"),
            "k": TensorSpec((5, 1, 4), "bfloat16"),
            "v": TensorSpec((5, 1, 4), "bfloat16"),
        },
    )
    ops = _lowered_ops(module, "forward_fast")
    dps = next(op for op in ops if isinstance(op, CallDPSOp))
    create = next(op for op in ops if isinstance(op, TensorCreateOp))
    assert dps.target_ref.name == "kernel.pi05_attention_bf16"
    assert isinstance(dps.target_ref, KernelRef)
    assert dps.target_ref.spec is not None
    assert dps.target_ref.spec.launch.grid == (3, 2, 1)
    assert dps.target_ref.spec.launch.shared_memory_bytes == 20
    assert create.results[0].struct_info.shape[0].value == 3
    assert create.results[0].struct_info.shape[1].value == 2
    assert create.results[0].struct_info.shape[2].value == 4


def test_pi05_direct_cuda_call_supports_multiple_outputs_without_registration():
    class SplitModule:
        def forward_fast(self, qkv):
            source = Path("python/devproc2/models/pi05/cuda/pi05_kernels.cu").resolve()
            q = dp.empty((1, 2), dtype="bfloat16", device="cuda")
            k = dp.empty((1, 2), dtype="bfloat16", device="cuda")
            v = dp.empty((1, 2), dtype="bfloat16", device="cuda")
            dp.cuda_call(
                f"{source}::pi05_qkv_split_bf16",
                qkv,
                1,
                2,
                2,
                2,
                q,
                k,
                v,
                metadata={
                    "kernel_name": "kernel.pi05_qkv_split_bf16",
                    "extra_nvcc_flags": ("--std=c++17",),
                },
            )
            return q

    module = GraphBuilder().build(
        SplitModule().forward_fast,
        {"qkv": TensorSpec((1, 6), "bfloat16")},
    )
    raw_ops = module.functions["forward_fast"].body.blocks[0].ops
    cuda = next(op for op in raw_ops if isinstance(op, CudaCallOp))
    assert cuda.output_indices == (5, 6, 7)

    ops = _lowered_ops(module, "forward_fast")
    dps = next(op for op in ops if isinstance(op, CallDPSOp))
    creates = [op for op in ops if isinstance(op, TensorCreateOp)]
    assert dps.target_ref.name == "kernel.pi05_qkv_split_bf16"
    assert dps.target_ref.spec is not None
    assert dps.target_ref.spec.input_dtypes == ("bfloat16", "", "", "", "")
    assert [param.source for param in dps.target_ref.spec.params][-3:] == ["output", "output", "output"]
    assert len(dps.inputs) == 8
    assert not dps.outputs
    assert len(creates) == 3
    assert all(create.results[0] in dps.effect.writes for create in creates)


def test_tensor_view_frontend_emits_alias_view():
    class SliceModule:
        def forward_fast(self, x):
            return dp.tensor_view(x, 3, (2, 4), byte_stride=8, base_offset=16)

    module = GraphBuilder().build(
        SliceModule().forward_fast,
        {"x": TensorSpec((8, 4), "bfloat16")},
    )
    ops = _lowered_ops(module, "forward_fast")
    view = next(op for op in ops if isinstance(op, TensorViewOp))
    assert view.byte_stride == 8
    assert view.base_offset == 16
    assert view.results[0].struct_info.dtype == "bfloat16"
    assert view.results[0].struct_info.shape[0].value == 2
    assert view.results[0].struct_info.shape[1].value == 4


def test_pi05_decoder_layer_fast_dynamic_wires_views_kv_and_inplace_residual():
    layer = PI05DecoderLayer(
        1,
        num_layers=18,
        hidden_size=8,
        intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        layer.forward_fast,
        {
            "hidden": TensorSpec((5, 8), "bfloat16"),
            "prefix_k_cache": TensorSpec((18, 3, 1, 4), "bfloat16"),
            "prefix_v_cache": TensorSpec((18, 3, 1, 4), "bfloat16"),
            "prefix_valid_rows": ScalarSpec("int64"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
            "style_attn_table": TensorSpec((10, 18, 5, 24), "bfloat16"),
            "style_ffn_table": TensorSpec((10, 18, 5, 24), "bfloat16"),
            "step": ScalarSpec("int64"),
        },
    )
    ops = _lowered_ops(module, "forward_fast")
    views = [op for op in ops if isinstance(op, TensorViewOp)]
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    assert len(views) >= 5
    assert views[0].byte_stride == 3 * 1 * 4 * 2
    assert views[2].byte_stride == 18 * 5 * 3 * 8 * 2
    targets = [op.target_ref.name for op in dps_ops]
    assert "kernel.pi05_qkv_split_rope_concat_bf16" in targets
    assert "kernel.pi05_kv_concat_bf16" not in targets
    assert "runtime.cuda.pi05_fa2_bf16" in targets
    assert targets.count("kernel.pi05_gate_mul_residual_bf16") == 2
    inplace = [op for op in dps_ops if op.target_ref.name == "kernel.pi05_gate_mul_residual_bf16"]
    assert all(not op.outputs for op in inplace)
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "fp8.decoder_attn_qkv_w_1.weight" in names
    assert "fp8.decoder_ffn_gate_up_w_1.weight" in names
    assert "constant.decoder_adarms_weight" in names


def test_pi05_denoise_step_fast_dynamic_wires_action_cast_layers_and_delta():
    denoise = PI05DenoiseStep(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        action_horizon=5,
        num_steps=10,
        action_dim=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        denoise.forward_fast,
        {
            "actions_f32": TensorSpec((5, 4), "float32"),
            "prefix_k_cache": TensorSpec((2, 3, 1, 4), "bfloat16"),
            "prefix_v_cache": TensorSpec((2, 3, 1, 4), "bfloat16"),
            "prefix_valid_rows": ScalarSpec("int64"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
            "step": ScalarSpec("int64"),
        },
    )
    ops = _lowered_ops(module, "forward_fast")
    targets = [
        op.target_ref.name
        for op in ops
        if isinstance(op, CallDPSOp)
    ]
    assert targets[0] == "kernel.pi05_cast_f32_to_bf16"
    assert targets.count("kernel.pi05_qkv_split_rope_concat_bf16") == 2
    assert targets.count("kernel.pi05_kv_concat_bf16") == 0
    assert targets[-2:] == [
        "runtime.cuda.bf16_nn_bf16",
        "kernel.pi05_bias_add_bf16",
    ]
    names = [p.name for p in module.functions["forward_fast"].params]
    assert "decoder_action_in_proj_w" in names
    assert "decoder_action_out_proj_b" in names
    assert "precomputed.decoder_style_attn" in names
    assert "precomputed.decoder_style_ffn" in names
    assert "precomputed.decoder_style_final" in names
    assert "fp8.decoder_attn_qkv_w_0.weight" in names
    assert "fp8.decoder_attn_qkv_w_1.weight" in names


def test_pi05_denoise_step_apply_delta_uses_bf16_euler_kernel():
    denoise = PI05DenoiseStep(num_layers=1, hidden_size=8, intermediate_size=16, action_dim=4)
    module = GraphBuilder().build(
        denoise._apply_delta_fast,
        {
            "actions_f32": TensorSpec((5, 4), "float32"),
            "delta_bf16": TensorSpec((5, 4), "bfloat16"),
        },
    )
    ops = _lowered_ops(module, "_apply_delta_fast")
    dps_ops = [
        op for op in ops
        if isinstance(op, CallDPSOp)
    ]
    assert len(dps_ops) == 1
    assert dps_ops[0].target_ref.name == "kernel.pi05_euler_update_bf16"
    assert not dps_ops[0].outputs


def test_pi05_denoise_loop_unrolls_steps_and_updates_actions_inplace():
    denoise_loop = PI05DenoiseLoop(
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        action_horizon=5,
        num_steps=3,
        action_dim=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        denoise_loop.forward_fast,
        {
            "actions_f32": TensorSpec((5, 4), "float32"),
            "prefix_k_cache": TensorSpec((1, 3, 1, 4), "bfloat16"),
            "prefix_v_cache": TensorSpec((1, 3, 1, 4), "bfloat16"),
            "prefix_valid_rows": ScalarSpec("int64"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
        },
    )
    fn = module.functions["forward_fast"]
    ops = _lowered_ops(module, "forward_fast")
    dps_ops = [op for op in ops if isinstance(op, CallDPSOp)]
    targets = [op.target_ref.name for op in dps_ops]

    assert "step" not in [p.name for p in fn.params]
    assert targets.count("kernel.pi05_cast_f32_to_bf16") == 3
    assert targets.count("kernel.pi05_euler_update_bf16") == 3
    assert targets[-1] == "kernel.pi05_euler_update_bf16"
    ret = ops[-1]
    assert isinstance(ret, ReturnOp)
    assert ret.values[0].struct_info.dtype == "float32"


def test_pi05_denoise_step_fast_dynamic_lowers_to_vm():
    denoise = PI05DenoiseStep(
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        action_horizon=5,
        num_steps=10,
        action_dim=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    module = GraphBuilder().build(
        denoise.forward_fast,
        {
            "actions_f32": TensorSpec((5, 4), "float32"),
            "prefix_k_cache": TensorSpec((1, 3, 1, 4), "bfloat16"),
            "prefix_v_cache": TensorSpec((1, 3, 1, 4), "bfloat16"),
            "prefix_valid_rows": ScalarSpec("int64"),
            "rope_interleaved": TensorSpec((5, 4), "bfloat16"),
            "step": ScalarSpec("int64"),
        },
    )
    module = InferStructInfoPass().run(module)
    module = DPSLoweringPass(dsl.get_kernel_registry(), sm_arch=89).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    names = [entry.name for entry in exe.function_table]
    assert "forward_fast" in names
    assert "vm.builtin.tensor_view" in names
    assert "kernel.pi05_qkv_split_rope_concat_bf16" in names
    assert "runtime.cuda.fp8_nt_bf16" in names


def test_pi05_denoise_input_specs_match_default_oracle_shapes():
    specs = pi05_denoise_input_specs()

    def shape(spec):
        return tuple(getattr(dim, "value", dim) for dim in spec.shape)

    assert shape(specs["actions_f32"]) == (50, 32)
    assert shape(specs["prefix_k_cache"]) == (18, 968, 1, 256)
    assert shape(specs["prefix_v_cache"]) == (18, 968, 1, 256)
    assert specs["prefix_valid_rows"].dtype == "int64"
    assert shape(specs["rope_interleaved"]) == (50, 256)
    assert specs["step"].dtype == "int64"


def test_pi05_denoise_loop_input_specs_drop_step_scalar():
    specs = pi05_denoise_loop_input_specs()

    def shape(spec):
        return tuple(getattr(dim, "value", dim) for dim in spec.shape)

    assert set(specs) == {
        "actions_f32",
        "prefix_k_cache",
        "prefix_v_cache",
        "prefix_valid_rows",
        "rope_interleaved",
    }
    assert shape(specs["actions_f32"]) == (50, 32)
    assert shape(specs["prefix_k_cache"]) == (18, 968, 1, 256)
    assert shape(specs["rope_interleaved"]) == (50, 256)


def test_pi05_sample_actions_precomputed_prefix_specs_use_noise_name():
    specs = pi05_sample_actions_precomputed_prefix_input_specs()

    def shape(spec):
        return tuple(getattr(dim, "value", dim) for dim in spec.shape)

    assert list(specs) == [
        "noise_f32",
        "prefix_k_cache",
        "prefix_v_cache",
        "prefix_valid_rows",
        "rope_interleaved",
    ]
    assert shape(specs["noise_f32"]) == (50, 32)
    assert shape(specs["prefix_k_cache"]) == (18, 968, 1, 256)
    assert shape(specs["rope_interleaved"]) == (50, 256)


def test_pi05_sample_actions_precomputed_prefix_embs_specs_include_prefix_inputs():
    specs = pi05_sample_actions_precomputed_prefix_embs_input_specs()

    def shape(spec):
        return tuple(getattr(dim, "value", dim) for dim in spec.shape)

    assert list(specs) == [
        "noise_f32",
        "prefix_embs",
        "prefix_valid_rows",
        "prefix_rope_interleaved",
        "suffix_rope_interleaved",
    ]
    assert shape(specs["noise_f32"]) == (50, 32)
    assert shape(specs["prefix_embs"]) == (968, 2048)
    assert specs["prefix_valid_rows"].dtype == "int64"
    assert shape(specs["prefix_rope_interleaved"]) == (968, 256)
    assert shape(specs["suffix_rope_interleaved"]) == (50, 256)


def test_pi05_vision_encoder_input_specs_match_default_shapes():
    specs = pi05_vision_encoder_input_specs()

    def shape(spec):
        return tuple(getattr(dim, "value", dim) for dim in spec.shape)

    assert list(specs) == ["images_u8"]
    assert shape(specs["images_u8"]) == (3, 224, 224, 3)
    assert specs["images_u8"].dtype == "uint8"


def test_pi05_paligemma_prefix_encoder_input_specs_match_default_shapes():
    specs = pi05_paligemma_prefix_encoder_input_specs()

    def shape(spec):
        return tuple(getattr(dim, "value", dim) for dim in spec.shape)

    assert list(specs) == ["prefix_embs", "rope_interleaved"]
    assert shape(specs["prefix_embs"]) == (968, 2048)
    assert shape(specs["rope_interleaved"]) == (968, 256)
    assert specs["prefix_embs"].dtype == "bfloat16"


def test_pi05_paligemma_prefix_kv_encoder_input_specs_include_valid_rows():
    specs = pi05_paligemma_prefix_kv_encoder_input_specs()

    def shape(spec):
        return tuple(getattr(dim, "value", dim) for dim in spec.shape)

    assert list(specs) == ["prefix_embs", "prefix_valid_rows", "rope_interleaved"]
    assert shape(specs["prefix_embs"]) == (968, 2048)
    assert specs["prefix_valid_rows"].dtype == "int64"
    assert shape(specs["rope_interleaved"]) == (968, 256)


def test_pi05_denoise_export_emits_main_vm_and_abi(tmp_path):
    summary = emit_pi05_denoise_executable(
        tmp_path,
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        action_dim=4,
        action_horizon=5,
        prefix_rows=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )

    assert (tmp_path / "executable.vm").read_bytes().startswith(b"DV2E")
    abi = json.loads((tmp_path / "abi.json").read_text())
    function_table = json.loads((tmp_path / "metadata" / "function_table.json").read_text())
    packed_table = json.loads((tmp_path / "metadata" / "packed_func_table.json").read_text())
    names = [entry["name"] for entry in abi["inputs"]]

    assert summary.function_name == "main"
    assert summary.num_user_inputs == 6
    assert function_table[-1]["name"] == "main"
    assert abi["inputs"][0]["name"] == "actions_f32"
    assert abi["outputs"] == [{"dtype": "bfloat16", "shape": [5, 4], "device": "cuda"}]
    assert names.count("constant.decoder_adarms_weight") == 1
    assert "precomputed.decoder_style_attn" in names
    assert "precomputed.decoder_style_ffn" in names
    assert "precomputed.decoder_style_final" in names
    assert "decoder_action_out_proj_b" in names
    assert {entry["name"] for entry in packed_table} >= {
        "runtime.cuda.bf16_nn_bf16",
        "runtime.cuda.fp8_nt_bf16",
    }


def test_pi05_denoise_loop_export_emits_main_vm_and_abi(tmp_path):
    summary = emit_pi05_denoise_loop_executable(
        tmp_path,
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        action_dim=4,
        action_horizon=5,
        prefix_rows=3,
        num_steps=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )

    assert (tmp_path / "executable.vm").read_bytes().startswith(b"DV2E")
    abi = json.loads((tmp_path / "abi.json").read_text())
    function_table = json.loads((tmp_path / "metadata" / "function_table.json").read_text())
    names = [entry["name"] for entry in abi["inputs"]]
    fn_names = [entry["name"] for entry in function_table]

    assert summary.function_name == "main"
    assert summary.num_user_inputs == 5
    assert function_table[-1]["name"] == "main"
    assert abi["outputs"] == [{"dtype": "float32", "shape": [5, 4], "device": "cuda"}]
    assert "step" not in names[: summary.num_user_inputs]
    assert "kernel.pi05_euler_update_bf16" in fn_names
    assert "runtime.cuda.pi05_fa2_bf16" in fn_names


def test_pi05_sample_actions_precomputed_prefix_export_emits_sample_actions_abi(tmp_path):
    summary = emit_pi05_sample_actions_precomputed_prefix_executable(
        tmp_path,
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        action_dim=4,
        action_horizon=5,
        prefix_rows=3,
        num_steps=3,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )

    abi = json.loads((tmp_path / "abi.json").read_text())
    function_table = json.loads((tmp_path / "metadata" / "function_table.json").read_text())
    names = [entry["name"] for entry in abi["inputs"]]
    fn_names = [entry["name"] for entry in function_table]

    assert summary.function_name == "main"
    assert summary.num_user_inputs == 5
    assert names[:5] == [
        "noise_f32",
        "prefix_k_cache",
        "prefix_v_cache",
        "prefix_valid_rows",
        "rope_interleaved",
    ]
    assert "actions_f32" not in names[:5]
    assert abi["outputs"] == [{"dtype": "float32", "shape": [5, 4], "device": "cuda"}]
    assert "kernel.pi05_euler_update_bf16" in fn_names
    assert "runtime.cuda.pi05_fa2_bf16" in fn_names


def test_pi05_sample_actions_precomputed_prefix_embs_export_emits_single_artifact_abi(tmp_path):
    summary = emit_pi05_sample_actions_precomputed_prefix_embs_executable(
        tmp_path,
        num_layers=1,
        prefix_hidden_size=8,
        prefix_intermediate_size=16,
        decoder_hidden_size=8,
        decoder_intermediate_size=16,
        action_dim=4,
        action_horizon=5,
        prefix_rows=3,
        num_steps=2,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )

    abi = json.loads((tmp_path / "abi.json").read_text())
    function_table = json.loads((tmp_path / "metadata" / "function_table.json").read_text())
    names = [entry["name"] for entry in abi["inputs"]]
    fn_names = [entry["name"] for entry in function_table]

    assert summary.function_name == "main"
    assert summary.num_user_inputs == 5
    assert names[:5] == [
        "noise_f32",
        "prefix_embs",
        "prefix_valid_rows",
        "prefix_rope_interleaved",
        "suffix_rope_interleaved",
    ]
    assert abi["outputs"] == [{"dtype": "float32", "shape": [5, 4], "device": "cuda"}]
    assert "kernel.pi05_qkv_split_rope_cache_bf16" in fn_names
    assert "runtime.cuda.pi05_fa2_bf16" in fn_names
    assert "kernel.pi05_euler_update_bf16" in fn_names
    assert "runtime.cuda.pi05_fa2_bf16" in fn_names
    assert "fp8.encoder_attn_qkv_w_0.weight" in names
    assert "fp8.decoder_attn_qkv_w_0.weight" in names


def test_pi05_vision_encoder_export_emits_main_vm_and_abi(tmp_path):
    summary = emit_pi05_vision_encoder_executable(
        tmp_path,
        num_layers=1,
        num_views=1,
        image_size=4,
        patch_size=2,
        image_channels=3,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        output_size=12,
    )

    abi = json.loads((tmp_path / "abi.json").read_text())
    function_table = json.loads((tmp_path / "metadata" / "function_table.json").read_text())
    names = [entry["name"] for entry in abi["inputs"]]
    fn_names = [entry["name"] for entry in function_table]

    assert summary.function_name == "main"
    assert summary.num_user_inputs == 1
    assert names[0] == "images_u8"
    assert abi["outputs"] == [{"dtype": "bfloat16", "shape": [4, 12], "device": "cuda"}]
    assert "kernel.pi05_image_u8_to_bf16_norm" in fn_names
    assert "runtime.cuda.pi05_fa2_bf16_batched" in fn_names
    assert "runtime.cuda.fp8_nt_bf16" in fn_names
    assert "fp8.vision_projector_w.weight" in names


def test_pi05_paligemma_prefix_encoder_export_emits_main_vm_and_abi(tmp_path):
    summary = emit_pi05_paligemma_prefix_encoder_executable(
        tmp_path,
        prefix_rows=5,
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )

    abi = json.loads((tmp_path / "abi.json").read_text())
    function_table = json.loads((tmp_path / "metadata" / "function_table.json").read_text())
    names = [entry["name"] for entry in abi["inputs"]]
    fn_names = [entry["name"] for entry in function_table]

    assert summary.function_name == "main"
    assert summary.num_user_inputs == 2
    assert names[:2] == ["prefix_embs", "rope_interleaved"]
    assert abi["outputs"] == [{"dtype": "bfloat16", "shape": [5, 8], "device": "cuda"}]
    assert "kernel.pi05_rms_norm_unit_bf16" in fn_names
    assert "kernel.pi05_qkv_split_rope_bf16" in fn_names
    assert "runtime.cuda.fp8_nt_bf16" in fn_names
    assert "fp8.encoder_attn_qkv_w_0.weight" in names
    assert "fp8.encoder_ffn_down_w_0.scale" in names


def test_pi05_paligemma_prefix_kv_encoder_export_emits_tuple_outputs(tmp_path):
    summary = emit_pi05_paligemma_prefix_kv_encoder_executable(
        tmp_path,
        prefix_rows=5,
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )

    abi = json.loads((tmp_path / "abi.json").read_text())
    function_table = json.loads((tmp_path / "metadata" / "function_table.json").read_text())
    names = [entry["name"] for entry in abi["inputs"]]
    fn_names = [entry["name"] for entry in function_table]

    assert summary.function_name == "main"
    assert summary.num_user_inputs == 3
    assert names[:3] == ["prefix_embs", "prefix_valid_rows", "rope_interleaved"]
    assert abi["outputs"] == [
        {"dtype": "bfloat16", "shape": [2, 5, 1, 4], "device": "cuda"},
        {"dtype": "bfloat16", "shape": [2, 5, 1, 4], "device": "cuda"},
    ]
    assert "kernel.pi05_qkv_split_rope_cache_bf16" in fn_names
    assert "vm.builtin.make_tuple" in fn_names
    assert "fp8.encoder_attn_qkv_w_1.weight" in names


def test_pi05_sample_tokens_normal_mode_compiles_without_cuda_kernels():
    result = compile_pi05_sample_actions_tokens_executable(
        function_name="main",
        action_horizon=2,
        action_dim=4,
        prefix_rows=12,
        num_steps=2,
        num_layers=1,
        num_views=1,
        image_size=4,
        patch_size=2,
        image_channels=3,
        vision_layers=1,
        vision_hidden_size=8,
        vision_intermediate_size=16,
        vision_heads=2,
        vocab_size=32,
        max_prompt_len=8,
        prefix_hidden_size=8,
        prefix_intermediate_size=16,
        decoder_hidden_size=8,
        decoder_intermediate_size=16,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        compile_mode="normal",
    )

    fn_names = [entry.name for entry in result.executable.function_table]

    assert "runtime.reference.attention" in fn_names
    assert "runtime.reference.image_patch_im2col" in fn_names
    assert not any(name.startswith("kernel.pi05") for name in fn_names)
