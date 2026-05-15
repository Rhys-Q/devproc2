"""devproc2 M12 End-to-End Demo: LLM decode step MVP.

Demonstrates all MVP core features:
- dp.empty() + call_dps_packed (tokenizer embed)
- dp.ops.relu (kernel dispatch via VMInterpreter mock)
- call_dps_packed (linear projection)
- Full compiler pipeline: DSL → InferStructInfo → DPS → MemoryPlan → VMCodegen
- VMInterpreter execution with mock PackedFuncs + mock kernel

Usage::

    python examples/kv_cache_mvp/run.py

Success output::

    PASS: max error = X.XXe-XX  (must be < 1e-3)

Exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import os
import struct
import sys

# Allow running from repo root or from examples/kv_cache_mvp/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _REPO_ROOT)  # for examples.kv_cache_mvp.ref_impl

import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_abi import EmitABIPass
from devproc2.compiler.passes.emit_executable import EmitExecutablePass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.kernel.registry import KernelRegistry, KernelSpec, KernelMatchKey
from devproc2.vm.interpreter import VMInterpreter, _Storage, _Tensor

from examples.kv_cache_mvp.ref_impl import (
    EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, VOCAB_SIZE,
    EMBED_WEIGHT, LINEAR_WEIGHT,
    reference_decode_step, pack_f32, unpack_f32,
)


# ---------------------------------------------------------------------------
# DSL model definition
# ---------------------------------------------------------------------------

@dp.function
def decode_step(token_id: dp.Tensor[(1,), "int32", "cpu"]):
    # Step 1: embed (PackedFunc)
    embedded = dp.empty((EMBED_DIM,), dtype="float32", device="cpu")
    dp.call_dps_packed("runtime.embed", inputs=[token_id], output=embedded)

    # Step 2: relu (kernel)
    relu_out = dp.ops.relu(embedded)

    # Step 3: linear projection (PackedFunc)
    output = dp.empty((OUTPUT_DIM,), dtype="float32", device="cpu")
    dp.call_dps_packed("runtime.linear", inputs=[relu_out], output=output)

    return output


# ---------------------------------------------------------------------------
# Compiler pipeline
# ---------------------------------------------------------------------------

def compile_model():
    """Run full compiler pipeline; return (exe, ctx, inferred_module)."""
    module = decode_step.lower_module()

    # Register a mock relu kernel so DPS lowering can select it
    relu_spec = KernelSpec(
        op_name="relu",
        device="cpu",
        input_dtypes=("float32",),
        kernel_name="kernel.relu_fp32",
    )
    kernel_registry = KernelRegistry()
    kernel_registry.register(relu_spec)

    module = InferStructInfoPass().run(module)
    inferred_module = module

    module = DPSLoweringPass(kernel_registry).run(module)

    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)

    exe = VMCodegenPass().run(module)
    return exe, ctx, inferred_module


# ---------------------------------------------------------------------------
# Mock PackedFunc implementations (numpy-based CPU)
# ---------------------------------------------------------------------------

def register_mock_packed_funcs(vm: VMInterpreter) -> None:
    """Register CPU numpy mocks for runtime.embed and runtime.linear."""

    def embed_fn(args: list) -> None:
        # args: [token_id_tensor, output_tensor]
        tok_tensor, out_tensor = args[0], args[1]
        token_id = struct.unpack_from("<i", tok_tensor.storage.data,
                                     tok_tensor.offset)[0]
        row = EMBED_WEIGHT[token_id % VOCAB_SIZE]
        out_data = pack_f32(row)
        out_tensor.storage.data[out_tensor.offset:out_tensor.offset + len(out_data)] = out_data

    def linear_fn(args: list) -> None:
        # args: [hidden_tensor, output_tensor]
        hidden_tensor, out_tensor = args[0], args[1]
        hidden = unpack_f32(hidden_tensor.storage.data, HIDDEN_DIM,
                            hidden_tensor.offset)
        result = hidden @ LINEAR_WEIGHT
        out_data = pack_f32(result)
        out_tensor.storage.data[out_tensor.offset:out_tensor.offset + len(out_data)] = out_data

    vm.register_packed_func("runtime.embed", embed_fn)
    vm.register_packed_func("runtime.linear", linear_fn)


def register_mock_kernel(vm: VMInterpreter) -> None:
    """Register a numpy relu mock for kernel.relu_fp32."""

    def relu_fn(args: list) -> None:
        # args: [input_tensor, output_tensor]
        in_t, out_t = args[0], args[1]
        vals = unpack_f32(in_t.storage.data, EMBED_DIM, in_t.offset)
        result = vals.clip(min=0.0)
        out_data = pack_f32(result)
        out_t.storage.data[out_t.offset:out_t.offset + len(out_data)] = out_data

    vm.register_kernel("kernel.relu_fp32", relu_fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(token_id: int = 5) -> float:
    """Run full demo pipeline for a given token_id; return max abs error."""
    dp.reset_module()
    exe, ctx, inferred_module = compile_model()

    vm = VMInterpreter(exe)
    register_mock_packed_funcs(vm)
    register_mock_kernel(vm)

    # Build input tensor (1 int32 token_id)
    in_storage = _Storage(bytearray(4), 1, 0)
    struct.pack_into("<i", in_storage.data, 0, token_id)
    in_tensor = _Tensor(in_storage, 0, (1,), 0, 32, 1)

    result = vm.invoke("decode_step", [in_tensor])
    vm_output = unpack_f32(result.storage.data, OUTPUT_DIM, result.offset)

    ref_output = reference_decode_step(token_id)
    max_err = float(abs(vm_output - ref_output).max())
    return max_err


def emit_artifact(output_dir: str) -> None:
    """Compile and emit artifact to output_dir (for CLI testing)."""
    dp.reset_module()
    exe, ctx, inferred_module = compile_model()
    EmitExecutablePass().run(exe, output_dir)
    EmitABIPass().run(inferred_module, exe, ctx, output_dir,
                      model_name="kv_cache_demo", target="cpu")


def main() -> int:
    max_err = run_demo(token_id=5)
    if max_err < 1e-3:
        print(f"PASS: max error = {max_err:.2e}")
        return 0
    else:
        print(f"FAIL: max error = {max_err:.2e} exceeds 1e-3")
        return 1


if __name__ == "__main__":
    sys.exit(main())
