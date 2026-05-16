"""M9 ABI + Artifact MVP tests."""
import json
import os
import tempfile

import pytest

import devproc2.frontend.dsl as dp
from devproc2.compiler.pass_context import PassContext
from devproc2.compiler.passes.dps_lowering import DPSLoweringPass
from devproc2.compiler.passes.emit_abi import EmitABIPass
from devproc2.compiler.passes.emit_executable import EmitExecutablePass
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.compiler.passes.lower_tensor_create_to_alloc import LowerTensorCreateToAllocPass
from devproc2.compiler.passes.memory_planning import MemoryPlanningPass
from devproc2.compiler.passes.vm_codegen import VMCodegenPass
from devproc2.ir import (
    Block,
    EffectSummary,
    Function,
    IRModule,
    KernelRef,
    PackedFuncRef,
    Region,
    ReturnOp,
    TensorStructInfo,
    Var,
)
from devproc2.ir.ops import (
    AllocStorageOp,
    AllocTensorOp,
    CallDPSOp,
)
from devproc2.ir.prim_expr import IntImm, PrimVar
from devproc2.kernel.registry import KernelRegistry, KernelSpec
from devproc2.vm import Executable, serializer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_dsl():
    dp.reset_module()
    yield
    dp.reset_module()


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


def _spec(op: str, **kw) -> KernelSpec:
    defaults = dict(device="cpu", input_dtypes=("float16",),
                    kernel_name=f"kernel.{op}_fp16")
    defaults.update(kw)
    return KernelSpec(op_name=op, **defaults)


def _run_pipeline(module: IRModule, *specs: KernelSpec):
    """Full pipeline; returns (module_after_infer, exe, ctx)."""
    reg = KernelRegistry()
    for s in specs:
        reg.register(s)
    module = InferStructInfoPass().run(module)
    inferred_module = module  # save for ABI extraction
    module = DPSLoweringPass(reg).run(module)
    ctx = PassContext()
    MemoryPlanningPass().run(module, ctx)
    module = LowerTensorCreateToAllocPass(ctx).run(module)
    exe = VMCodegenPass().run(module)
    return inferred_module, exe, ctx


def _simple_tensor_module() -> IRModule:
    """Single-function module: main(x: Tensor[(4,), f16, cpu]) → Tensor."""
    x = Var("x", TensorStructInfo((IntImm(4),), "float16", "cpu"))
    s0 = AllocStorageOp("s0", IntImm(8), 256, "cpu")
    y = AllocTensorOp("y", s0.results[0], 0, (IntImm(4),), "float16")
    dps = CallDPSOp(
        KernelRef("kernel.relu_fp16"),
        (x,),
        (y.results[0],),
        EffectSummary.opaque_call(),
    )
    ret = ReturnOp(values=(y.results[0],))
    si = TensorStructInfo((IntImm(4),), "float16", "cpu")
    fn = Function(body=Region(blocks=(Block(args=(x,), ops=(s0, y, dps, ret)),)),
                  ret_struct_info=si)
    return IRModule(functions={"main": fn})


def _symbolic_module() -> tuple[IRModule, PrimVar, PrimVar]:
    """Module with symbolic dims B(upper=4) and S(upper=512)."""
    B = PrimVar("B", upper=4)
    S = PrimVar("S", upper=512)
    x = Var("x", TensorStructInfo((B, S, 64), "float16", "cpu"))
    ret = ReturnOp(x)
    si = TensorStructInfo((B, S, 64), "float16", "cpu")
    fn = Function(body=Region(blocks=(Block(args=(x,), ops=(ret,)),)),
                  ret_struct_info=si)
    module = IRModule(functions={"main": fn})
    return module, B, S


def _packed_func_module() -> IRModule:
    """Module with a packed_func CallDPS (no kernel, no output)."""
    x = Var("x", TensorStructInfo((IntImm(4),), "float16", "cpu"))
    s0 = AllocStorageOp("s0", IntImm(8), 256, "cpu")
    out = AllocTensorOp("out", s0.results[0], 0, (IntImm(4),), "float16")
    pf_call = CallDPSOp(
        PackedFuncRef("runtime.tokenizer.encode"),
        (x,),
        (out.results[0],),
        EffectSummary.opaque_call(),
    )
    ret = ReturnOp(values=(out.results[0],))
    si = TensorStructInfo((IntImm(4),), "float16", "cpu")
    fn = Function(body=Region(blocks=(Block(args=(x,), ops=(s0, out, pf_call, ret)),)),
                  ret_struct_info=si)
    return IRModule(functions={"main": fn})


# ---------------------------------------------------------------------------
# EmitExecutablePass tests
# ---------------------------------------------------------------------------

def test_emit_executable_writes_file(tmp_dir):
    module = _simple_tensor_module()
    _, exe, _ = _run_pipeline(module)
    EmitExecutablePass().run(exe, tmp_dir)
    vm_path = os.path.join(tmp_dir, "executable.vm")
    assert os.path.exists(vm_path)
    with open(vm_path, "rb") as f:
        magic = f.read(4)
    assert magic == b"DV2E"


def test_emit_executable_returns_bytes(tmp_dir):
    module = _simple_tensor_module()
    _, exe, _ = _run_pipeline(module)
    data = EmitExecutablePass().run(exe, tmp_dir)
    assert isinstance(data, bytes)
    assert data[:4] == b"DV2E"


def test_emit_executable_round_trip(tmp_dir):
    module = _simple_tensor_module()
    _, exe, _ = _run_pipeline(module)
    data = EmitExecutablePass().run(exe, tmp_dir)
    exe2 = serializer.deserialize(data)
    assert len(exe2.function_table) == len(exe.function_table)
    assert len(exe2.instructions) == len(exe.instructions)
    names1 = {fe.name for fe in exe.function_table}
    names2 = {fe.name for fe in exe2.function_table}
    assert names1 == names2


def test_emit_executable_file_round_trip(tmp_dir):
    module = _simple_tensor_module()
    _, exe, _ = _run_pipeline(module)
    EmitExecutablePass().run(exe, tmp_dir)
    vm_path = os.path.join(tmp_dir, "executable.vm")
    with open(vm_path, "rb") as f:
        data = f.read()
    exe2 = serializer.deserialize(data)
    assert len(exe2.function_table) == len(exe.function_table)


def test_emit_executable_creates_output_dir(tmp_path):
    nested = str(tmp_path / "nested" / "subdir")
    module = _simple_tensor_module()
    _, exe, _ = _run_pipeline(module)
    EmitExecutablePass().run(exe, nested)
    assert os.path.exists(os.path.join(nested, "executable.vm"))


# ---------------------------------------------------------------------------
# EmitABIPass — file structure tests
# ---------------------------------------------------------------------------

def test_emit_abi_creates_required_files(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    assert os.path.exists(os.path.join(tmp_dir, "abi.json"))
    assert os.path.exists(os.path.join(tmp_dir, "manifest.json"))
    assert os.path.exists(os.path.join(tmp_dir, "metadata", "function_table.json"))
    assert os.path.exists(os.path.join(tmp_dir, "metadata", "kernel_table.json"))
    assert os.path.exists(os.path.join(tmp_dir, "metadata", "packed_func_table.json"))
    assert os.path.exists(os.path.join(tmp_dir, "metadata", "storage_plan.json"))
    assert os.path.exists(os.path.join(tmp_dir, "metadata", "shape_constraints.json"))


def test_emit_abi_creates_placeholder_dirs(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    assert os.path.isdir(os.path.join(tmp_dir, "kernels"))
    assert os.path.isdir(os.path.join(tmp_dir, "constants"))


# ---------------------------------------------------------------------------
# EmitABIPass — abi.json content tests
# ---------------------------------------------------------------------------

def test_abi_version(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    assert abi["devproc_abi_version"] == "0.1"


def test_abi_json_has_required_keys(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    for key in ("devproc_abi_version", "inputs", "outputs", "shape_constraints",
                "required_packed_funcs", "target"):
        assert key in abi, f"missing key: {key}"


def test_abi_inputs_correct_dtype_shape(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    assert len(abi["inputs"]) == 1
    inp = abi["inputs"][0]
    assert inp["name"] == "x"
    assert inp["dtype"] == "float16"
    assert inp["shape"] == [4]


def test_abi_outputs_correct(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    assert len(abi["outputs"]) == 1
    assert abi["outputs"][0]["dtype"] == "float16"
    assert abi["outputs"][0]["shape"] == [4]


def test_abi_shape_constraints_from_symbolic_dims(tmp_dir):
    raw_module, B, S = _symbolic_module()
    # No ops to lower, just validate ABI extraction from a pure-passthrough fn
    ctx = PassContext()
    EmitABIPass().run(raw_module, Executable(), ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    sc = abi["shape_constraints"]
    assert "B" in sc
    assert sc["B"]["upper"] == 4
    assert "S" in sc
    assert sc["S"]["upper"] == 512


def test_abi_shape_constraints_no_symbolic_dims(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    assert abi["shape_constraints"] == {}


def test_abi_required_packed_funcs_listed(tmp_dir):
    pf_module = _packed_func_module()
    ctx = PassContext()
    exe = VMCodegenPass().run(pf_module)
    EmitABIPass().run(pf_module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    assert "runtime.tokenizer.encode" in abi["required_packed_funcs"]


def test_abi_required_packed_funcs_empty_for_kernel_only(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    assert "runtime.tokenizer.encode" not in abi["required_packed_funcs"]


def test_abi_symbolic_shape_as_string_in_inputs(tmp_dir):
    raw_module, B, S = _symbolic_module()
    ctx = PassContext()
    EmitABIPass().run(raw_module, Executable(), ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    shape = abi["inputs"][0]["shape"]
    # B and S are symbolic → string, 64 is static → int
    assert "B" in shape
    assert "S" in shape
    assert 64 in shape


# ---------------------------------------------------------------------------
# manifest.json tests
# ---------------------------------------------------------------------------

def test_manifest_has_required_fields(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir, model_name="my_model", target="cpu")
    with open(os.path.join(tmp_dir, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["name"] == "my_model"
    assert manifest["version"] == "0.1.0"
    assert "build_time" in manifest
    assert manifest["target"] == "cpu"


def test_manifest_build_time_is_iso8601(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "manifest.json")) as f:
        manifest = json.load(f)
    import datetime
    # Should parse without raising
    build_time = manifest["build_time"].rstrip("Z")
    datetime.datetime.fromisoformat(build_time)


# ---------------------------------------------------------------------------
# metadata/*.json tests
# ---------------------------------------------------------------------------

def test_function_table_json_contains_all_entries(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "metadata", "function_table.json")) as f:
        ft = json.load(f)
    assert isinstance(ft, list)
    assert len(ft) == len(exe.function_table)
    names = {entry["name"] for entry in ft}
    assert "kernel.relu_fp16" in names


def test_kernel_table_json_only_kernels(tmp_dir):
    module, exe, ctx = _run_pipeline(_simple_tensor_module())
    EmitABIPass().run(module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "metadata", "kernel_table.json")) as f:
        kt = json.load(f)
    for entry in kt:
        assert entry["kind"] == "kernel"


def test_packed_func_table_json_only_packed_funcs(tmp_dir):
    pf_module = _packed_func_module()
    ctx = PassContext()
    exe = VMCodegenPass().run(pf_module)
    EmitABIPass().run(pf_module, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "metadata", "packed_func_table.json")) as f:
        pft = json.load(f)
    assert any(e["name"] == "runtime.tokenizer.encode" for e in pft)
    for entry in pft:
        assert entry["kind"] == "packed_func"


def test_storage_plan_json_present_after_memory_planning(tmp_dir):
    @dp.function
    def main(x: dp.Tensor[(4,), "float16", "cpu"]):
        y = dp.ops.relu(x)
        return y

    raw_module = main.lower_module()
    inferred, exe, ctx = _run_pipeline(raw_module, _spec("relu"))
    EmitABIPass().run(inferred, exe, ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "metadata", "storage_plan.json")) as f:
        sp = json.load(f)
    assert isinstance(sp, list)
    assert len(sp) >= 1
    entry = sp[0]
    assert "id" in entry
    assert "device" in entry
    assert "size_bytes" in entry
    assert "alignment" in entry
    assert "reused_by" in entry


def test_storage_plan_json_empty_without_planning(tmp_dir):
    module, _, _ = _symbolic_module()
    ctx = PassContext()
    EmitABIPass().run(module, Executable(), ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "metadata", "storage_plan.json")) as f:
        sp = json.load(f)
    assert sp == []


def test_shape_constraints_json_matches_abi(tmp_dir):
    raw_module, B, S = _symbolic_module()
    ctx = PassContext()
    EmitABIPass().run(raw_module, Executable(), ctx, tmp_dir)
    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    with open(os.path.join(tmp_dir, "metadata", "shape_constraints.json")) as f:
        sc_meta = json.load(f)
    assert abi["shape_constraints"] == sc_meta


# ---------------------------------------------------------------------------
# Full artifact (execute + abi) integration tests
# ---------------------------------------------------------------------------

def test_full_artifact_structure(tmp_dir):
    """Emit both executable and ABI for a simple function; verify all files exist."""
    B = dp.symbolic_dim("B", upper=8)
    S = dp.symbolic_dim("S", upper=2048)

    @dp.function
    def main(x: dp.Tensor[(B, S, 64), "float16", "cpu"]):
        y = dp.ops.relu(x)
        return y

    raw_module = main.lower_module()
    inferred, exe, ctx = _run_pipeline(raw_module, _spec("relu"))

    EmitExecutablePass().run(exe, tmp_dir)
    EmitABIPass().run(inferred, exe, ctx, tmp_dir, model_name="test_model")

    # All required files exist
    for fname in ("executable.vm", "abi.json", "manifest.json"):
        assert os.path.exists(os.path.join(tmp_dir, fname)), f"missing: {fname}"

    with open(os.path.join(tmp_dir, "abi.json")) as f:
        abi = json.load(f)
    assert abi["devproc_abi_version"] == "0.1"
    assert abi["shape_constraints"]["B"]["upper"] == 8
    assert abi["shape_constraints"]["S"]["upper"] == 2048
    assert abi["inputs"][0]["name"] == "x"
    assert "B" in abi["inputs"][0]["shape"]
    assert "S" in abi["inputs"][0]["shape"]


def test_cross_process_artifact_round_trip(tmp_dir):
    """Write artifact then read it back from disk; executable must match."""
    module = _simple_tensor_module()
    _, exe, ctx = _run_pipeline(module)
    EmitExecutablePass().run(exe, tmp_dir)

    # Re-read and compare
    with open(os.path.join(tmp_dir, "executable.vm"), "rb") as f:
        data = f.read()
    exe2 = serializer.deserialize(data)
    assert {fe.name for fe in exe2.function_table} == {fe.name for fe in exe.function_table}
    assert len(exe2.instructions) == len(exe.instructions)
    assert len(exe2.constants) == len(exe.constants)
