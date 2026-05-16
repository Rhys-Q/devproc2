import devproc2 as dp
import devproc2.nn as nn
from devproc2.compiler.passes.infer_struct_info import InferStructInfoPass
from devproc2.ir import print_module, verify
from devproc2.ir.nodes import ObjectStructInfo, ScalarStructInfo, TensorStructInfo
from devproc2.compiler.op import gelu, get_op, infer_struct_info, matmul, permute_dims
from devproc2.ir.ops import CallOp
from devproc2.ir.prim_expr import IntImm


def test_nested_module_named_parameters_are_stable():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 8)
            self.norm = nn.RMSNorm(8)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])
            self.out = nn.Linear(8, 2, bias=False)

    model = Model()

    assert list(model.state_dict()) == [
        "layers.0.proj.weight",
        "layers.0.proj.bias",
        "layers.0.norm.weight",
        "layers.1.proj.weight",
        "layers.1.proj.bias",
        "layers.1.norm.weight",
        "out.weight",
    ]


def test_linear_silu_linear_builds_standard_ops():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 8),
                nn.SiLU(),
                nn.Linear(8, 2),
            )

        def forward(self, x):
            return self.net(x)

    module = nn.GraphBuilder().build(
        MLP().forward,
        {"x": nn.TensorSpec((1, 4), "float16")},
    )
    verify(module)
    text = print_module(module)

    assert "@linear" not in text
    assert text.count("@permute_dims") == 2
    assert text.count("@matmul") == 2
    assert text.count("@add") == 2
    assert "@silu" in text
    assert "%net.layers.0.weight" in text
    assert "%net.layers.2.bias" in text


def test_reused_module_does_not_duplicate_parameters():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            a = self.proj(x)
            return self.proj(a)

    module = nn.GraphBuilder().build(
        Shared().forward,
        {"x": nn.TensorSpec((1, 4), "float16")},
    )
    fn = module.functions["forward"]

    param_names = [p.name for p in fn.params if p.name.startswith("proj.")]
    assert param_names == ["proj.weight"]
    assert print_module(module).count("%proj.weight") == 3


def test_custom_module_can_emit_top_level_dp_ops():
    class Activation(nn.Module):
        def forward(self, x):
            return dp.gelu(x, approximate="tanh")

    module = nn.GraphBuilder().build(
        Activation().forward,
        {"x": nn.TensorSpec((2, 4), "float16")},
    )

    text = print_module(module)
    assert "@gelu(%x) {approximate='tanh'}" in text


def test_top_level_layer_norm_accepts_parameters_and_kwargs():
    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter((4,), "float16")
            self.bias = nn.Parameter((4,), "float16")

        def forward(self, x):
            return dp.layer_norm(
                x,
                self.weight,
                self.bias,
                eps=1e-5,
                normalized_shape=(4,),
            )

    module = nn.GraphBuilder().build(
        Norm().forward,
        {"x": nn.TensorSpec((2, 4), "float16")},
    )
    op = next(
        op
        for op in module.functions["forward"].body.entry_block.ops
        if isinstance(op, CallOp)
    )

    assert op.callee == "@layer_norm"
    assert op.attrs == {
        "axes": (-1,),
        "center": True,
        "epsilon": 1e-5,
        "scale": True,
    }
    assert [p.name for p in module.functions["forward"].params] == [
        "x",
        "weight",
        "bias",
    ]


def test_basic_modules_emit_expected_attrs_and_paths():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 4, padding_idx=0, scale=2.0)
            self.norm = nn.LayerNorm(4, eps=1e-5)
            self.rms = nn.RMSNorm(4, eps=1e-6, use_adarms=True)

        def forward(self, ids, cond):
            x = self.embed(ids)
            x = self.norm(x)
            return self.rms(x, cond)

    module = nn.GraphBuilder().build(
        Model().forward,
        {
            "ids": nn.TensorSpec((2,), "int32"),
            "cond": nn.TensorSpec((2, 4), "float16"),
        },
    )
    verify(module)

    ops = [
        op
        for op in module.functions["forward"].body.entry_block.ops
        if isinstance(op, CallOp)
    ]
    assert ops[0].callee == "@embedding"
    assert ops[0].attrs == {"padding_idx": 0}
    assert ops[1].callee == "@multiply"
    assert ops[2].callee == "@layer_norm"
    assert ops[2].attrs == {
        "axes": (-1,),
        "center": True,
        "epsilon": 1e-5,
        "scale": True,
    }
    assert ops[3].callee == "@adarms_norm"
    assert ops[3].attrs == {"axes": (-1,), "epsilon": 1e-6}

    assert [p.name for p in module.functions["forward"].params] == [
        "ids",
        "cond",
        "embed.weight",
        "norm.weight",
        "norm.bias",
        "rms.weight",
    ]


def test_non_elementwise_nn_ops_stamp_correct_struct_info():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 4, dtype="float16")
            self.proj = nn.Linear(4, 8, dtype="float16", bias=False)

        def forward(self, ids):
            return self.proj(self.embed(ids))

    module = nn.GraphBuilder().build(
        Model().forward,
        {"ids": nn.TensorSpec((2, 3), "int32")},
    )
    inferred = InferStructInfoPass().run(module)
    ops = [
        op
        for op in inferred.functions["forward"].body.entry_block.ops
        if isinstance(op, CallOp)
    ]

    assert ops[0].callee == "@embedding"
    assert ops[0].results[0].struct_info == TensorStructInfo(
        (IntImm(2), IntImm(3), IntImm(4)),
        "float16",
        "cuda",
    )
    assert ops[1].callee == "@permute_dims"
    assert ops[1].attrs == {"axes": (1, 0)}
    assert ops[1].results[0].struct_info == TensorStructInfo(
        (IntImm(4), IntImm(8)),
        "float16",
        "cuda",
    )
    assert ops[2].callee == "@matmul"
    assert ops[2].results[0].struct_info == TensorStructInfo(
        (IntImm(2), IntImm(3), IntImm(8)),
        "float16",
        "cuda",
    )


def test_shared_module_alias_uses_first_parameter_path_once():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(4, 4, bias=False)
            self.first = shared
            self.second = shared

        def forward(self, x):
            a = self.first(x)
            return self.second(a)

    model = Model()
    assert list(model.state_dict()) == ["first.weight"]

    module = nn.GraphBuilder().build(
        model.forward,
        {"x": nn.TensorSpec((1, 4), "float16")},
    )

    assert [p.name for p in module.functions["forward"].params] == [
        "x",
        "first.weight",
    ]
    assert "%second.weight" not in print_module(module)


def test_graph_builder_accepts_scalar_and_object_specs():
    class Model(nn.Module):
        def forward(self, x, timestep, kv_cache):
            y = dp.call("add_timestep", x, timestep)
            return dp.call("use_cache", y, kv_cache)

    module = nn.GraphBuilder().build(
        Model().forward,
        {
            "x": nn.TensorSpec((1, 4), "float16"),
            "timestep": nn.ScalarSpec("int64"),
            "kv_cache": nn.ObjectSpec("kv_cache", role="mutable_runtime_state"),
        },
    )
    verify(module)

    params = module.functions["forward"].params
    assert params[1].struct_info == ScalarStructInfo("int64")
    assert params[2].struct_info == ObjectStructInfo("kv_cache", "mutable_runtime_state")

    text = print_module(module)
    assert "%timestep: Scalar[int64]" in text
    assert "%kv_cache: Object[kv_cache, role=mutable_runtime_state]" in text


def test_standard_ops_are_explicitly_registered_with_infer_fns():
    assert get_op("gelu") is gelu.op_def
    assert gelu.op_def.inputs[0].name == "x"
    assert gelu.op_def.attrs[0].name == "approximate"
    assert matmul.op_def.inputs[0].name == "a"
    assert matmul.op_def.inputs[1].name == "b"
    assert matmul.op_def.attrs[0].name == "out_dtype"
    assert permute_dims.op_def.attrs[0].name == "axes"

    a = TensorStructInfo((IntImm(2), IntImm(3), IntImm(4)), "float16", "cuda")
    b = TensorStructInfo((IntImm(4), IntImm(8)), "float16", "cuda")
    assert infer_struct_info("@matmul", (a, b), {}) == TensorStructInfo(
        (IntImm(2), IntImm(3), IntImm(8)),
        "float16",
        "cuda",
    )
