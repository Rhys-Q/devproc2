from __future__ import annotations

import devproc2.frontend.dsl as dp
from devproc2.kernel import KernelMatchKey
from devproc2.models.pi05.kernels import PI05_KERNELS, register_pi05_kernels


def test_register_pi05_kernels_populates_cuda_specs():
    dp.reset_module()
    register_pi05_kernels(sm_arch=89)
    registry = dp.get_kernel_registry()

    for decl in PI05_KERNELS:
        spec = registry.lookup(
            KernelMatchKey(decl.op, "cuda", decl.dtypes),
            sm_arch=89,
        )
        assert spec is not None, decl.op
        assert spec.backend == "cuda"
        assert spec.symbol == decl.symbol
        assert spec.source_path and spec.source_path.endswith("pi05_kernels.cu")
        assert spec.params == decl.params


def test_register_pi05_kernels_filters_sm_arch():
    dp.reset_module()
    register_pi05_kernels(sm_arch=89)
    first = PI05_KERNELS[0]
    assert dp.get_kernel_registry().lookup(
        KernelMatchKey(first.op, "cuda", first.dtypes),
        sm_arch=90,
    ) is None
