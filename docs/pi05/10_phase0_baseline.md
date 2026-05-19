# Pi0.5 Phase 0 Baseline

Date: 2026-05-19

This records the starting point for the staged Pi0.5 refactor in
`docs/pi05/09_pi05_refactor_plan.md`. Phase 0 is guardrail-only: existing
source debt is documented and protected against growth, but model files are not
restructured yet.

## Test Baseline

Command used in this checkout:

```bash
PYTHONPATH=python pytest tests/compiler/test_pi05_fast_modules.py tests/compiler/test_pi05_artifact.py tests/compiler/test_pi05_weight_package.py tests/compiler/test_pi05_nn_frontend.py
```

Result:

```text
57 passed in 2.06s
```

The command without `PYTHONPATH=python` currently fails during collection
because the checkout is not installed as an editable package in this shell.

## Model Source Boundary

The Phase 0 model-layer debt has been retired. Pi0.5 model code now lives in
`model.py`, direct backend calls live behind the Pi0.5 op facade, and the legacy
`modules.py` compatibility file is removed.

```text
python/devproc2/models/pi05/model.py
  dp.cuda_call: 0
  dp.call_dps_packed: 0
  dp.tensor_view: 0
  tensor_view(: 0
  runtime.cuda.: 0
```

Current FlashRT/runtime dependency scan:

```bash
rg "FlashRT|DEVPROC2_FLASHRT_FA2_SO|build-fa2-sm89|libflashrt" python/devproc2 runtime tests
```

The command is expected to return no matches for formal code and test paths.

## Export And Artifact Baseline

Current export entrypoints live in `python/devproc2/models/pi05/export.py`:

```text
compile_pi05_*_executable(...)
emit_pi05_*_executable(...)
export_pi05_*_artifact(...)
```

Current artifact resource packaging lives in
`python/devproc2/models/pi05/artifact.py` via `prepare_pi05_artifact(...)`.
The artifact layout produced by tests is:

```text
artifact/
  executable.vm
  abi.json
  metadata/
    function_table.json
    packed_func_table.json
    kernel_table.json
    pi05_artifact.json
    weight_map.json
    quantization.json
  weights/
    weights.bin
    weights.index.json
  kernels/
    *.cubin
  resources/
    tokenizer.model
```

`metadata/pi05_artifact.json` currently has format
`python/devproc2/artifact/pi05.py`, `format_version` 1, model `openpi0.5`, target
`cuda`, and keeps the weight, tokenizer, kernel table, and FP8 layout metadata
covered by `tests/compiler/test_pi05_artifact.py`.

## Benchmark Baseline

The current deploy benchmark target is:

```bash
cmake --build build/root-cuda --target bench_pi05_denoise -j2
```

Representative run commands are documented in
`docs/pi05/05_runtime_accuracy_milestones.md` and
`docs/pi05/08_build_run_profile.md`, using:

```bash
build/root-cuda/runtime/tests/bench_pi05_denoise <iters> --entry-kind <entry> --artifact-dir <artifact>
```

The benchmark reads raw oracle inputs from the existing Pi0.5 dump/artifact
layout; Phase 0 does not regenerate or retune performance artifacts.
