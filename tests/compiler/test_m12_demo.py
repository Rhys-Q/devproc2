"""M12 End-to-End Demo tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Group A: Demo correctness
# ---------------------------------------------------------------------------

class TestDemoCorrectness:
    def test_demo_runs_without_error(self):
        """run_demo() completes without raising an exception."""
        # Import run_demo from examples (add repo root to path if needed)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from examples.kv_cache_mvp.run import run_demo
        max_err = run_demo(token_id=5)
        assert isinstance(max_err, float)

    def test_demo_output_within_tolerance(self):
        """VM output matches numpy reference with max error < 1e-3."""
        from examples.kv_cache_mvp.run import run_demo
        max_err = run_demo(token_id=3)
        assert max_err < 1e-3, f"max error {max_err:.2e} exceeds 1e-3"

    def test_demo_multiple_token_ids(self):
        """Demo produces correct output for several different token IDs."""
        from examples.kv_cache_mvp.run import run_demo
        for token_id in [0, 1, 7, 15]:
            max_err = run_demo(token_id=token_id)
            assert max_err < 1e-3, \
                f"token_id={token_id}: max error {max_err:.2e} exceeds 1e-3"


# ---------------------------------------------------------------------------
# Group B: CLI inspect command
# ---------------------------------------------------------------------------

class TestCLIInspect:
    def _emit_artifact(self, tmp_dir: str) -> None:
        """Emit a minimal artifact to tmp_dir for CLI testing."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from examples.kv_cache_mvp.run import emit_artifact
        emit_artifact(tmp_dir)

    def test_cli_inspect_exit_code_zero(self, tmp_dir):
        """devproc_cli.py inspect exits with code 0."""
        self._emit_artifact(tmp_dir)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cli_path = os.path.join(repo_root, "devproc_cli.py")
        result = subprocess.run(
            [sys.executable, cli_path, "inspect", tmp_dir],
            capture_output=True, text=True
        )
        assert result.returncode == 0, \
            f"CLI failed with:\n{result.stdout}\n{result.stderr}"

    def test_cli_inspect_shows_abi_version(self, tmp_dir):
        """CLI output contains the ABI version field."""
        self._emit_artifact(tmp_dir)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cli_path = os.path.join(repo_root, "devproc_cli.py")
        result = subprocess.run(
            [sys.executable, cli_path, "inspect", tmp_dir],
            capture_output=True, text=True
        )
        assert "devproc_abi_version" in result.stdout

    def test_cli_inspect_shows_packed_funcs(self, tmp_dir):
        """CLI output lists the required packed funcs."""
        self._emit_artifact(tmp_dir)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cli_path = os.path.join(repo_root, "devproc_cli.py")
        result = subprocess.run(
            [sys.executable, cli_path, "inspect", tmp_dir],
            capture_output=True, text=True
        )
        assert "runtime.embed" in result.stdout or "runtime.linear" in result.stdout

    def test_cli_inspect_shows_function_table(self, tmp_dir):
        """CLI output includes function table section."""
        self._emit_artifact(tmp_dir)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cli_path = os.path.join(repo_root, "devproc_cli.py")
        result = subprocess.run(
            [sys.executable, cli_path, "inspect", tmp_dir],
            capture_output=True, text=True
        )
        assert "Function Table" in result.stdout or "decode_step" in result.stdout

    def test_cli_inspect_nonexistent_dir(self, tmp_dir):
        """CLI returns error code 1 for a non-existent directory."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cli_path = os.path.join(repo_root, "devproc_cli.py")
        result = subprocess.run(
            [sys.executable, cli_path, "inspect", "/tmp/definitely_does_not_exist_xyz"],
            capture_output=True, text=True
        )
        assert result.returncode == 1


# ---------------------------------------------------------------------------
# Group C: Artifact structure
# ---------------------------------------------------------------------------

class TestArtifactStructure:
    def test_artifact_files_present(self, tmp_dir):
        """emit_artifact() creates all expected files."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from examples.kv_cache_mvp.run import emit_artifact
        emit_artifact(tmp_dir)

        assert os.path.exists(os.path.join(tmp_dir, "executable.vm"))
        assert os.path.exists(os.path.join(tmp_dir, "abi.json"))
        assert os.path.exists(os.path.join(tmp_dir, "manifest.json"))
        assert os.path.exists(os.path.join(tmp_dir, "metadata", "function_table.json"))

    def test_abi_json_has_required_packed_funcs(self, tmp_dir):
        """abi.json lists runtime.embed and runtime.linear."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from examples.kv_cache_mvp.run import emit_artifact
        emit_artifact(tmp_dir)

        with open(os.path.join(tmp_dir, "abi.json")) as f:
            abi = json.load(f)
        pf = abi.get("required_packed_funcs", [])
        assert "runtime.embed" in pf
        assert "runtime.linear" in pf
