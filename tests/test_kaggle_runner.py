import argparse
import importlib.util
import json
import subprocess

import pytest

from adaptive_vcm.evaluate import ROOT

spec = importlib.util.spec_from_file_location("kaggle_runner", ROOT / "scripts/kaggle_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("task", ["ar", "od"])
def test_private_notebook_pins_commit_and_contains_no_credential(task, tmp_path):
    args = argparse.Namespace(commit="a" * 40, slug="v22-test-" + task, account="example",
                              count=100, steps=1000, bootstrap=200, task=task, mode="learned",
                              seed=302001, directory=tmp_path / task)
    directory = runner.prepare(args)
    notebook = json.loads((directory / "notebook.ipynb").read_text())
    metadata = json.loads((directory / "kernel-metadata.json").read_text())
    cell = "".join(notebook["cells"][0]["source"])
    assert cell.startswith("%%bash\n")
    assert args.commit in cell and "adaptive_vcm.train" in cell and "adaptive_vcm.evaluate" in cell
    assert "KAGGLE_API_TOKEN" not in cell and "pool.json" not in cell
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True


def test_pool_auth_drops_inherited_other_account_without_printing_secret(tmp_path, monkeypatch):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"example": "PRIVATE_TEST_VALUE"}))
    monkeypatch.setenv("KAGGLE_KEY", "different-account")
    monkeypatch.setenv("KAGGLE_USERNAME", "different-account")
    env, secret = runner.pool_environment(pool, "example")
    assert env["KAGGLE_API_TOKEN"] == secret == "PRIVATE_TEST_VALUE"
    assert "KAGGLE_KEY" not in env and "KAGGLE_USERNAME" not in env


def test_v23_payload_registers_measured_rate_training_and_components(tmp_path):
    args = argparse.Namespace(commit="b" * 40, slug="v23-test-ar", account="example", recipe="v23",
                              count=128, steps=1000, bootstrap=2000, task="ar", mode="learned",
                              seed=302001, directory=tmp_path)
    runner.prepare(args)
    notebook = json.loads((tmp_path / "notebook.ipynb").read_text())
    cell = "".join(notebook["cells"][0]["source"])
    assert "adaptive_vcm.train_rateaware" in cell and "configs/v23_screen.json" in cell
    assert "--ablate-learned" in cell


def test_v24_payload_has_pinned_guard_control_and_profile_training(tmp_path):
    args = argparse.Namespace(commit='c' * 40, slug='v24-test-ar', account='example', recipe='v24',
                              count=128, steps=1000, bootstrap=2000, task='ar', mode='learned',
                              seed=302001, directory=tmp_path)
    runner.prepare(args)
    cell = ''.join(json.loads((tmp_path / 'notebook.ipynb').read_text())['cells'][0]['source'])
    metadata = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert 'adaptive_vcm.train_profiles' in cell and 'configs/v24_screen.json' in cell
    assert 'find_v23_checkpoint.py' in cell and '$OUT/guard_only' in cell
    assert metadata['kernel_sources'] == ['qktttttttttt/v23-rateaware-ar-s302001']
    assert cell.index('$OUT/guard_only') < cell.index('adaptive_vcm.train_profiles')


def test_kaggle_output_uses_utf8_and_redacts_secret(tmp_path, monkeypatch, capsys):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"example": "PRIVATE_TEST_VALUE"}))

    def completed(command, **kwargs):
        assert command[1:3] == ["-m", "kaggle"]
        assert kwargs["env"]["PYTHONUTF8"] == "1"
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "replace"
        return subprocess.CompletedProcess(command, 0, "Hoàn tất — ✓\n", "PRIVATE_TEST_VALUE\n")

    monkeypatch.setattr(runner.subprocess, "run", completed)
    output = runner.kaggle_call(["kernels", "status", "example/v22-test"], pool, "example")
    assert "Hoàn tất — ✓" in output and "[REDACTED]" in output
    assert "PRIVATE_TEST_VALUE" not in capsys.readouterr().out

