"""Offline tests for arima-vs-lstm/check_gpu_setup.py.

Shell commands and the tensorflow import are faked, so the tests run on any
machine (with or without a GPU) in well under a second.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "arima-vs-lstm" / "check_gpu_setup.py"


@pytest.fixture
def gpu_check():
    spec = importlib.util.spec_from_file_location("check_gpu_setup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_commands(outputs):
    """Return a run_command replacement driven by a {command prefix: (stdout, ok)} map."""
    def run_command(cmd):
        for prefix, result in outputs.items():
            if cmd.startswith(prefix):
                return result
        return "", False
    return run_command


def fake_tensorflow(gpus, built_with_cuda=True):
    tf = types.ModuleType("tensorflow")
    tf.__version__ = "0.0-test"
    tf.config = types.SimpleNamespace(list_physical_devices=lambda kind: list(gpus))
    tf.test = types.SimpleNamespace(
        is_built_with_cuda=lambda: built_with_cuda,
        is_built_with_gpu_support=lambda: built_with_cuda,
    )
    return tf


def test_run_command_captures_stdout_and_status(gpu_check):
    assert gpu_check.run_command("echo hello") == ("hello", True)
    out, ok = gpu_check.run_command("exit 3")
    assert out == "" and ok is False


def test_no_nvidia_hardware(gpu_check, monkeypatch):
    monkeypatch.setattr(gpu_check, "run_command", fake_commands({}))
    assert gpu_check.check_nvidia_gpu() is False


def test_gpu_present_but_driver_missing(gpu_check, monkeypatch):
    monkeypatch.setattr(gpu_check, "run_command", fake_commands({
        "lspci": ("01:00.0 VGA compatible controller: NVIDIA Corporation", True),
    }))
    assert gpu_check.check_nvidia_gpu() is False


def test_gpu_and_driver_present(gpu_check, monkeypatch):
    monkeypatch.setattr(gpu_check, "run_command", fake_commands({
        "lspci": ("01:00.0 VGA compatible controller: NVIDIA Corporation", True),
        "nvidia-smi --query": ("535.54", True),
        "nvidia-smi": ("GPU table", True),
    }))
    assert gpu_check.check_nvidia_gpu() is True


def test_cuda_missing(gpu_check, monkeypatch):
    monkeypatch.setattr(gpu_check, "run_command", fake_commands({}))
    assert gpu_check.check_cuda() is False


def test_cuda_found(gpu_check, monkeypatch, capsys):
    monkeypatch.setattr(gpu_check, "run_command", fake_commands({
        "nvcc": ("Cuda compilation tools, release 12.1, V12.1.105", True),
    }))
    assert gpu_check.check_cuda() is True
    assert "release 12.1" in capsys.readouterr().out


@pytest.mark.parametrize("gpus, expected", [([], False), (["/physical_device:GPU:0"], True)])
def test_tensorflow_gpu_detection(gpu_check, monkeypatch, gpus, expected):
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow(gpus))
    assert gpu_check.check_tensorflow_gpu() is expected


def test_tensorflow_not_installed(gpu_check, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)  # makes the import raise ImportError
    assert gpu_check.check_tensorflow_gpu() is False


def test_main_without_gpu_writes_fix_script(gpu_check, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpu_check, "run_command", fake_commands({}))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow([], built_with_cuda=False))
    for name in ("cupy", "numba", "torch"):
        monkeypatch.setitem(sys.modules, name, None)
    gpu_check.main()
    out = capsys.readouterr().out
    assert "GPU setup issues detected" in out
    script = tmp_path / "fix_gpu_setup.sh"
    assert script.exists()
    assert script.read_text().startswith("#!/bin/bash")
