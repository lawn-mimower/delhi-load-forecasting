"""Tests for the TRAINING_RUN option in the LightGBM vs LSTM notebook.

Only the lines that pick the training settings are executed, so the tests
need neither the data nor PyTorch.
"""
import json
import types
from pathlib import Path

import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "model-comparisons"
    / "PyTorch_GPU_Reactive_LGBM_vs_LSTM.ipynb"
)


def code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def training_settings(environ):
    cell = next(src for src in code_cells() if "TRAINING_RUN" in src)
    lines = [
        line
        for line in cell.splitlines()
        if line.startswith(("TRAINING_RUN =", "EPOCHS, PATIENCE ="))
    ]
    assert len(lines) == 2
    namespace = {"os": types.SimpleNamespace(environ=environ)}
    exec("\n".join(lines), namespace)
    return namespace["EPOCHS"], namespace["PATIENCE"]


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, (30, 5)),
        ({"TRAINING_RUN": "standard"}, (30, 5)),
        ({"TRAINING_RUN": "long"}, (100, 100)),
    ],
)
def test_training_run_selects_epochs_and_patience(environ, expected):
    assert training_settings(environ) == expected


def test_unknown_training_run_is_rejected():
    with pytest.raises(KeyError):
        training_settings({"TRAINING_RUN": "short"})


def test_training_loop_uses_selected_settings():
    train_cell = next(src for src in code_cells() if "early_stopping = EarlyStopping(" in src)
    assert "EarlyStopping(patience=PATIENCE)" in train_cell
    assert "for epoch in range(EPOCHS):" in train_cell
