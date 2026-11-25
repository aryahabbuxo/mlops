"""The training window is versioned in params.yaml and loaded explicitly by the tools."""

import pandas as pd
import pytest

from solarwind.devices import pipeline as pipe


@pytest.fixture(autouse=True)
def _restore_split():
    before = (pipe.TRAIN_END, pipe.VAL_END)
    yield
    pipe.set_split(*before)


def write(tmp_path, body: str):
    path = tmp_path / "params.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_reads_the_split_block(tmp_path):
    path = write(
        tmp_path,
        '# comment\nsplit:\n  train_end: "2024-12-31"  # note\n  val_end: 2025-12-31\n'
        "other:\n  x: 1\n",
    )
    assert pipe.read_split(path) == ("2024-12-31", "2025-12-31")


def test_missing_file_means_no_change(tmp_path):
    assert pipe.read_split(tmp_path / "nope.yaml") is None
    assert pipe.use_params(tmp_path / "nope.yaml") == (pipe.TRAIN_END, pipe.VAL_END)


def test_incomplete_split_is_refused(tmp_path):
    path = write(tmp_path, "split:\n  train_end: 2024-12-31\n")
    with pytest.raises(ValueError, match="must set"):
        pipe.read_split(path)


def test_use_params_moves_the_split(tmp_path):
    path = write(tmp_path, "split:\n  train_end: 2024-12-31\n  val_end: 2025-12-31\n")
    assert pipe.use_params(path) == ("2024-12-31", "2025-12-31")
    labels = pipe.split_of(pd.Series(pd.to_datetime(["2024-06-01", "2025-06-01", "2026-06-01"])))
    assert list(labels) == ["train", "val", "test"]


def test_split_dates_must_be_in_order():
    with pytest.raises(ValueError, match="before"):
        pipe.set_split("2025-12-31", "2024-12-31")


def test_repository_params_are_the_window_of_a_deployed_model():
    # params.yaml is the training window of the newest release: at least one deployed model must
    # have been trained on it. Skipped when serve/ is not pulled.
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    manifest = root / "serve" / "manifest.json"
    if not manifest.is_file():
        pytest.skip("serve/ bundle not pulled")
    from solarwind.devices import bundle

    m = json.loads(manifest.read_text(encoding="utf-8"))
    windows = {bundle.task_split(m, t) for t in m["tasks"]}
    assert pipe.read_split(root / "params.yaml") in windows
