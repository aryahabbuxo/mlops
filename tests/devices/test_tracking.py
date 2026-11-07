"""TensorBoard tracking: HParams rows per model, same metric columns, flat folders, no TF."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)
from tensorboard.plugins.hparams import plugin_data_pb2  # noqa: E402

from solarwind.devices import tracking  # noqa: E402


@pytest.fixture
def tables():
    reg = pd.DataFrame(
        [
            {"task": t, "planner": p, "split": s, "mae": m, "rmse": m * 1.2, "bias": 0.0,
             "ratio_to_naive": 0.9, "ratio_to_persistence": 0.8}
            for t in ("sunlight_tomorrow", "hub_wind_tomorrow")
            for p, m in (("ridge", 0.6), ("hgb", 0.55))
            for s in ("train", "val", "test")
        ]
    )  # fmt: skip
    clf = pd.DataFrame(
        [
            {"rung": r, "split": s, "precision": 0.5, "recall": 0.5, "f1": 0.5, "pr_auc": 0.6}
            for r in ("yesterday_flag", "logistic")
            for s in ("train", "val", "test")
        ]
    )
    dev = pd.DataFrame(
        [
            {"planner": p, "lamp_mae_hours": 0.0, "heater_mae_C": 1.0, "turbine_mae_kwh_day": 5.0,
             "backup_precision": 0.9, "backup_recall": 0.8}
            for p in ("naive", "ridge")
        ]
    )  # fmt: skip
    picks = {
        "sunlight_tomorrow": "hgb",
        "hub_wind_tomorrow": "ridge",
        "low_wind_tomorrow": "logistic",
    }
    return reg, clf, dev, picks


def _log(tables, root, stamp):
    reg, clf, dev, picks = tables
    return tracking.log_tensorboard(
        reg, clf, dev, picks, git_commit="g", specs_sha256="s", root=root, stamp=stamp
    )


def _scalars(folder: Path) -> dict[str, float]:
    acc = EventAccumulator(str(folder))
    acc.Reload()
    return {tag: acc.Scalars(tag)[0].value for tag in acc.Tags()["scalars"]}


def _declared_metrics(folder: Path) -> list[str]:
    acc = EventAccumulator(str(folder))
    acc.Reload()
    raw = acc.PluginTagToContent("hparams")["_hparams_/experiment"]
    data = plugin_data_pb2.HParamsPluginData.FromString(raw)
    return [m.name.tag for m in data.experiment.metric_infos]


def test_one_flat_folder_per_model(tables, tmp_path):
    root = _log(tables, tmp_path, "t1")
    assert root == tmp_path
    names = {p.name for p in tmp_path.rglob("*") if p.is_dir()}
    for expected in (
        "t1-sunlight_tomorrow-ridge", "t1-sunlight_tomorrow-hgb", "t1-hub_wind_tomorrow-ridge",
        "t1-low_wind_tomorrow-logistic", "t1-device_error_test-naive", "t1-summary",
    ):  # fmt: skip
        assert expected in names
    # No nesting: nested run folders get back-slash names on Windows and the HParams tab then
    # shows empty metric cells.
    assert all(p.parent == tmp_path for p in tmp_path.rglob("*") if p.is_dir())


def test_validation_and_test_metrics_are_recorded(tables, tmp_path):
    _log(tables, tmp_path, "t2")
    s = _scalars(tracking.row_dir(tmp_path, "t2", "sunlight_tomorrow", "hgb"))
    assert s["val_mae"] == pytest.approx(0.55) and s["test_mae"] == pytest.approx(0.55)
    assert "train_mae" not in s
    c = _scalars(tracking.row_dir(tmp_path, "t2", "low_wind_tomorrow", "logistic"))
    assert c["val_pr_auc"] == pytest.approx(0.6)


def test_every_row_declares_the_same_metrics(tables, tmp_path):
    """The HParams tab takes its columns from one row, so all rows must agree (the val_mae bug)."""
    _log(tables, tmp_path, "t3")
    rows = [
        ("sunlight_tomorrow", "hgb"),
        ("low_wind_tomorrow", "logistic"),
        ("device_error_test", "naive"),
    ]
    declared = [_declared_metrics(tracking.row_dir(tmp_path, "t3", t, m)) for t, m in rows]
    assert declared[0] == declared[1] == declared[2] == list(tracking.ALL_METRICS)
    assert "val_mae" in declared[2]


def test_selected_flag_is_in_the_hparams(tables, tmp_path):
    _log(tables, tmp_path, "t4")
    acc = EventAccumulator(str(tracking.row_dir(tmp_path, "t4", "sunlight_tomorrow", "hgb")))
    acc.Reload()
    content = b"".join(acc.PluginTagToContent("hparams").values())
    assert b"selected" in content and b"task" in content


def test_leaderboard_text_is_ranked_on_validation(tables, tmp_path):
    _log(tables, tmp_path, "t5")
    acc = EventAccumulator(str(tmp_path / "t5-summary"))
    acc.Reload()
    tag = next(iter(acc.PluginTagToContent("text")))
    text = acc.Tensors(tag)[0].tensor_proto.string_val[0].decode()
    assert "## sunlight_tomorrow" in text and "## low_wind_tomorrow" in text
    sun = text.split("## sunlight_tomorrow")[1].split("##")[0]
    assert sun.index("| hgb |") < sun.index("| ridge |")  # hgb has the lower validation MAE
    assert "**yes**" in sun and sun.count("**yes**") == 1
    assert "train" not in text


def test_log_text_writes_a_note_to_the_text_tab(tmp_path):
    folder = tracking.log_text(tmp_path, "t6", "lstm-leaderboard", "# hello\n\n| a |\n|---|\n| 1 |")
    assert folder == tmp_path / "t6-lstm-leaderboard"
    acc = EventAccumulator(str(folder))
    acc.Reload()
    assert acc.PluginTagToContent("text")


def test_two_launches_do_not_collide(tables, tmp_path):
    for stamp in ("a", "b"):
        _log(tables, tmp_path, stamp)
    names = {p.name for p in tmp_path.iterdir()}
    assert "a-sunlight_tomorrow-hgb" in names and "b-sunlight_tomorrow-hgb" in names


def test_tracking_needs_no_tensorflow():
    """Run in a fresh interpreter: other tests in this process may already have imported TF."""
    code = (
        "import sys\n"
        "import pandas as pd\n"
        "from solarwind.devices import tracking\n"
        "assert 'tensorflow' not in sys.modules, 'importing tracking pulled in TensorFlow'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_training_no_longer_logs_to_mlflow():
    from solarwind.devices import train

    text = Path(train.__file__).read_text(encoding="utf-8")
    assert "import mlflow" not in text and "tracking.log_tensorboard" in text
