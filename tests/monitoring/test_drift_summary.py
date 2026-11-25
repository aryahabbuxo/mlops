"""The CI job summary for the drift workflow (monitoring/drift_summary.py)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "web" / "sample" / "drift.json"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "drift_summary", ROOT / "monitoring" / "drift_summary.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load_script()


def sample() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_table_has_one_row_per_feature_and_model_output():
    result = sample()
    text = summary.markdown(result)

    assert "`cloudy_monsoon` is **ALERT**" in text
    assert "| solar | sun | 0.4100 | ALERT |" in text
    assert "| sunlight | 0.3300 | ALERT |" in text
    rows = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 2 + len(result["features"]) + len(result["prediction_drift"])


def test_missing_psi_and_missing_prediction_drift_are_handled():
    result = sample()
    result["features"][0]["psi"] = None
    result["prediction_drift"] = {}
    text = summary.markdown(result)

    assert "| solar | sun | n/a | ALERT |" in text
    assert "Model output" not in text


def test_only_a_drifting_result_is_annotated():
    result = sample()
    assert summary.annotation(result).startswith("::warning title=Drift ALERT::")
    result["status"] = "OK"
    assert summary.annotation(result) is None


def test_summary_is_appended_and_the_exit_code_is_always_zero(tmp_path, capsys):
    result_file = tmp_path / "drift.json"
    result_file.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "summary.md"
    out.write_text("earlier step\n", encoding="utf-8")

    assert summary.main([str(result_file), "--summary", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("earlier step\n## Drift check")
    assert "::warning" in capsys.readouterr().out

    assert summary.main([str(tmp_path / "missing.json")]) == 0
