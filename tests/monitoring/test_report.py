"""Evidently drift reports are written as HTML (skipped when evidently is not installed)."""

import pytest

from monitoring_fixtures import make_weather

pytest.importorskip("evidently")

from solarwind.monitoring import report  # noqa: E402


def test_report_is_written(tmp_path):
    out = report.build(make_weather(), "cloudy_monsoon", tmp_path / "drift_cloudy_monsoon.html")
    text = out.read_text(encoding="utf-8")
    assert out.stat().st_size > 10_000
    assert "<html" in text.lower()
