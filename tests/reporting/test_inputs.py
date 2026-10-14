from dataclasses import replace

import pytest

from reporting_fixtures import make_report_features
from solarwind.config import Settings
from solarwind.reporting.inputs import forbidden_columns, input_spec, lag_columns, validate_spec


def test_serveable_spec_excludes_every_forbidden_column():
    settings = Settings()
    for target in settings.baseline_targets:
        spec = input_spec(target, settings, with_lags=False)
        assert spec.serveable is True
        assert not set(spec.columns) & forbidden_columns(target, settings)
        assert f"{target}_lag1" not in spec.columns
        assert f"{target}_lag7" not in spec.columns


def test_with_lags_adds_only_this_targets_cf_lags_and_is_not_serveable():
    settings = Settings()
    for target in settings.baseline_targets:
        plain = input_spec(target, settings, with_lags=False)
        full = input_spec(target, settings, with_lags=True)
        assert full.serveable is False
        assert set(full.columns) - set(plain.columns) == set(lag_columns(target, settings))
        other = next(t for t in settings.baseline_targets if t != target)
        assert f"{other}_lag1" not in full.columns
        assert f"{other}_lag7" not in full.columns


def test_spec_never_contains_the_other_sources_weather():
    settings = Settings()
    for target in settings.baseline_targets:
        other = next(t for t in settings.baseline_targets if t != target).removesuffix("_cf")
        spec = input_spec(target, settings, with_lags=False)
        for column in spec.columns:
            assert not column.startswith(f"{other}_") or column == "wind_shear"


def test_forbidden_columns_never_flag_calendar_or_wind_shear():
    settings = Settings()
    for target in settings.baseline_targets:
        forbidden = forbidden_columns(target, settings)
        assert "wind_shear" not in forbidden
        for column in ("month_sin", "month_cos", "doy_sin", "doy_cos"):
            assert column not in forbidden


def test_validate_spec_raises_on_an_injected_forbidden_column():
    settings = Settings()
    frame = make_report_features(settings)
    spec = input_spec("solar_cf", settings, with_lags=False)
    bad_spec = replace(spec, columns=(*spec.columns, "wind_mu"))
    with pytest.raises(ValueError, match="forbidden"):
        validate_spec(bad_spec, settings, frame)


def test_validate_spec_raises_on_a_missing_column():
    settings = Settings()
    frame = make_report_features(settings).drop(columns=["wind_shear"])
    spec = input_spec("solar_cf", settings, with_lags=False)
    with pytest.raises(ValueError, match="missing"):
        validate_spec(spec, settings, frame)


def test_validate_spec_passes_on_a_well_formed_table():
    settings = Settings()
    frame = make_report_features(settings)
    for target in settings.baseline_targets:
        for with_lags in (False, True):
            validate_spec(input_spec(target, settings, with_lags), settings, frame)
