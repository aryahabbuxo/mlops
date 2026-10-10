import pandas as pd
import pytest

from nasa_fixtures import make_text
from solarwind.config import Settings, Site
from solarwind.data import join, tables


def settings_for(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def site(name: str, source: str, weight: float, lat: float) -> Site:
    return Site(
        state="karnataka",
        source_type=source,
        site_name=name,
        lat=lat,
        lon=77.0,
        weight=weight,
    )


def save_site(settings: Settings, s: Site, **kwargs) -> None:
    settings.nasa_raw_dir.mkdir(parents=True, exist_ok=True)
    (settings.nasa_raw_dir / s.filename).write_text(make_text(settings, s, **kwargs))


def test_load_flags_splits_date_and_description(tmp_path):
    settings = settings_for(tmp_path)
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "flags.txt").write_text(
        "2026-02-17 00:00:00 solar_state_control_mu zero\n"
        "2024-12-19 00:00:00 wind generation < state-wise\n"
    )
    flags = tables.load_flags(settings)
    assert list(flags["date"].dt.date.astype(str)) == ["2026-02-17", "2024-12-19"]
    assert flags["flag"].tolist() == [
        "solar_state_control_mu zero",
        "wind generation < state-wise",
    ]


def test_load_flags_returns_an_empty_frame_for_an_empty_file(tmp_path):
    settings = settings_for(tmp_path)
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "flags.txt").write_text("")
    assert tables.load_flags(settings).empty


def test_load_flags_rejects_an_unknown_line_format(tmp_path):
    settings = settings_for(tmp_path)
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "flags.txt").write_text("not a flag line\n")
    with pytest.raises(ValueError, match="unrecognised flag line"):
        tables.load_flags(settings)


def test_load_parse_failures_and_daily(tmp_path):
    settings = settings_for(tmp_path)
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "parse_failures.txt").write_text(
        "2025-10-14.xlsx KeyError: 'Generation'\n"
    )
    (settings.processed_dir / "karnataka_daily.csv").write_text(
        "date,wind_mu,solar_mu\n2024-07-06,1.5,2.5\n"
    )
    failures = tables.load_parse_failures(settings)
    assert failures.loc[0, "file"] == "2025-10-14.xlsx"
    assert failures.loc[0, "error"] == "KeyError: 'Generation'"
    daily = tables.load_daily(settings)
    assert daily["date"].iloc[0] == pd.Timestamp("2024-07-06")


def test_source_weather_is_the_equal_weight_mean_of_the_sites(tmp_path):
    settings = settings_for(tmp_path)
    a, b = site("a", "solar", 0.5, 14.0), site("b", "solar", 0.5, 15.0)
    save_site(settings, a)
    save_site(settings, b, value_shift=2.0)
    weather = join.source_weather(settings, [a, b], "solar")
    assert weather["T2M"].iloc[0] == pytest.approx(26.0)
    assert weather["T2M"].dtype == "float64"


def test_source_weather_uses_the_configured_weights(tmp_path):
    settings = settings_for(tmp_path)
    a, b = site("a", "wind", 0.75, 14.0), site("b", "wind", 0.25, 15.0)
    save_site(settings, a)
    save_site(settings, b, value_shift=4.0)
    weather = join.source_weather(settings, [a, b], "wind")
    assert weather["WS10M"].iloc[0] == pytest.approx(0.75 * 3.0 + 0.25 * 7.0)


def test_source_weather_is_nan_when_a_site_is_missing_that_day(tmp_path):
    settings = settings_for(tmp_path)
    a, b = site("a", "solar", 0.5, 14.0), site("b", "solar", 0.5, 15.0)
    save_site(settings, a, missing_at=2)
    save_site(settings, b)
    weather = join.source_weather(settings, [a, b], "solar")
    assert weather["ALLSKY_SFC_SW_DWN"].isna().tolist()[:4] == [False, False, True, False]


def test_source_weather_needs_at_least_one_site(tmp_path):
    with pytest.raises(ValueError, match="no wind sites"):
        join.source_weather(settings_for(tmp_path), [site("a", "solar", 1.0, 14.0)], "wind")


def test_join_keeps_only_dates_present_in_all_inputs():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2025-01-01"]),
            "wind_mu": [1.0, 2.0, 3.0],
            "solar_mu": [4.0, 5.0, 6.0],
        }
    )
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    solar = pd.DataFrame({"date": dates, "T2M": [20.0, 21.0, 22.0]})
    wind = pd.DataFrame({"date": dates, "WS50M": [5.0, 6.0, 7.0]})
    joined = join.join_with_weather(daily, solar, wind)
    assert joined["date"].tolist() == list(dates[:2])
    assert {"solar_T2M", "wind_WS50M", "wind_mu", "solar_mu"} <= set(joined.columns)
