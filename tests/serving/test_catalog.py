def test_sites_lists_eight_with_kind(client):
    sites = client.get("/v1/sites").json()
    assert len(sites) == 8
    assert all(site["kind"] in {"solar", "wind"} for site in sites)
    names = {site["name"] for site in sites}
    assert {"pavagada", "chitradurga"} <= names


def test_devices_returns_specs(client):
    specs = client.get("/v1/devices").json()
    assert specs
    assert {row["device"] for row in specs} >= {"lamp", "heater", "turbine"}
    assert all({"parameter", "value", "unit"} <= row.keys() for row in specs)
