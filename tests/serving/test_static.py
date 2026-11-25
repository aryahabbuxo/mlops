"""The browser UI is served from the static mount at / without shadowing the API routes."""


def test_root_serves_the_ui_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "GreenOps" in res.text


def test_static_mount_does_not_shadow_the_api(client):
    # The catch-all mount is registered last, so declared routes still win.
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["ready"] is True


def test_ops_page_is_served_and_reads_the_ops_and_drift_endpoints(client):
    res = client.get("/ops.html")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "/v1/ops/status" in res.text
    assert "/v1/monitoring/drift?scenario=" in res.text


def test_ops_page_only_uses_sections_the_status_endpoint_returns(client):
    import re

    from solarwind.serving.schemas import OpsStatusResponse

    page = client.get("/ops.html").text
    used = set(re.findall(r"\bst\.(\w+)", page))
    assert used, "the page should read the status sections through `st.`"
    assert used <= set(OpsStatusResponse.model_fields)
