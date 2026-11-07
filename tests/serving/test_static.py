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
