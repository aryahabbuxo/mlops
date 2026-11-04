import pytest
from fastapi.testclient import TestClient

from solarwind.serving.app import app


@pytest.fixture
def client():
    """A client whose lifespan has run, so the predictor is loaded and /ready is true."""
    with TestClient(app) as test_client:
        yield test_client
