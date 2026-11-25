"""Shared test safety: the command-line tools load params.yaml and change the training window
for their process. Inside one pytest run that would leak into every later test, so the default
window is put back after each test."""

import pytest

from solarwind.devices import pipeline as pipe


@pytest.fixture(autouse=True)
def _keep_the_default_split():
    before = (pipe.TRAIN_END, pipe.VAL_END)
    yield
    pipe.TRAIN_END, pipe.VAL_END = before
