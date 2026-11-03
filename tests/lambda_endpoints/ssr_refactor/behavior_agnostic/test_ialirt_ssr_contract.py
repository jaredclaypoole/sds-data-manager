"""Discover SSR dependency contracts with only instrument names as concrete inputs."""

import json
import os
from pathlib import Path

import pytest

from tests.lambda_endpoints.ssr_refactor.behavior_agnostic.utils import (
    Inputs,
    discover,
    external_bindings,
    readable_runs,
)
from tests.lambda_endpoints.ssr_refactor.utils import format_trace, load_ssr_modules

processing, ssr_utils = load_ssr_modules()
TARGET_MODULES = (processing,) if ssr_utils is None else (processing, ssr_utils)

INSTRUMENTS = (
    "mag",
    "hit",
    "swe",
    "codice_lo",
    "codice_hi",
    "swapi",
    "spacecraft_status",
    "spacecraft",
    "unknown",
)
TRACE_DIR = Path(__file__).with_name("ssr_traces")
# Mock imported dependencies and external instances,
# never helpers defined in the target modules.
MOCKS = external_bindings(TARGET_MODULES)


@pytest.mark.parametrize("instrument", INSTRUMENTS)
def test_lambda_dependency_contract(instrument):
    """Compare automatically explored executions with a reviewed baseline."""

    def invoke(recorder):
        processing.os.environ = Inputs(
            recorder, "environment", {"INSTRUMENT": instrument}
        )
        return processing.lambda_handler(
            Inputs(recorder, "event"), Inputs(recorder, "context")
        )

    actual = readable_runs(discover(invoke, MOCKS), TARGET_MODULES)
    baseline = TRACE_DIR / f"{instrument}.json"
    if os.environ.get("SSR_UPDATE_TRACES") == "1":
        TRACE_DIR.mkdir(exist_ok=True)
        baseline.write_text(format_trace(actual, sort_keys=False) + "\n")
    assert baseline.exists(), f"Missing reviewed baseline: {baseline}"
    assert actual == json.loads(baseline.read_text())
