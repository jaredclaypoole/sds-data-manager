"""Discover SSR dependency contracts with only instrument names as concrete inputs."""

import json
import os
import sys
from pathlib import Path

import pytest

from sds_data_manager.lambda_code.IAlirtCode import ialirt_ssr_utils as ssr_utils
from tests.lambda_endpoints.ssr_refactor.behavior_agnostic.utils import (
    Inputs,
    discover,
    external_bindings,
    readable_runs,
)
from tests.lambda_endpoints.ssr_refactor.utils import format_trace

sys.modules["ialirt_ssr_utils"] = ssr_utils
from sds_data_manager.lambda_code.IAlirtCode import (  # noqa: E402
    ialirt_ssr_processing as processing,
)

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
# Mock imported dependencies and external instances, never either module's helpers.
MOCKS = external_bindings((processing, ssr_utils))


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

    actual = readable_runs(discover(invoke, MOCKS), (processing, ssr_utils))
    baseline = TRACE_DIR / f"{instrument}.json"
    if os.environ.get("SSR_UPDATE_TRACES") == "1":
        TRACE_DIR.mkdir(exist_ok=True)
        baseline.write_text(format_trace(actual, sort_keys=False) + "\n")
    assert baseline.exists(), f"Missing reviewed baseline: {baseline}"
    assert actual == json.loads(baseline.read_text())
