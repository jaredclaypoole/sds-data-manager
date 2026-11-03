"""Characterize SSR's external calls without mocking either module's helpers.

See README.md in this directory for the baseline/update workflow and scope.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import imap_data_access
import numpy as np
import pytest
import xarray as xr

from sds_data_manager.lambda_code.IAlirtCode import ialirt_ssr_utils as utils

# Match the Lambda's flat import without loading a second copy of the utilities.
sys.modules["ialirt_ssr_utils"] = utils
from sds_data_manager.lambda_code.IAlirtCode import (  # noqa: E402
    ialirt_ssr_processing as processing,
)

DAY = datetime(2026, 10, 8)
DAY_NS = 86400 * 10**9
TRACE_DIR = Path(__file__).with_name("ssr_traces")
INSTRUMENTS = (
    "mag",
    "hit",
    "swe",
    "codice_lo",
    "codice_hi",
    "swapi",
    "spacecraft_status",
    "spacecraft",
)
CASES = (
    [(instrument, "success") for instrument in INSTRUMENTS]
    + [
        ("hit", scenario)
        for scenario in (
            "empty",
            "mixed",
            "no_packets",
            "late_start",
            "early_end",
            "sequence_gap",
            "missing_l0",
            "http_error",
            "ssm_error",
            "processor_error",
            "write_error",
        )
    ]
    + [("swe", "missing_ancillary"), ("unknown", "success")]
)


def format_trace(value, indent=0):
    """Render JSON with two-space indentation and arrays fitting 100 columns inline."""
    compact = json.dumps(value, sort_keys=True)
    if isinstance(value, list) and indent + len(compact) <= 100:
        return compact
    if isinstance(value, dict) and value:
        entries = [
            " " * (indent + 2)
            + json.dumps(key)
            + ": "
            + format_trace(value[key], indent + 2)
            for key in sorted(value)
        ]
        return "{\n" + ",\n".join(entries) + "\n" + " " * indent + "}"
    if isinstance(value, list) and value:
        entries = [
            " " * (indent + 2) + format_trace(item, indent + 2) for item in value
        ]
        return "[\n" + ",\n".join(entries) + "\n" + " " * indent + "]"
    return compact


def freeze(value):  # noqa: PLR0911, PLR0912
    """Copy arguments at call time, before callers can mutate them."""
    if isinstance(value, xr.Dataset):
        return {"dataset": freeze(value.to_dict())}
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(k): sorted(v.split(",")) if k == "file_types" else freeze(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [freeze(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(freeze(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, BaseException):
        return {"exception": type(value).__name__, "message": str(value)}
    if isinstance(value, type):
        return value.__name__
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "filename_list"):
        return {"filenames": freeze(value.filename_list)}
    if hasattr(value, "name"):
        return {"name": value.name}
    raise TypeError(f"Add an explicit trace representation for {type(value)}")


class Boundary:
    """Provide deterministic external dependencies sharing one ordered trace."""

    def __init__(self, monkeypatch):
        """Start a fresh trace for one invocation."""
        self.monkeypatch = monkeypatch
        self.calls = []

    def function(self, name, result=None, effect=None):
        """Record each call before returning a value or invoking an effect."""

        def invoke(*args, **kwargs):
            self.calls.append([name, freeze(args), freeze(kwargs)])
            if effect is not None:
                return effect(*args, **kwargs)
            return result

        return Mock(side_effect=invoke)

    def patch(self, owner, attribute, result=None, effect=None, name=None):
        """Replace one dependency with a recording mock."""
        mock = self.function(name or attribute, result, effect)
        self.monkeypatch.setattr(owner, attribute, mock)
        return mock


@pytest.fixture
def boundary(monkeypatch):  # noqa: PLR0915
    """Install mocks only outside the two modules under refactoring."""
    recorder = Boundary(monkeypatch)
    for name, value in {
        "SSR_TABLE": "ssr-table",
        "SSM_API_KEY_PARAMETER": "/ssr/api-key",
        "IMAP_DATA_ACCESS_URL": "https://ssr.example",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(imap_data_access, "config", {})
    # Stable provenance, clock, and path; no actual /tmp/ssr files are touched.
    monkeypatch.setattr(
        processing,
        "SOFTWARE_VERSIONS",
        {"imap_processing": "fixture", "imap_data_access": "fixture"},
    )
    monkeypatch.setattr(processing, "SPACECRAFT_CADENCE", timedelta(hours=12))

    class FixturePath(type(Path())):
        def mkdir(self, *args, **kwargs):
            recorder.calls.append(["Path.mkdir", freeze([self, *args]), freeze(kwargs)])

        def stat(self, *args, **kwargs):
            recorder.calls.append(["Path.stat", freeze([self, *args]), freeze(kwargs)])
            return SimpleNamespace(st_size=100)

    monkeypatch.setattr(processing, "DOWNLOAD_DIR", FixturePath("/ssr-fixture"))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            recorder.calls.append(["datetime.now", [], {"tz": str(tz)}])
            return datetime(2026, 11, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(processing, "datetime", Clock)
    recorder.patch(processing.time, "perf_counter", result=100.0)
    recorder.patch(processing.shutil, "rmtree")

    batch = SimpleNamespace(put_item=recorder.function("batch.put_item"))
    writer = Mock()
    writer.__enter__ = recorder.function("batch.__enter__", batch)
    writer.__exit__ = Mock(
        side_effect=lambda exc_type, exc, tb: (
            recorder.calls.append(["batch.__exit__", freeze([exc_type, exc]), {}])
            or False
        )
    )
    table = SimpleNamespace(
        batch_writer=recorder.function("table.batch_writer", writer)
    )
    resource = SimpleNamespace(Table=recorder.function("dynamodb.Table", table))
    ssm = SimpleNamespace(
        get_parameter=recorder.function(
            "ssm.get_parameter", {"Parameter": {"Value": "fixture-key"}}
        )
    )
    recorder.patch(processing.boto3, "resource", result=resource, name="boto3.resource")
    recorder.patch(processing.boto3, "client", result=ssm, name="boto3.client")
    pool = Mock()
    pool.__enter__ = recorder.function("KernelPool.__enter__")
    # Record exception type/message, excluding the nondeterministic traceback.
    pool.__exit__ = recorder.function("KernelPool.__exit__", effect=lambda *args: False)
    pool.__exit__.side_effect = lambda exc_type, exc, tb: (
        recorder.calls.append(["KernelPool.__exit__", freeze([exc_type, exc]), {}])
        or False
    )
    recorder.patch(processing.spiceypy, "KernelPool", result=pool)

    response = SimpleNamespace(
        raise_for_status=recorder.function("response.raise_for_status"),
        json=recorder.function(
            "response.json",
            ["naif0012.tls", "imap_sclk_0000.tsc", "imap_sclk_0001.tsc"],
        ),
    )
    recorder.patch(utils.requests, "get", result=response, name="requests.get")
    # Preserve real metadata parsing, but record constructors and collection methods.
    recorder.patch(utils, "SPICEInput", effect=utils.SPICEInput)
    collection_type = utils.ProcessingInputCollection
    collection = collection_type()
    recorder.patch(utils, "ProcessingInputCollection", result=collection)
    recorder.patch(collection, "add", effect=collection.add, name="collection.add")
    recorder.patch(
        collection, "download_all_files", name="collection.download_all_files"
    )
    recorder.patch(
        collection,
        "get_file_paths",
        result=[FixturePath("/ssr-fixture/naif0012.tls")],
        name="collection.get_file_paths",
    )

    def query(**kwargs):
        if kwargs["table"] == "science":
            return [
                {"start_date": date, "file_path": f"{date}.pkts"}
                for date in ("20261007", "20261008", "20261009")
            ]
        return [
            {
                "start_date": "20261001",
                "version": "v001",
                "end_date": None,
                "file_path": f"{kwargs['descriptor']}.old",
            },
            {
                "start_date": "20261007",
                "version": "v002",
                "end_date": "20261008",
                "file_path": f"{kwargs['descriptor']}.cdf",
            },
            {
                "start_date": "20261008",
                "version": "v003",
                "end_date": "20261007",
                "file_path": "expired.cdf",
            },
        ]

    recorder.patch(
        imap_data_access, "query", effect=query, name="imap_data_access.query"
    )
    recorder.patch(
        imap_data_access,
        "download",
        effect=lambda filename: Path("/ssr-fixture") / filename,
        name="imap_data_access.download",
    )
    epoch = np.array([-30, 0, 30, 86400, 86430]) * 10**9
    packets = xr.Dataset(
        {"src_seq_ctr": ("epoch", [16383, 0, 1, 2, 3])}, coords={"epoch": epoch}
    )
    recorder.patch(utils, "packet_file_to_datasets", result={478: packets})
    recorder.patch(
        processing,
        "str_yyyymmdd_to_ttj2000ns",
        effect=lambda date: (datetime.strptime(date, "%Y%m%d") - DAY).days * DAY_NS,
    )
    recorder.patch(processing, "str_to_et", result=42.0)
    recorder.patch(processing, "et_to_ttj2000ns", result=42)
    recorder.patch(processing, "ttj2000ns_to_et", result=42.0)
    recorder.patch(processing, "et_to_utc", result="fixture-utc")
    recorder.patch(processing, "imap_state", result=np.arange(12).reshape(2, 6))
    product = {
        "met_in_utc": "2026-10-08T12:00:00",
        "instrument": "fixture",
        "mag_hk_status": {"hk": 1},
        "value": Decimal("1.25"),
        "met": 10,
        "apid": 478,
    }
    products = [
        dict(product, met_in_utc="2026-10-07T23:59:59"),
        product,
        dict(product, met_in_utc="2026-10-09T00:00:00"),
    ]
    for name in (
        "process_packet",
        "process_hit",
        "process_swe",
        "process_swapi_ialirt",
    ):
        recorder.patch(processing, name, result=products)
    recorder.patch(processing, "process_codice", result=(products, products))
    recorder.patch(processing, "process_status", result=[product])
    recorder.patch(
        processing, "load_cdf", effect=lambda path: {"calibration": path.name}
    )
    recorder.patch(
        processing.pd, "read_csv", result={"calibration": "csv"}, name="pd.read_csv"
    )
    return SimpleNamespace(
        recorder=recorder, packets=packets, response=response, ssm=ssm, batch=batch
    )


@pytest.mark.parametrize(
    ("instrument", "scenario"), CASES, ids=[f"{i}-{s}" for i, s in CASES]
)
def test_lambda_dependency_contract(boundary, monkeypatch, instrument, scenario):  # noqa: PLR0912
    """Compare the full ordered boundary trace with the pre-refactor baseline."""
    monkeypatch.setenv("INSTRUMENT", instrument)
    if scenario in {"empty", "mixed"}:
        processing.process_hit.side_effect = boundary.recorder.function(
            "process_hit",
            [] if scenario == "empty" else [{"met_in_utc": "2026-10-08T12:00:00"}, {}],
        ).side_effect
    elif scenario in {"no_packets", "late_start", "early_end", "sequence_gap"}:
        packets = boundary.packets
        if scenario == "no_packets":
            packets = packets.isel(epoch=[])
        elif scenario == "late_start":
            packets = packets.assign_coords(
                epoch=np.array([-30, 61, 90, 86400, 86430]) * 10**9
            )
        elif scenario == "early_end":
            packets = packets.isel(epoch=slice(0, 3))
        else:
            packets = packets.assign(src_seq_ctr=("epoch", [16383, 0, 2, 3, 4]))
        utils.packet_file_to_datasets.side_effect = boundary.recorder.function(
            "packet_file_to_datasets", {478: packets}
        ).side_effect
    elif scenario in {"missing_l0", "missing_ancillary"}:
        original_query = imap_data_access.query.side_effect

        def missing_query(**kwargs):
            if scenario == "missing_ancillary" and kwargs["table"] == "science":
                return original_query(**kwargs)
            return boundary.recorder.function("imap_data_access.query", [])(**kwargs)

        imap_data_access.query.side_effect = missing_query
    elif scenario in {"http_error", "ssm_error", "processor_error", "write_error"}:
        target = {
            "http_error": boundary.response.raise_for_status,
            "ssm_error": boundary.ssm.get_parameter,
            "processor_error": processing.process_hit,
            "write_error": boundary.batch.put_item,
        }[scenario]
        original = target.side_effect

        def failing(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError(scenario)

        target.side_effect = failing

    error = None
    try:
        result = processing.lambda_handler({"day": "20261008", "version": "v001"}, {})
    except Exception as exc:  # Capture the public failure contract as well as calls.
        error = freeze(exc)
        result = None
    actual = {
        "calls": boundary.recorder.calls,
        "result": freeze(result),
        "error": error,
        "config": freeze(imap_data_access.config),
    }
    expected_exception = None
    if instrument == "unknown":
        expected_exception = "ValueError"
    elif scenario in {"missing_l0", "missing_ancillary"}:
        expected_exception = "FileNotFoundError"
    elif scenario in {"no_packets", "late_start", "early_end", "sequence_gap"}:
        expected_exception = "IalirtDataGapError"
    elif scenario.endswith("_error"):
        expected_exception = "RuntimeError"
    assert (error["exception"] if error else None) == expected_exception
    baseline = TRACE_DIR / f"{instrument}-{scenario}.json"
    if os.environ.get("SSR_UPDATE_TRACES") == "1":
        baseline.write_text(format_trace(actual) + "\n")
    assert baseline.exists(), f"Missing reviewed baseline: {baseline}"
    assert actual == json.loads(baseline.read_text())
