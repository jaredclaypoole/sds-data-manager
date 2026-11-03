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
from unittest.mock import DEFAULT, Mock

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


def format_trace(value, indent=0, width=100):
    """Render JSON with two-space indentation and short containers inline."""
    compact = json.dumps(value, sort_keys=True)
    if indent + len(compact) <= width:
        return compact
    if isinstance(value, dict) and value:
        entries = []
        for index, key in enumerate(sorted(value)):
            prefix = json.dumps(key) + ": "
            comma = int(index < len(value) - 1)
            entries.append(
                " " * (indent + 2)
                + prefix
                + format_trace(value[key], indent + 2, 100 - len(prefix) - comma)
            )
        return "{\n" + ",\n".join(entries) + "\n" + " " * indent + "}"
    if isinstance(value, list) and value:
        entries = [
            " " * (indent + 2)
            + format_trace(item, indent + 2, 100 - int(index < len(value) - 1))
            for index, item in enumerate(value)
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
            return DEFAULT

        return Mock(return_value=result, side_effect=invoke)

    def context_exit(self, name):
        """Record exception type/message without the nondeterministic traceback."""

        def exit_context(exc_type, exc, tb):
            self.calls.append([name, freeze([exc_type, exc]), {}])
            return False

        return Mock(side_effect=exit_context)

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
    monkeypatch.setattr(
        utils, "imap_module_directory", Path("/imap-processing-fixture")
    )
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
    writer.__exit__ = recorder.context_exit("batch.__exit__")
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
    pool.__exit__ = recorder.context_exit("KernelPool.__exit__")
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
    # Distinct files overlap at midnight and just before the next midnight.
    # Include both buffer boundaries and packets just outside them; leave the
    # final file unsorted so the processor arguments also characterize sorting.
    packets_by_filename = {
        filename: xr.Dataset(
            {"src_seq_ctr": ("epoch", counters)},
            coords={"epoch": np.array(seconds) * 10**9},
        )
        for filename, seconds, counters in (
            ("20261007.pkts", [-601, -600, -30, 0], [16381, 16382, 16383, 0]),
            ("20261008.pkts", [0, 30, 86399], [0, 1, 2]),
            ("20261009.pkts", [86430, 86399, 86400, 87000, 87001], [4, 2, 3, 5, 6]),
        )
    }

    def decode_packets(path, *args, **kwargs):
        return {478: packets_by_filename[path.name].copy(deep=True)}

    recorder.patch(utils, "packet_file_to_datasets", effect=decode_packets)
    recorder.patch(
        processing,
        "str_yyyymmdd_to_ttj2000ns",
        effect=lambda date: (datetime.strptime(date, "%Y%m%d") - DAY).days * DAY_NS,
    )
    # Use a simple fixture time scale relative to DAY, with distinct outputs
    # for each timestamp, rather than reproducing real SPICE time conversions.
    recorder.patch(
        processing,
        "str_to_et",
        effect=lambda value: (datetime.fromisoformat(value) - DAY).total_seconds(),
    )
    recorder.patch(processing, "et_to_ttj2000ns", effect=lambda et: round(et * 10**9))
    recorder.patch(processing, "ttj2000ns_to_et", effect=lambda ns: ns / 10**9)
    recorder.patch(
        processing,
        "et_to_utc",
        effect=lambda et: (DAY + timedelta(seconds=float(et))).isoformat(
            timespec="microseconds"
        ),
    )

    def state(ets, ref_frame, observer):
        offset = {
            processing.SpiceFrame.IMAP_GSM: 100,
            processing.SpiceFrame.IMAP_GSE: 200,
        }
        return np.asarray(ets)[:, None] / 3600 + np.arange(6) + offset[ref_frame]

    recorder.patch(processing, "imap_state", effect=state)
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
    recorder.patch(
        processing,
        "process_codice",
        result=(
            [dict(item, instrument="codice_lo") for item in products],
            [dict(item, instrument="codice_hi") for item in products],
        ),
    )
    recorder.patch(processing, "process_status", result=[product])
    recorder.patch(
        processing, "load_cdf", effect=lambda path: {"calibration": path.name}
    )
    recorder.patch(
        processing.pd, "read_csv", result={"calibration": "csv"}, name="pd.read_csv"
    )
    return SimpleNamespace(
        recorder=recorder,
        packets_by_filename=packets_by_filename,
        response=response,
        ssm=ssm,
        batch=batch,
    )


@pytest.mark.parametrize(
    ("instrument", "scenario"), CASES, ids=[f"{i}-{s}" for i, s in CASES]
)
def test_lambda_dependency_contract(boundary, monkeypatch, instrument, scenario):
    """Compare the full ordered boundary trace with the pre-refactor baseline."""
    monkeypatch.setenv("INSTRUMENT", instrument)
    if scenario in {"empty", "mixed"}:
        processing.process_hit.return_value = (
            [] if scenario == "empty" else [{"met_in_utc": "2026-10-08T12:00:00"}, {}]
        )
    elif scenario in {"no_packets", "late_start", "early_end", "sequence_gap"}:
        for filename, source_packets in boundary.packets_by_filename.items():
            epoch = source_packets["epoch"].values
            if scenario == "no_packets":
                packets = source_packets.isel(epoch=[])
            elif scenario == "late_start":
                shifted = np.where(epoch == 0, 61 * 10**9, epoch)
                packets = source_packets.assign_coords(
                    epoch=np.where(epoch == 30 * 10**9, 90 * 10**9, shifted)
                )
            elif scenario == "early_end":
                packets = source_packets.isel(epoch=epoch < DAY_NS)
            else:
                packets = source_packets.assign(
                    src_seq_ctr=source_packets["src_seq_ctr"] + (epoch >= 30 * 10**9)
                )
            boundary.packets_by_filename[filename] = packets
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
    baseline = TRACE_DIR / f"{instrument}-{scenario}.json"
    if os.environ.get("SSR_UPDATE_TRACES") == "1":
        baseline.write_text(format_trace(actual) + "\n")
    assert baseline.exists(), f"Missing reviewed baseline: {baseline}"
    assert actual == json.loads(baseline.read_text())
