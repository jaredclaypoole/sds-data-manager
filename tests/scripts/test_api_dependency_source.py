"""Offline coverage for the API source and shared-handler local preparation."""

import datetime as dt
import json
import subprocess
import sys
from unittest.mock import Mock

import imap_data_access as ida
import pytest
from imap_data_access import processing_input

from sds_data_manager.orchestration.api_dependency_source import (
    APIDependencySource,
    DataAccessClient,
    parse_time,
)
from sds_data_manager.orchestration.dependency import DependencyConfigReader
from sds_data_manager.orchestration.dependency_handler_registry import (
    DependencyHandlerRegistry,
)
from sds_data_manager.orchestration.local_processing import main, prepare_local_job
from sds_data_manager.orchestration.types import DependencyNode

START = parse_time("20260101")
END = parse_time("20260102")


def science(
    day="20260101",
    version="001.0001",
    level="l1a",
    descriptor="norm-magi",
    repoint=None,
):
    tag = f"-repoint{repoint:05d}" if repoint is not None else ""
    return dict(
        file_path=f"imap_mag_{level}_{descriptor}_{day}{tag}_v{version}.cdf",
        start_date=day,
        version=f"v{version}",
        repointing=repoint,
    )


def test_science_latest_exclusive_end_and_repoint_filter():
    client = Mock()
    client.query.return_value = [
        science(),
        science(version="001.0002"),
        science(day="20260102"),
        science(repoint=1),
        science(repoint=2),
    ]
    source = APIDependencySource(client, repointing=1)
    node = DependencyNode("mag", "l1a", "norm-magi")
    groups = source.science_files(node, START, END)
    assert {f for group in groups for f in group} == {
        science(version="001.0002")["file_path"],
        science(repoint=1)["file_path"],
    }
    assert client.query.call_args.kwargs["version"] == "latest"
    assert parse_time("2026-01-01T01:00:00+01:00") == START


def test_ancillary_overlap_and_latest_version_per_start():
    client = Mock()
    client.query.return_value = [
        dict(file_path="old", start_date="20250101", end_date=None, version="v099"),
        dict(file_path="new", start_date="20251201", end_date=None, version="v001"),
        dict(file_path="newest", start_date="20251201", end_date=None, version="v002"),
        dict(
            file_path="expired",
            start_date="20241201",
            end_date="20251231",
            version="v999",
        ),
        dict(file_path="future", start_date="20260102", end_date=None, version="v999"),
    ]
    records = APIDependencySource(client).ancillary_records(
        DependencyNode("mag", "ancillary", "l1b-calibration"), START, END
    )
    assert {r.file_path for r in records} == {"old", "newest"}
    assert "start_date" not in client.query.call_args.kwargs


def test_spin_split_supersession_and_priority():
    client = Mock()
    client.records.return_value = [
        dict(
            file_path="old",
            version=1,
            start_date="2025-12-31, 00:00:00",
            end_date="2026-01-02, 00:00:00",
            ingestion_date="2026-01-03, 00:00:00",
        ),
        dict(
            file_path="new2",
            version=10,
            start_date="2026-01-02, 00:00:00",
            end_date="2026-01-03, 00:00:00",
            ingestion_date="2026-01-04, 00:00:00",
        ),
        dict(
            file_path="new1",
            version=10,
            start_date="2025-12-31, 00:00:00",
            end_date="2026-01-01, 00:00:00",
            ingestion_date="2026-01-04, 00:00:00",
        ),
    ]
    assert APIDependencySource(client).spin_files(START, END) == ["new1", "new2"]
    client.records.assert_called_once_with("spin-table")


def test_repoint_latest_file_requires_coverage():
    client = Mock()
    client.records.return_value = [
        dict(file_path="imap_2025_365_01.repoint.csv", end_date="2026-01-01, 00:00:00"),
        dict(file_path="imap_2025_364_01.repoint.csv", end_date="2026-01-03, 00:00:00"),
    ]
    source = APIDependencySource(client)
    assert source.repoint_files(START, END) is None
    assert source.repoint_files(dt.datetime.min, dt.datetime.min) == [
        "imap_2025_365_01.repoint.csv"
    ]


def test_spice_uses_metakernel_and_shared_type_checks():
    client = Mock()
    client.files.return_value = ["naif0012.tls"]
    source = APIDependencySource(client)
    assert source.spice_files(["leapseconds"], START, END) == (["naif0012.tls"], [])
    assert client.files.call_args.kwargs["list_files"] == "true"
    assert source.spice_files(["spacecraft_clock"], START, END)[1] == [
        "spacecraft_clock"
    ]
    client.files.side_effect = ida.io.IMAPDataAccessError("503 unavailable")
    with pytest.raises(ida.io.IMAPDataAccessError):
        source.spice_files(["leapseconds"], START, END)


def test_local_hi_uses_shared_neighbors_without_readiness_queries():
    job = DependencyConfigReader().config[("hi", "l1b", "45sensor-goodtimes")]
    source = Mock(spec=APIDependencySource)
    source.science_files.side_effect = lambda dep, *args: [
        [
            f"imap_hi_{dep.data_type}_{dep.descriptor}_20260101-repoint00005_v001.0001.cdf"
        ]
    ]
    source.available_repoints.return_value = [4, 5, 6]
    source.science_files_by_repoint.return_value = [
        ["imap_hi_l1b_45sensor-de_20260101-repoint00004_v001.0001.cdf"]
    ]
    handler = DependencyHandlerRegistry.get_handler(
        job, source, wait_for_dependencies=False
    )
    inputs = handler.get_science_files_inputs(START, END, 5)
    assert any("repoint00004" in f for inp in inputs for f in inp.filename_list)
    source.inprogress_repoints.assert_not_called()
    source.pointing_exists.assert_not_called()
    production = DependencyHandlerRegistry.get_handler(job, source)
    source.inprogress_repoints.return_value = [4]
    from sds_data_manager.orchestration.job_dependencies import MissingDependenciesError

    with pytest.raises(MissingDependenciesError, match="in progress"):
        production.get_science_files_inputs(START, END, 5)


def test_local_json_round_trip_versions_and_no_overwrite(tmp_path):
    job = DependencyConfigReader().config[("mag", "l1b", "norm-magi")]
    client = Mock()

    def query(**params):
        if params.get("table") == "ancillary":
            return [
                dict(
                    file_path="imap_mag_l1b-calibration_20251201_v001.csv",
                    start_date="20251201",
                    end_date=None,
                    version="v001",
                )
            ]
        if params["data_level"] == "l1a":
            return [science()]
        return [science(level="l1b", version="001.0007")]

    client.query.side_effect = query
    client.files.return_value = ["naif0012.tls", "imap_sclk_0001.tsc"]
    source = APIDependencySource(client)
    path = prepare_local_job(job, START, END, tmp_path, source=source)
    payload = json.loads(path.read_text())
    collection = processing_input.ProcessingInputCollection()
    collection.deserialize(json.dumps(payload["dependency"]))
    assert len(collection.processing_input) == 3
    assert payload["version"] == {"norm-magi": {"major_version": 1, "minor_version": 8}}
    assert path.name == "imap_mag_l1b_norm-magi_20260101_v001.0008.json"
    with pytest.raises(FileExistsError):
        prepare_local_job(job, START, END, tmp_path, source=source)


def test_validation_happens_before_api_calls(tmp_path):
    source = APIDependencySource(Mock())
    job = DependencyConfigReader().config[("mag", "l1b", "norm-magi")]
    with pytest.raises(ValueError, match="Start must be before"):
        prepare_local_job(job, END, START, tmp_path, source=source)
    source.client.query.assert_not_called()
    with pytest.raises(SystemExit):
        main(
            [
                "--instrument",
                "mag",
                "--data-level",
                "l1b",
                "--descriptor",
                "unknown",
                "--start-date",
                "20260101",
                "--end-date",
                "20260102",
            ]
        )


def test_authentication_and_error_propagation(monkeypatch):
    monkeypatch.setitem(ida.config, "API_KEY", "test-key")
    monkeypatch.setitem(ida.config, "DATA_ACCESS_URL", "https://example.com")
    response = Mock()
    response.json.return_value = []
    send = Mock(return_value=response)
    monkeypatch.setattr("requests.Session.send", send)
    assert DataAccessClient().records("repoint-table") == []
    prepared = send.call_args.args[0]
    assert prepared.url == "https://example.com/api-key/repoint-table"
    assert prepared.headers["x-api-key"] == "test-key"


def test_cli_help_has_no_aws_or_dagster_runtime():
    script = """
import sys
from sds_data_manager.orchestration.local_processing import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
for name in ('dagster', 'boto3', 'sqlalchemy', 'spiceypy'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)  # noqa: S603
