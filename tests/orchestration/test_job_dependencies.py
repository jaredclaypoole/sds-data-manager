"""Check the boundary between dependency selection and production retrieval."""

import datetime
import subprocess
import sys
from unittest.mock import Mock

import pytest
from dagster import MetadataValue

from sds_data_manager.orchestration import imap_job
from sds_data_manager.orchestration.custom_behavior.hi import HiGoodtimesJob
from sds_data_manager.orchestration.custom_behavior.mag import MagL1DJob
from sds_data_manager.orchestration.dependency_source import DagsterDatabaseSource
from sds_data_manager.orchestration.imap_dagster import dependency_config
from sds_data_manager.orchestration.job_dependencies import (
    JobDependencyHandler,
    MissingDependenciesError,
)

START = datetime.datetime(2026, 1, 1)
END = START + datetime.timedelta(days=1)


def test_standalone_selection_and_legacy_names():
    """Select dependencies using plain source results without Dagster or SQLAlchemy."""
    source = Mock()
    source.science_files.return_value = [["imap_mag_l1a_norm-magi_20260101_v002.cdf"]]
    handler = JobDependencyHandler(
        dependency_config.config[("mag", "l1b", "norm-magi")], source
    )
    inputs = handler.get_science_files_inputs(START, END)
    assert inputs[0].filename_list == ["imap_mag_l1a_norm-magi_20260101_v001.0002.cdf"]
    source.science_files.return_value = []
    with pytest.raises(imap_job.MissingDependenciesError):
        handler.get_science_files_inputs(START, END)
    assert imap_job.MissingDependenciesError is MissingDependenciesError


def test_source_normalizes_science_groups_without_losing_missing_metadata():
    """Distinguish absent file metadata from a present empty filename list."""
    dependency = Mock()
    dependency.get_all_files_in_time_range.return_value = [
        {},
        {"file_names": MetadataValue.json([])},
        {"file_names": MetadataValue.text("one.cdf")},
        {"file_names": MetadataValue.json(["two.cdf", "three.cdf"])},
    ]
    source = DagsterDatabaseSource(None, Mock())
    assert source.science_files(dependency, START, END) == [
        None,
        [],
        ["one.cdf"],
        ["two.cdf", "three.cdf"],
    ]


def test_binding_preserves_custom_spice_padding_and_does_not_mutate_handler():
    """Bind sources independently while preserving registered subclass behavior."""
    handler = MagL1DJob(dependency_config.config[("mag", "l1d", "norm-srf")])
    source = Mock()
    source.spice_files.return_value = (["naif0012.tls"], [])
    bound = handler.with_dependency_source(source)
    assert bound.get_spice_file_inputs(START, END) == ["naif0012.tls"]
    source.spice_files.assert_called_once_with(
        handler.job_config.spice_types,
        START - datetime.timedelta(minutes=30),
        END + datetime.timedelta(minutes=30),
    )
    assert not hasattr(handler, "dependency_source")
    assert handler.with_dependency_source(Mock()).dependency_source is not source


def test_ancillary_query_and_selection_preserve_overlap_and_versions():
    """Keep latest-per-start query results separate from newest-start selection."""
    from sds_data_manager.lambda_code.SDSCode.database.models import AncillaryFiles
    from tests.conftest import in_memory_session

    config = dependency_config.config[("mag", "l1b", "norm-magi")]
    dependency = config.ancillary_inputs[0]
    older = START - datetime.timedelta(days=10)
    newer = START - datetime.timedelta(days=1)

    def record(date, version, end_date=None, descriptor=None):
        return AncillaryFiles(
            file_path=(
                f"imap_mag_{descriptor or dependency.descriptor}_"
                f"{date:%Y%m%d}_{version}.csv"
            ),
            instrument="mag",
            descriptor=descriptor or dependency.descriptor,
            start_date=date,
            end_date=end_date,
            version=version,
            extension="csv",
        )

    old = record(older, "v099")
    selected = record(newer, "v002")
    with in_memory_session() as session:
        session.add_all(
            [
                old,
                record(newer, "v001"),
                selected,
                record(END, "v003"),
                record(older, "v100", end_date=START - datetime.timedelta(seconds=1)),
                record(newer, "v009", descriptor="other-calibration"),
            ]
        )
        session.flush()
        source = DagsterDatabaseSource(session, None)
        records = source.ancillary_records(dependency, START, END)
        assert {r.file_path for r in records} == {old.file_path, selected.file_path}
        assert not hasattr(records[0], "_sa_instance_state")
        handler = JobDependencyHandler(config, source)
        assert handler.get_ancillary_files_inputs(START, END)[0].filename_list == [
            selected.file_path
        ]


@pytest.mark.parametrize(
    ("kind", "method"),
    [
        ("spin", "get_spin_files_inputs"),
        ("repoint", "get_repoint_file_inputs"),
        ("spice", "get_spice_file_inputs"),
    ],
)
def test_source_override_preserves_missing_input_checks(kind, method):
    """Keep required-input checks in selection instead of retrieval."""
    config = Mock()
    config.spice_types = ["leapseconds"]
    source = Mock()
    getattr(source, f"{kind}_files").return_value = (
        ([], ["leapseconds"]) if kind == "spice" else None
    )
    with pytest.raises(MissingDependenciesError):
        getattr(JobDependencyHandler(config, source), method)(START, END)


def test_hi_neighbor_readiness_uses_plain_source_results():
    """Preserve nearest-neighbor selection and the running-neighbor gate."""
    source = Mock()
    source.available_repoints.return_value = list(range(1, 12))
    source.inprogress_repoints.return_value = []
    handler = HiGoodtimesJob(
        dependency_config.config[("hi", "l1b", "45sensor-goodtimes")]
    ).with_dependency_source(source)
    dependency = handler.job_config.science_inputs[0]
    assert set(handler.get_n_nearest_repoints(dependency, 5)) == {
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
    }
    source.inprogress_repoints.return_value = [6]
    assert handler.get_n_nearest_repoints(dependency, 5) is None


def test_selection_module_import_has_no_orchestration_runtime_dependencies():
    """Import the reusable handler without loading Dagster, SQLAlchemy, or boto3."""
    script = """
import sys
from sds_data_manager.orchestration.job_dependencies import JobDependencyHandler
assert 'dagster' not in sys.modules
assert 'sqlalchemy' not in sys.modules
assert 'boto3' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def test_complete_dependency_collection_with_plain_source():
    """Prepare a real configured job with no session or execution context."""
    source = Mock()
    source.science_files.return_value = [["imap_mag_l0_raw_20260101_v001.pkts"]]
    source.spice_files.return_value = (["naif0012.tls"], [])
    handler = JobDependencyHandler(
        dependency_config.config[("mag", "l1a", "all")], source
    )
    inputs = handler.get_dependencies(START, END)
    assert [inp.filename_list for inp in inputs.processing_input] == [
        ["imap_mag_l0_raw_20260101_v001.0001.pkts"],
        ["naif0012.tls"],
    ]
