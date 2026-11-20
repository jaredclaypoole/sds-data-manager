"""Compatibility checks for dependency gathering inherited by job handlers."""

import datetime
from unittest.mock import Mock

import pytest

from sds_data_manager.orchestration import imap_job, job_dependencies
from sds_data_manager.orchestration.custom_behavior.mag import MagL1DJob
from sds_data_manager.orchestration.imap_dagster import dependency_config


def test_custom_spice_window_through_inherited_dependency_gathering(monkeypatch):
    """Preserve custom overrides and super calls through the extracted base."""
    handler = MagL1DJob(dependency_config.config[("mag", "l1d", "norm-srf")])
    monkeypatch.setattr(handler, "get_science_files_inputs", Mock(return_value=[]))
    monkeypatch.setattr(handler, "get_ancillary_files_inputs", Mock(return_value=[]))
    monkeypatch.setattr(handler, "get_spin_files_inputs", Mock(return_value=[]))
    monkeypatch.setattr(handler, "get_repoint_file_inputs", Mock(return_value=[]))
    query_spice = Mock(return_value=(["naif0012.tls"], []))
    monkeypatch.setattr(
        job_dependencies.spice, "get_upstream_dependency_inputs_spice", query_spice
    )
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    end = start + datetime.timedelta(days=1)

    inputs = handler.get_dependencies(Mock(), Mock(), start, end)

    query_spice.assert_called_once_with(
        handler.job_config.spice_types,
        start - datetime.timedelta(minutes=30),
        end + datetime.timedelta(minutes=30),
    )
    assert inputs.processing_input[0].filename_list == ["naif0012.tls"]


def test_missing_dependency_exception_keeps_original_import():
    """Allow existing callers to catch exceptions raised by the extracted base."""
    handler = imap_job.IMAPJobHandler(
        dependency_config.config[("mag", "l1b", "norm-magi")]
    )
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    context = Mock()
    context.instance.get_materialized_partitions.return_value = set()

    with pytest.raises(imap_job.MissingDependenciesError):
        handler.get_dependencies(
            Mock(), context, start, start + datetime.timedelta(days=1)
        )

    assert (
        imap_job.MissingDependenciesError is job_dependencies.MissingDependenciesError
    )
