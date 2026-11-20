"""Verify shared dependency construction and production registration."""

import datetime
import subprocess
import sys
from unittest.mock import Mock

import pytest

from sds_data_manager.orchestration.custom_behavior.dependencies import (
    HiGoodtimesDependencies,
    L3Dependencies,
    MagL1CDependencies,
    MagL1DDependencies,
    PointingAttitudeDependencies,
)
from sds_data_manager.orchestration.dependency import DependencyConfigReader
from sds_data_manager.orchestration.dependency_handler_registry import (
    DependencyHandlerRegistry,
)
from sds_data_manager.orchestration.job_dependencies import JobDependencyHandler
from sds_data_manager.orchestration.job_handler_registry import JobBuilderRegistry


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (("mag", "l1c", "norm-magi"), MagL1CDependencies),
        (("mag", "l1d", "norm-srf"), MagL1DDependencies),
        (("hi", "l1b", "45sensor-goodtimes"), HiGoodtimesDependencies),
        (("hi", "l3", "sp-maps"), L3Dependencies),
        (("spacecraft", "l1a", "pointing-attitude"), PointingAttitudeDependencies),
        (("mag", "l1b", "norm-magi"), JobDependencyHandler),
    ],
)
def test_shared_construction_preserves_configuration_and_source(key, expected):
    config = DependencyConfigReader().config[key]
    source = Mock()
    handler = DependencyHandlerRegistry.get_handler(config, source)
    assert type(handler) is expected
    assert handler.job_config is config
    assert handler.dependency_source is source
    assert not hasattr(handler, "dagster_job")


def test_production_handlers_inherit_every_registered_dependency_rule():
    reader = DependencyConfigReader()
    for key, dependency_handler in DependencyHandlerRegistry._registry.items():
        assert issubclass(JobBuilderRegistry._registry[key], dependency_handler), key
        # Preserve registrations whose products are not currently configured.
        if key in reader.config:
            production = JobBuilderRegistry.get_builder(reader.config[key])
            assert isinstance(production, dependency_handler), key
            assert production.job_config is reader.config[key]
            assert production.dagster_job is not None


def test_shared_mag_previous_day_uses_the_same_selection_rule():
    config = DependencyConfigReader().config[("mag", "l1c", "norm-mago")]
    source = Mock()
    previous = "imap_mag_l1c_norm-mago_20260101_v001.cdf"

    def files(dependency, start, end):
        if dependency.data_type == "l1c":
            assert start == datetime.datetime(2026, 1, 1)
            assert end == datetime.datetime(2026, 1, 2)
            return [[previous]]
        return [
            [
                f"imap_mag_{dependency.data_type}_{dependency.descriptor}_20260102_v001.cdf"
            ]
        ]

    source.science_files.side_effect = files
    handler = DependencyHandlerRegistry.get_handler(config, source)
    inputs = handler.get_science_files_inputs(
        datetime.datetime(2026, 1, 2), datetime.datetime(2026, 1, 3)
    )
    assert inputs[-1].filename_list == ["imap_mag_l1c_norm-mago_20260101_v001.0001.cdf"]


def test_configuration_and_registered_handlers_load_without_runtime_dependencies():
    script = """
import sys
from unittest.mock import Mock
from sds_data_manager.orchestration.dependency import DependencyConfigReader
from sds_data_manager.orchestration.dependency_handler_registry import (
    DependencyHandlerRegistry,
)
reader = DependencyConfigReader()
for key in reader.config:
    handler = DependencyHandlerRegistry.get_handler(reader.config[key], Mock())
    assert handler.job_config is reader.config[key]
for module in ('dagster', 'boto3', 'sqlalchemy', 'spiceypy',
               'sds_data_manager.orchestration.imap_job'):
    assert module not in sys.modules, module
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
