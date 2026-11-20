"""Dagster registration for Hi goodtimes processing."""

from sds_data_manager.orchestration.custom_behavior.dependencies import (
    HI_GOODTIMES_NUM_NEAREST_REPOINTS,
    HiGoodtimesDependencies,
)
from sds_data_manager.orchestration.imap_job import IMAPJobHandler
from sds_data_manager.orchestration.job_handler_registry import JobBuilderRegistry

__all__ = ["HI_GOODTIMES_NUM_NEAREST_REPOINTS", "HiGoodtimesJob"]


@JobBuilderRegistry.register_dependencies(HiGoodtimesDependencies)
class HiGoodtimesJob(HiGoodtimesDependencies, IMAPJobHandler):
    """Run Hi goodtimes with shared dependency selection."""
