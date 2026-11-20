"""Production metadata retrieval for dependency selection."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING

from dagster import MetadataValue
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from sds_data_manager.lambda_code.SDSCode.database import models
from sds_data_manager.orchestration import repoint_file, spice, spin
from sds_data_manager.orchestration.dependency_metadata import AncillaryRecord

if TYPE_CHECKING:
    from dagster import AssetExecutionContext

    from sds_data_manager.orchestration.types import DependencyNode


class DagsterDatabaseSource:
    """Read dependencies through the existing database and materialization queries.

    The caller owns the session lifetime. Science file groups retain partition
    order; None means missing file_names metadata, while [] means present but empty.
    """

    def __init__(self, session: Session, context: AssetExecutionContext) -> None:
        """Bind retrieval to one run's session and execution context."""
        self.session = session
        self.context = context

    @staticmethod
    def _file_groups(
        metadata_list: list[Mapping[str, MetadataValue]],
    ) -> list[list[str] | None]:
        groups = []
        for metadata in metadata_list:
            if "file_names" not in metadata:
                groups.append(None)
                continue
            names = metadata["file_names"].value
            groups.append([names] if isinstance(names, str) else names)
        return groups

    def ancillary_records(
        self,
        dependency: DependencyNode,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> list[AncillaryRecord]:
        """Return overlapping ancillary records at the latest version per start date."""
        table = models.AncillaryFiles
        type_specific_conditions = []
        type_specific_conditions.append(
            and_(
                table.start_date < target_end,
                or_(table.end_date >= target_start, table.end_date.is_(None)),
            )
        )
        filter_conditions = [
            table.instrument == dependency.source,
            table.descriptor == dependency.descriptor,
            *type_specific_conditions,
        ]
        max_version_query = (
            self.session.query(
                table.start_date,
                func.max(table.version).label("latest_version"),
            )
            .filter(*filter_conditions)
            .group_by(table.start_date)
            .subquery()
        )
        # Query records
        records = (
            self.session.query(table)
            .join(
                max_version_query,
                (table.start_date == max_version_query.c.start_date)
                & (table.version == max_version_query.c.latest_version),
            )
            .filter(*filter_conditions)
            .all()
        )

        return [AncillaryRecord(r.file_path, r.start_date) for r in records]

    def science_files(
        self,
        dependency: DependencyNode,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> list[list[str] | None]:
        """Return plain filename groups for overlapping science partitions."""
        return self._file_groups(
            dependency.get_all_files_in_time_range(
                self.context, target_start, target_end
            )
        )

    def science_files_by_repoint(
        self, dependency: DependencyNode, repoints: list[int]
    ) -> list[list[str] | None]:
        """Return plain filename groups for the requested repoints."""
        return self._file_groups(
            dependency.get_all_files_by_repoint_numbers(self.context, repoints)
        )

    def spin_files(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str] | None:
        """Return spin filenames in helper priority order, or None."""
        return spin.get_upstream_dependency_inputs_spin(
            target_start, target_end, False, self.session
        )

    def repoint_files(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str] | None:
        """Return the helper's single-element repoint list, or None."""
        return repoint_file.get_upstream_dependency_inputs_repoint(
            target_start, target_end, self.session
        )

    def spice_files(
        self,
        spice_types: list[str],
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> tuple[list[str], list[str]] | tuple[None, None]:
        """Return selected filenames and missing kernel types."""
        return spice.get_upstream_dependency_inputs_spice(
            spice_types, target_start, target_end
        )

    def available_repoints(self, dependency: DependencyNode) -> list[int]:
        """Retrieve available repoints using the existing production query."""
        repoints = []
        materialized_partitions = self.context.instance.get_materialized_partitions(
            dependency.to_dagster_asset()
        )
        for m in materialized_partitions:
            parts = m.split("_")
            if "repoint" in parts[0]:
                repoints.append(int(parts[0][7:]))
        return repoints

    def inprogress_repoints(self, dependency: DependencyNode) -> list[int]:
        """Retrieve inprogress repoints using the existing production query."""
        results = (
            self.session.query(models.ProcessingJob.repointing)
            .filter(
                models.ProcessingJob.instrument == dependency.source,
                models.ProcessingJob.data_level == dependency.data_type,
                models.ProcessingJob.descriptor == dependency.descriptor,
                models.ProcessingJob.status == models.Status.INPROGRESS,
                models.ProcessingJob.repointing.isnot(None),
            )
            .distinct()
            .order_by(models.ProcessingJob.repointing)
            .all()
        )
        return [rp[0] for rp in results]

    def pointing_exists(self, repoint: int) -> bool:
        """Retrieve pointing exists using the existing production query."""
        pointing_record = (
            self.session.query(models.PointingTable)
            .filter(models.PointingTable.pointing_id == repoint)
            .first()
        )
        return pointing_record is not None
