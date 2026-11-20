"""Plain dependency metadata and the interface implemented by metadata sources."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sds_data_manager.orchestration.types import DependencyNode


@dataclass(frozen=True)
class AncillaryRecord:
    """Ancillary metadata used by dependency selection, without an ORM object."""

    file_path: str
    start_date: datetime.datetime


class DependencySource(Protocol):
    """Retrieve metadata without exposing database or Dagster objects to selection."""

    def ancillary_records(
        self,
        dependency: DependencyNode,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> list[AncillaryRecord]:
        """Return overlapping ancillary records at the latest version per start date."""
        ...

    def science_files(
        self,
        dependency: DependencyNode,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> list[list[str] | None]:
        """Return partition-ordered groups; None means missing file_names metadata."""
        ...

    def science_files_by_repoint(
        self, dependency: DependencyNode, repoints: list[int]
    ) -> list[list[str] | None]:
        """Return filename groups for the requested repoints."""
        ...

    def spin_files(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str] | None:
        """Return spin filenames in priority order, or None when none exist."""
        ...

    def repoint_files(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str] | None:
        """Return a single-element filename list, or None for insufficient coverage."""
        ...

    def spice_files(
        self,
        spice_types: list[str],
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> tuple[list[str], list[str]] | tuple[None, None]:
        """Return filenames and missing types, or (None, None) on request failure."""
        ...

    def available_repoints(self, dependency: DependencyNode) -> list[int]:
        """Return materialized repoint numbers for a dependency."""
        ...

    def inprogress_repoints(self, dependency: DependencyNode) -> list[int]:
        """Return repoint numbers with in-progress processing jobs."""
        ...

    def pointing_exists(self, repoint: int) -> bool:
        """Return whether the pointing has been recorded."""
        ...
