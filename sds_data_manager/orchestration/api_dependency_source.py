"""Resolve dependency metadata through authenticated imap-data-access requests."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imap_data_access as ida
import requests
from imap_data_access.file_validation import Version

from sds_data_manager.orchestration.dependency_file_selection import (
    _seconds_since_j2000,
    check_for_missing_kernels,
    select_spin_records,
)
from sds_data_manager.orchestration.dependency_metadata import AncillaryRecord
from sds_data_manager.orchestration.types import DependencyNode

# External API JSON has different record shapes for science, ancillary and tables.
APIRecord = dict[str, Any]


def parse_time(value: str) -> dt.datetime:
    """Normalize date/ISO/table timestamps to naive UTC for metadata comparisons."""
    result = dt.datetime.fromisoformat(value.replace(", ", "T"))
    if result.tzinfo is not None:
        result = result.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return result


def record_filename(record: APIRecord) -> str:
    """Extract a basename from either API record format."""
    return Path(record.get("file_path") or record["file_name"]).name


class DataAccessClient:
    """Share imap-data-access configuration, authentication and HTTP error handling."""

    def query(self, **params: str | int) -> list[APIRecord]:
        """Use the public science/ancillary query API."""
        return ida.query(**params)

    def _get(self, route: str, **params: str | float) -> list[Any]:
        # imap-data-access has no public wrapper for these GET routes yet.
        request = requests.Request(
            "GET", f"{ida.io._get_base_url()}/{route}", params=params
        ).prepare()
        with ida.io._make_request(request) as response:
            result = response.json()
        if not isinstance(result, list):
            raise ValueError(f"Expected a list from /{route}")
        return result

    def records(self, route: str, **params: str | float) -> list[APIRecord]:
        """Read a table route that returns JSON records."""
        result = self._get(route, **params)
        if not all(isinstance(record, dict) for record in result):
            raise ValueError(f"Expected metadata records from /{route}")
        return result

    def files(self, route: str, **params: str | float) -> list[str]:
        """Read a route that returns a list of filenames."""
        result = self._get(route, **params)
        if not all(isinstance(name, str) for name in result):
            raise ValueError(f"Expected filenames from /{route}")
        return result


@dataclass(frozen=True)
class SpinRecord:
    """Plain spin metadata for the shared priority-selection helper."""

    file_path: str
    start_date: dt.datetime
    end_date: dt.datetime
    version: str


class APIDependencySource:
    """Use latest indexed files rather than Dagster's materialization history.

    Supply the target repoint for intraday repoint jobs. Science dates are day
    granularity, so date-only queries cannot distinguish repoints within a day.
    """

    def __init__(
        self, client: DataAccessClient | None = None, repointing: int | None = None
    ) -> None:
        """Configure a local snapshot without AWS credentials or a database session."""
        self.client = client or DataAccessClient()
        self.repointing = repointing
        self._repoint_records: dict[tuple[str, str, str], list[APIRecord]] = {}

    def science_records(
        self, dependency: DependencyNode, **params: str | int
    ) -> list[APIRecord]:
        """Read science files for a product with optional query constraints."""
        return self.client.query(
            instrument=dependency.source,
            data_level=dependency.data_type,
            descriptor=dependency.descriptor,
            **params,
        )

    @staticmethod
    def _science_groups(records: list[APIRecord]) -> list[list[str]]:
        latest: dict[tuple[str, int | None, str], tuple[Version, str]] = {}
        for record in records:
            name = record_filename(record)
            file = ida.ScienceFilePath(name)
            key = (file.start_date, file.repointing, file.extension)
            version = Version(file.major_version, file.minor_version)
            if key not in latest or version > latest[key][0]:
                latest[key] = (version, name)
        return [[name] for _, name in sorted(latest.values(), key=lambda item: item[1])]

    def science_files(
        self,
        dependency: DependencyNode,
        target_start: dt.datetime,
        target_end: dt.datetime,
    ) -> list[list[str] | None]:
        """Select the latest files on days overlapping the requested [start, end)."""
        records = self.science_records(
            dependency,
            start_date=target_start.strftime("%Y%m%d"),
            end_date=target_end.strftime("%Y%m%d"),
            version="latest",
        )
        return self._science_groups(
            [
                r
                for r in records
                if target_start.date() <= parse_time(r["start_date"]).date()
                and parse_time(r["start_date"]) < target_end
                and (
                    self.repointing is None
                    or r.get("repointing") in (None, self.repointing)
                )
            ]
        )

    def _available_records(self, dependency: DependencyNode) -> list[APIRecord]:
        key = (dependency.source, dependency.data_type, dependency.descriptor)
        if key not in self._repoint_records:
            self._repoint_records[key] = self.science_records(
                dependency, version="latest"
            )
        return self._repoint_records[key]

    def science_files_by_repoint(
        self, dependency: DependencyNode, repoints: list[int]
    ) -> list[list[str] | None]:
        """Select latest files for the requested repoints, including Hi neighbors."""
        return self._science_groups(
            [
                r
                for r in self._available_records(dependency)
                if r.get("repointing") in repoints
            ]
        )

    def ancillary_records(
        self,
        dependency: DependencyNode,
        target_start: dt.datetime,
        target_end: dt.datetime,
    ) -> list[AncillaryRecord]:
        """Return overlapping records at the highest version per start date."""
        # No lower start bound: an older open-ended calibration may still apply.
        records = self.client.query(
            table="ancillary",
            instrument=dependency.source,
            descriptor=dependency.descriptor,
            end_date=target_end.strftime("%Y%m%d"),
        )
        latest: dict[dt.datetime, APIRecord] = {}
        for record in records:
            start = parse_time(record["start_date"])
            end = parse_time(record["end_date"]) if record.get("end_date") else None
            if start >= target_end or (end is not None and end < target_start):
                continue
            if start not in latest or Version.from_version(
                record["version"]
            ) > Version.from_version(latest[start]["version"]):
                latest[start] = record
        return [
            AncillaryRecord(record_filename(r), start) for start, r in latest.items()
        ]

    def spin_files(
        self, target_start: dt.datetime, target_end: dt.datetime
    ) -> list[str] | None:
        """Use shared spin supersession rules on overlapping API records."""
        # Table date filters select contained ranges, not overlap; filter locally.
        records = sorted(
            self.client.records("spin-table"),
            key=lambda r: parse_time(r["ingestion_date"]),
        )
        candidates = [
            SpinRecord(
                record_filename(r),
                parse_time(r["start_date"]),
                parse_time(r["end_date"]),
                str(r["version"]),
            )
            for r in records
            if parse_time(r["start_date"]) <= target_end
            and parse_time(r["end_date"]) >= target_start
        ]
        selected = select_spin_records(candidates, target_start, target_end)
        return [r.file_path for r in selected] or None

    def repoint_files(
        self, target_start: dt.datetime, target_end: dt.datetime
    ) -> list[str] | None:
        """Return the newest repoint filename when it covers the requested end."""
        records = self.client.records("repoint-table")
        if not records:
            raise ValueError("No Repoint file found through the API.")
        record = max(records, key=record_filename)
        if parse_time(record["end_date"]) < target_end:
            return None
        return [record_filename(record)]

    def spice_files(
        self, spice_types: list[str], target_start: dt.datetime, target_end: dt.datetime
    ) -> tuple[list[str], list[str]] | tuple[None, None]:
        """Use production metakernel selection and shared kernel-presence checks."""
        if target_start == target_end:
            target_end += dt.timedelta(days=1)
        files = self.client.files(
            "metakernel",
            start_time=_seconds_since_j2000(target_start),
            end_time=_seconds_since_j2000(target_end),
            list_files="true",
            file_types=",".join(spice_types),
        )
        return files, check_for_missing_kernels(",".join(spice_types), files)

    def available_repoints(self, dependency: DependencyNode) -> list[int]:
        """Return repoints with currently indexed science files."""
        return sorted(
            {
                int(r["repointing"])
                for r in self._available_records(dependency)
                if r.get("repointing") is not None
            }
        )

    def inprogress_repoints(self, dependency: DependencyNode) -> list[int]:
        """Reject production readiness queries; local handlers must disable waiting."""
        raise NotImplementedError(
            "Use wait_for_dependencies=False for local API sources"
        )

    def pointing_exists(self, repoint: int) -> bool:
        """Reject pointing readiness queries instead of fabricating metadata."""
        raise NotImplementedError(
            "Use wait_for_dependencies=False for local API sources"
        )
