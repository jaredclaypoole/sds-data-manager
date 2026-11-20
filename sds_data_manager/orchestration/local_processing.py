"""Prepare processing-compatible dependency JSON using shared selection handlers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import imap_data_access as ida
from imap_data_access.file_validation import Version

from sds_data_manager.orchestration.api_dependency_source import (
    APIDependencySource,
    parse_time,
)
from sds_data_manager.orchestration.dependency import DependencyConfigReader
from sds_data_manager.orchestration.dependency_handler_registry import (
    DependencyHandlerRegistry,
)
from sds_data_manager.orchestration.types import ProcessingJobNode


def estimate_output_versions(
    job: ProcessingJobNode,
    source: APIDependencySource,
    start: dt.datetime,
    repointing: int | None = None,
) -> dict[str, dict[str, int]]:
    """Estimate a shared next minor version from indexed output files, ignoring jobs."""
    minor = 0
    for output in job.outputs:
        params = dict(
            start_date=start.strftime("%Y%m%d"), end_date=start.strftime("%Y%m%d")
        )
        if output.data_type == "ancillary":
            records = source.client.query(
                table="ancillary",
                instrument=output.source,
                descriptor=output.descriptor,
                **params,
            )
        elif output.data_type in ida.VALID_DATALEVELS:
            records = source.science_records(output, **params)
        else:
            # A local label is sufficient for outputs indexed as SPICE.
            records = []
        for record in records:
            if record.get("repointing") == repointing:
                minor = max(minor, Version.from_version(record["version"]).minor)
    return {
        output.descriptor: {
            "major_version": output.major_version or 1,
            "minor_version": minor + 1,
        }
        for output in job.outputs
    }


def prepare_local_job(
    job: ProcessingJobNode,
    start: dt.datetime,
    end: dt.datetime,
    output_dir: Path,
    repointing: int | None = None,
    source: APIDependencySource | None = None,
) -> Path:
    """Resolve fresh inputs and write JSON without uploading or submitting a job."""
    start = parse_time(start.isoformat())
    end = parse_time(end.isoformat())
    if start >= end:
        raise ValueError("Start must be before the exclusive end")
    if not job.outputs:
        raise ValueError("The selected job has no configured outputs")
    if job.partition == "repoint" and repointing is None:
        raise ValueError(
            "Repoint jobs require --repointing and the corresponding time window"
        )
    if repointing is not None and repointing < 0:
        raise ValueError("Repointing must be nonnegative")
    source = source or APIDependencySource(repointing=repointing)
    if source.repointing != repointing:
        raise ValueError("Source and job repointing must agree")
    handler = DependencyHandlerRegistry.get_handler(
        job, source, logging.getLogger(__name__), wait_for_dependencies=False
    )
    inputs = handler.get_dependencies(start, end, repointing)
    versions = estimate_output_versions(job, source, start, repointing)
    filename = (
        ida.DependencyFilePath.generate_from_inputs(
            instrument=job.source,
            data_level=job.data_type,
            descriptor=job.descriptor,
            start_time=start.strftime("%Y%m%d"),
            repointing=repointing,
            major_version=max(v["major_version"] for v in versions.values()),
            minor_version=max(v["minor_version"] for v in versions.values()),
            extension="json",
        )
        .construct_path()
        .name
    )
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(
            {"dependency": json.loads(inputs.serialize()), "version": versions},
            stream,
            indent=2,
        )
        stream.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """Read job configuration and generate a fresh dependency file from the APIs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--data-level", required=True)
    parser.add_argument(
        "--descriptor", required=True, help="Job descriptor in the dependency YAML"
    )
    parser.add_argument("--start-date", required=True, type=parse_time)
    parser.add_argument(
        "--end-date",
        required=True,
        type=parse_time,
        help="Exclusive UTC end; YYYYMMDD or ISO timestamp",
    )
    parser.add_argument("--repointing", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    key = (args.instrument, args.data_level, args.descriptor)
    config = DependencyConfigReader().config
    if key not in config:
        parser.error(f"No configured processing job: {key}")
    logging.basicConfig(level=logging.INFO)
    try:
        path = prepare_local_job(
            config[key],
            args.start_date,
            args.end_date,
            args.output_dir,
            args.repointing,
        )
    except (ValueError, FileExistsError, ida.io.IMAPDataAccessError) as exc:
        parser.error(str(exc))
    print(path)
