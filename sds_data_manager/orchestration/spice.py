"""Contains all functions needed to calculate SPICE files dependencies."""

import datetime
import json
import logging

from sds_data_manager.lambda_code.SDSCode.api_lambdas import spice_metakernel_api
from sds_data_manager.lambda_code.SDSCode.database import models
from sds_data_manager.orchestration.dependency_file_selection import (
    _seconds_since_j2000,
    check_for_missing_kernels,
)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# These kernels are delivered as append-only time series: newer files either
# extend an existing coverage window or replace it with a higher version of the
# same window, so downstream triggering should be narrowed to genuinely new
# coverage where possible.
GROWING_KERNEL_TYPES = (
    "attitude_history",
    "pointing_attitude",
    "ephemeris_reconstructed",
)
# Kernel types that we don't want to trigger processing.
NON_TRIGGERING_KERNEL_TYPES = ("ephemeris_predicted",)


def get_upstream_dependency_inputs_spice(
    dependencies: list,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
):
    """Construct a ProcessingInputCollection of dependency files.

    For each dependency, query for existing files in s3 and add any matching files
    found to a ProcessingInputCollection.

    Parameters
    ----------
    dependencies : list
        List of dependency dictionaries either downstream or upstream from the
        dependency in the query parameters.
    start_date : datetime
        Start date to find dependent files with.
    end_date : datetime
        End date to find dependent files with.

    Returns
    -------
    ProcessingInputCollection
        Dependency files that can include Ancillary, SPICE, or Science inputs.
    list
        Missing kernels types that were requested but not found in the metakernel files.
    """
    # TODO revisit setting end_time after SIT-4. Should be handled upstream
    query_end_date = (
        end_date + datetime.timedelta(hours=24) if end_date == start_date else end_date
    )
    start_time = _seconds_since_j2000(start_date)
    end_time = _seconds_since_j2000(query_end_date)
    metakernel_response = spice_metakernel_api.lambda_handler(
        {
            "queryStringParameters": {
                "start_time": start_time,
                "end_time": end_time,
                "list_files": "True",
                "file_types": ",".join(dependencies),
                # TODO: revisit this after SIT-4
                # "require_coverage": "True",
            }
        },
        None,
    )
    if metakernel_response["statusCode"] != 200:
        logger.error(f"Metakernel lambda raised error: {metakernel_response['body']}")
        return None, None
    metakernel_files = json.loads(metakernel_response["body"])
    # Get list of any missing kernels
    missing_kernels = check_for_missing_kernels(
        ",".join(dependencies), metakernel_files
    )

    logger.info(f"Found metakernel files: {metakernel_files}. Adding to collection.")
    return metakernel_files, missing_kernels


def parse_interval_list(
    raw_intervals: list[list[str]] | None,
) -> list[list[datetime.datetime]]:
    """Convert a list of ISO-formatted [start, end] pairs to datetime pairs.

    Parameters
    ----------
    raw_intervals : list[list[str]] | None
        The `file_intervals_datetime` value from a SPICEFiles row: a list of
        [start, end] pairs of ISO-formatted datetime strings.

    Returns
    -------
    list[list[datetime.datetime]]
        The same intervals, parsed to datetime objects.
    """
    if not raw_intervals:
        return []
    return [
        [
            start
            if isinstance(start, datetime.datetime)
            else datetime.datetime.fromisoformat(start),
            end
            if isinstance(end, datetime.datetime)
            else datetime.datetime.fromisoformat(end),
        ]
        for start, end in raw_intervals
    ]


def subtract_intervals(
    new_intervals: list[list[datetime.datetime]],
    old_intervals: list[list[datetime.datetime]],
) -> list[list[datetime.datetime]]:
    """Return the portions of new_intervals not covered by old_intervals.

    Both inputs are lists of [start, end] datetime pairs. This is used to find
    the segments of a new attitude history kernel's SPICE-derived coverage
    that were not already covered by its predecessor kernel, so that only
    genuinely new time coverage gets reprocessed.

    Parameters
    ----------
    new_intervals : list[list[datetime.datetime]]
        The [start, end] segments of the new kernel's coverage.
    old_intervals : list[list[datetime.datetime]]
        The [start, end] segments already covered by the predecessor kernel.

    Returns
    -------
    list[list[datetime.datetime]]
        The leftover [start, end] segments of new_intervals not covered by
        old_intervals.

    Notes
    -----
    Neither `new_intervals` nor `old_intervals` needs to be pre-sorted, and
    `old_intervals` is allowed to contain overlapping entries: each new
    interval is handled independently (sorting the relevant slice of
    `old_intervals` itself, below), so ordering of the inputs doesn't affect
    correctness. The one thing that does depend on input order is the
    ordering of the *output*: leftover segments are appended in the order
    `new_intervals` was given, so pass a chronologically sorted
    `new_intervals` if you want a chronologically sorted result.
    """
    leftover = []
    for new_start, new_end in new_intervals:
        # Old intervals that don't overlap this new interval at all can't
        # subtract anything from it, so drop them. Sort what's left by start
        # time so the sweep below can walk left to right in one pass.
        overlaps = sorted(
            (o for o in old_intervals if o[0] < new_end and o[1] > new_start),
            key=lambda o: o[0],
        )
        # `cursor` tracks how far into [new_start, new_end) we've accounted
        # for so far - everything before it has already been covered by an
        # old interval already processed, or emitted as leftover, else emit
        # the gap before it as leftover.
        cursor = new_start
        for old_start, old_end in overlaps:
            if old_start > cursor:
                # Gap between what we've covered so far and this old
                # interval's start: that gap is new (uncovered) coverage.
                leftover.append([cursor, min(old_start, new_end)])
            # Advance the cursor past this old interval. max() (rather than
            # plain assignment) handles old intervals that overlap each
            # other, where old_end could be behind the cursor already.
            cursor = max(cursor, old_end)
            if cursor >= new_end:
                break
        if cursor < new_end:
            # Anything left after the last old interval is a trailing
            # leftover, e.g. brand new segments appended past old_end.
            leftover.append([cursor, new_end])
    return leftover


def get_growing_kernel_trigger_ranges(
    session, kernel_type: str, new_files: list["models.SPICEFiles"]
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    """Determine the time ranges that require reprocessing for new AH/DPS kernels.

    Newly ingested attitude_history or pointing_attitude kernels are handled
    with one of three rules:

    1. Same coverage as an existing kernel, but a higher version number (a
       correction to already-delivered data) -> the entire coverage range is
       reprocessed, since previously processed data can no longer be assumed
       to be valid.
    2. Coverage that extends an existing kernel with the same start date
       (the normal append-only growth pattern, since new segments are only
       ever appended) -> only the segments of coverage that are new are
       reprocessed. Per-segment coverage was already computed with SPICE
       tools (spiceypy ckcov) at ingestion time and is stored on each file's
       `file_intervals_datetime`.
    3. A new start date, meaning no earlier kernel shares it (the kernel has
       "rolled over" to a new coverage window) -> the entire coverage range
       is reprocessed, since none of it has been processed before.

    Parameters
    ----------
    session : orm session
        Database session used to look up existing kernels of this type.
    kernel_type : str
        One of `GROWING_KERNEL_TYPES` ("attitude_history" or
        "pointing_attitude").
    new_files : list[models.SPICEFiles]
        Newly ingested SPICEFiles rows of the given kernel_type.

    Returns
    -------
    list[tuple[datetime.datetime, datetime.datetime]]
        The time ranges that require reprocessing.
    """
    ranges = []
    for new_file in sorted(new_files, key=lambda f: f.max_date_datetime):
        predecessor = (
            session.query(models.SPICEFiles)
            .filter(
                models.SPICEFiles.kernel_type == kernel_type,
                models.SPICEFiles.file_name != new_file.file_name,
                models.SPICEFiles.min_date_datetime == new_file.min_date_datetime,
                models.SPICEFiles.max_date_datetime <= new_file.max_date_datetime,
            )
            .order_by(models.SPICEFiles.max_date_datetime.desc())
            .first()
        )

        if predecessor is None:
            # Case 3: new start date - nothing to diff against.
            ranges.append((new_file.min_date_datetime, new_file.max_date_datetime))
        elif predecessor.max_date_datetime == new_file.max_date_datetime:
            # Case 1: version increment with identical coverage.
            ranges.append((new_file.min_date_datetime, new_file.max_date_datetime))
        else:
            # Case 2: coverage extended - only trigger the new segments.
            new_intervals = parse_interval_list(new_file.file_intervals_datetime)
            if not new_intervals:
                # Defensive fallback: no segment data to diff against.
                ranges.append((new_file.min_date_datetime, new_file.max_date_datetime))
                continue
            old_intervals = parse_interval_list(predecessor.file_intervals_datetime)
            new_segments = subtract_intervals(new_intervals, old_intervals)
            ranges.extend((seg[0], seg[1]) for seg in new_segments)
    return ranges
