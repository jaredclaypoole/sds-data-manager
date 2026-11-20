"""Shared functions for SPICE-related lambdas."""

import json
import logging
import os
import tempfile
from collections.abc import Collection
from pathlib import Path

import boto3
import imap_data_access
import spiceypy
from imap_data_access import SPICEFilePath

from .api_lambdas import spice_query_api
from .api_lambdas.metakernel import MetaKernel
from .spice_types import (
    SEAM_GAP_TOLERANCE,
    EarthAttitudeKernels,
    IMAPFramesKernels,
    KernelCollection,
    LeapsecondKernels,
    LoPivotAttitudeKernels,
    PlanetaryConstantsKernels,
    PlanetaryEphemerisKernels,
    PointingAttitudeKernels,
    ScienceFramesKernels,
    SpacecraftAttitudeKernels,
    SpacecraftClockKernels,
    SpacecraftEphemerisKernels,
    SpinTables,
    ThrusterFiles,
)

__all__ = [
    "MAXIMUM_MISSION_J2000_TIME",
    "SEAM_GAP_TOLERANCE",
    "SPICE_FILE_PRIORITY",
    "EarthAttitudeKernels",
    "IMAPFramesKernels",
    "KernelCollection",
    "LeapsecondKernels",
    "LoPivotAttitudeKernels",
    "PlanetaryConstantsKernels",
    "PlanetaryEphemerisKernels",
    "PointingAttitudeKernels",
    "ScienceFramesKernels",
    "SpacecraftAttitudeKernels",
    "SpacecraftClockKernels",
    "SpacecraftEphemerisKernels",
    "SpinTables",
    "ThrusterFiles",
    "download_from_s3",
    "furnish_best_spice_file",
    "metakernel_builder",
]

MAXIMUM_MISSION_J2000_TIME = 4575787269.183866

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Fields that rank files of the same type, compared in order. The latest
# ingested file is the newest solution or reprocessing. Versions can't rank
# files, since they restart with each new file name. Files delivered in a batch
# can share an ingestion time (S3 records whole seconds), so the file whose
# coverage starts, then ends, later wins. For ephemeris, the later start is the
# newer orbit determination.
SPICE_FILE_PRIORITY = ("timestamp", "min_date_j2000", "max_date_j2000")


def download_from_s3(s3_key: str, bucket_name: str | None = None) -> Path:
    """Download a file from S3 to a local temporary path.

    Parameters
    ----------
    s3_key : str
        The S3 key (path) of the file to download.
    bucket_name : Optional[str], optional
        The S3 bucket name. If not provided, will use the S3_BUCKET
        environment variable.

    Returns
    -------
    Path
        The local path where the file was downloaded.

    Raises
    ------
    ValueError
        If bucket_name is not provided and S3_BUCKET environment variable is
        not set.
    """
    if bucket_name is None:
        bucket_name = os.environ.get("S3_BUCKET")
        if bucket_name is None:
            raise ValueError(
                "bucket_name must be provided or S3_BUCKET environment "
                "variable must be set"
            )

    # Create a temporary file path
    filename = os.path.basename(s3_key)
    temp_dir = tempfile.gettempdir()
    local_path = Path(temp_dir) / filename

    # Download from S3
    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket_name, s3_key, str(local_path))
        logger.info(f"Downloaded {s3_key} from bucket {bucket_name} to {local_path}")
        return local_path
    except Exception as e:
        logger.error(e)
        raise FileNotFoundError(
            f"Failed to download {s3_key} from bucket {bucket_name}: {e}"
        ) from e


def furnish_best_spice_file(kernel_type: str):
    """Furnish the best kernel for given type.

    Parameters
    ----------
    kernel_type: str
        Kernel type to furnish, e.g. 'leapseconds' or 'spacecraft_clock'.

    Returns
    -------
    highest_version_spice_file: Path
        The path to the SPICE file that was furnished

    Raises
    ------
    FileNotFoundError
        If S3_BUCKET or DATA_DIR are not set, no files are found in the database,
        or the file is not in the S3 bucket, FileNotFoundError will raise.
    """
    # Check if S3_BUCKET and DATA_DIR are set
    if "S3_BUCKET" not in os.environ or "DATA_DIR" not in imap_data_access.config:
        raise FileNotFoundError(
            f"Unable to find the latest {kernel_type} kernel. "
            "Please ensure S3_BUCKET and DATA_DIR are set in the environment variables."
        )

    # Query for latest kernel
    metakernel = metakernel_builder(
        0,
        MAXIMUM_MISSION_J2000_TIME,
        file_types={kernel_type.upper()},
    )
    metakernel_files = metakernel.return_spice_files_in_order(detailed=False)
    if not metakernel_files:
        raise FileNotFoundError(
            f"Unable to find the latest {kernel_type} kernel. "
            "Please ensure that the kernel is available in the database."
        )
    kernel_filename = Path(metakernel_files[0]).name
    logger.info(f"Furnishing the latest {kernel_type} kernel: {kernel_filename}")
    # Download the latest kernel file
    # Convert this into an s3 key
    # Relative to our base directory to trim off the initial path
    s3_key = str(
        SPICEFilePath(kernel_filename)
        .construct_path()
        .relative_to(imap_data_access.config["DATA_DIR"])
    )
    highest_version_spice_file = download_from_s3(s3_key)
    logger.info(f"Downloaded SPICE file: {highest_version_spice_file}")
    # Furnish the SPICE file
    spiceypy.furnsh(str(highest_version_spice_file))
    return highest_version_spice_file


def metakernel_builder(
    start_time: float | None,
    end_time: float | None,
    file_types: Collection[str] | None = None,
) -> MetaKernel:
    """Create a MetaKernel class and inserts files into it.

    Parameters
    ----------
    start_time: float | None
        Start time in seconds since J2000, used to filter the SPICE file database.
        If None, no lower bound is used in the query.
    end_time: float | None
        End time in seconds since J2000, used to filter the SPICE file database.
        If None, no upper bound is used in the query.
    file_types: Collection[str] | None
        Optional restriction on which SPICE file types to include. Spin and
        thruster files are only included when requested here.

    Returns
    -------
    MetaKernel
        Constructed MetaKernel containing all matching SPICE files.
    """
    # Establish start and end times for the metakernel
    metakernel_start_time = 0 if start_time is None else start_time
    metakernel_end_time = MAXIMUM_MISSION_J2000_TIME if end_time is None else end_time
    kernel_collection = KernelCollection()
    requested_types = kernel_collection.requested_types(file_types)
    # Only the requested categories are tracked, so categories that were not
    # requested are not reported as gaps.
    metakernel = MetaKernel(
        metakernel_start_time,
        metakernel_end_time,
        allowed_spice_types=[
            category.spice_category_name() for category in requested_types
        ],
        gap_tolerances=kernel_collection.gap_tolerances,
    )

    for spice_category, spice_subtypes in requested_types.items():
        for spice_subtype in spice_subtypes:
            query_string_parameters = {
                "type": spice_subtype.name.lower(),
                "latest": "True",
            }
            # Add query start and end times if provided, skip if not included
            if start_time is not None:
                query_string_parameters["start_time"] = start_time
            if end_time is not None:
                query_string_parameters["end_time"] = end_time

            spice_files = spice_query_api.lambda_handler(
                {"queryStringParameters": query_string_parameters},
                None,
            )
            metakernel.load_spice(
                json.loads(spice_files["body"]),
                spice_category.spice_category_name(),
                "file_intervals_j2000",
                priority_fields=SPICE_FILE_PRIORITY,
            )

    return metakernel
