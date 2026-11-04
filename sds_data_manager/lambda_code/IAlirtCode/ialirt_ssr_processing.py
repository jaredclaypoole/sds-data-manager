"""I-ALiRT SSR processing lambdas.

Each instrument (and the spacecraft) is deployed as its own Lambda from the
same image. The INSTRUMENT environment variable selects which processor runs.

Realtime (ialirt_ingest.py) processes all instruments over the last 7
minutes. SSR processes one instrument for one whole day. Where a function
differs from realtime, its docstring Notes say how.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib.metadata import version as package_version
from pathlib import Path

import boto3
import imap_data_access
import numpy as np
import pandas as pd
import requests
import spiceypy
import xarray as xr
from imap_data_access.processing_input import (
    ProcessingInputCollection,
    SPICEInput,
    SPICESource,
)
from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.ialirt.l0.parse_mag import process_packet
from imap_processing.ialirt.l0.process_codice import process_codice
from imap_processing.ialirt.l0.process_hit import process_hit
from imap_processing.ialirt.l0.process_status import process_status
from imap_processing.ialirt.l0.process_swapi import process_swapi_ialirt
from imap_processing.ialirt.l0.process_swe import process_swe
from imap_processing.spice.geometry import SpiceBody, SpiceFrame, imap_state
from imap_processing.spice.time import (
    et_to_ttj2000ns,
    et_to_utc,
    str_to_et,
    str_yyyymmdd_to_ttj2000ns,
    ttj2000ns_to_et,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KERNELS = {
    "ephemeris_predicted",
    "ephemeris_90days",
    "planetary_ephemeris",
    "spacecraft_clock",
    "leapseconds",
    "imap_frames",
    "science_frames",
    "planetary_constants",
}


# Only MAG and the spacecraft use SPICE geometry. The other instruments only
# convert times, which needs just these kernels.
GEOMETRY_INSTRUMENTS = {"mag", "spacecraft"}
TIME_KERNELS = {"leapseconds", "spacecraft_clock"}
# Spacecraft position/velocity cadence over the day.
SPACECRAFT_CADENCE = timedelta(minutes=1)
# Packets, SPICE kernels and ancillary files for one invocation.
DOWNLOAD_DIR = Path("/tmp/ssr")  # noqa: S108
# The two values below do different jobs:
# - COVERAGE_TOLERANCE is the pass/fail completeness check in
#   check_day_coverage. The first packet must be in the first 60 s of the day,
#   and there must be a packet in the first 60 s of the next day to show the
#   data continues through the end of the day.
# - BOUNDARY_BUFFER is used by trim_packets to decide how much data is passed
#   to the instrument processing. It is larger because some products are built
#   from packets spread over 4 minutes, so a product near midnight needs
#   packets from the neighboring day.
COVERAGE_TOLERANCE = timedelta(seconds=60)
BOUNDARY_BUFFER = timedelta(minutes=10)
# src_seq_ctr is a 14-bit counter.
SEQ_CTR_MODULO = 16384
# How often to store spacecraft status. Matches ialirt-ingest's ~1 per minute.
STATUS_CADENCE_NS = 60 * 10**9
SOFTWARE_VERSIONS = {
    "imap_processing": package_version("imap-processing"),
    "imap_data_access": package_version("imap-data-access"),
}


def get_ancillary(
    instrument: str, descriptor: str, day: datetime, download_dir: Path
) -> Path:
    """Query and download the ancillary file valid for a given day.

    This is consistent with how ancillary files are selected for processing
    (ImapJob.get_ancillary_files_inputs in orchestration/imap_job.py):

    - A file is valid for the day if its start_date is on or before the day and
      its end_date is on or after the day. Files without an end_date
      (e.g. imap_swe_l1b-in-flight-cal_20260406_v001.csv) are valid from their
      start_date onward. Files with an end_date
      (e.g. imap_swe_l1b-in-flight-cal_20240510_20260716_v020.csv) are only
      valid within that range.
    - Only the latest version for each start_date is considered.
    - Of the valid files, the one with the latest start_date is selected.

    Parameters
    ----------
    instrument : str
        The name of the instrument.
    descriptor : str
        The name of the descriptor.
    day : datetime
        The day being processed.
    download_dir : Path
        The directory where the file will be downloaded.

    Returns
    -------
    download_path : Path
        Download path of the ancillary file.

    Notes
    -----
    Realtime uses the newest file. SSR uses the file that was valid on
    the day being processed.
    """
    imap_data_access.config["DATA_DIR"] = download_dir
    date_str = day.strftime("%Y%m%d")
    # end_date filters on the file start_date, so this returns files that
    # start on or before the day.
    ancillary_files = imap_data_access.query(
        table="ancillary",
        instrument=instrument,
        descriptor=descriptor,
        end_date=date_str,
        version="latest",
    )

    # Drop files whose end_date is before the day.
    valid_files = [
        f for f in ancillary_files if not f.get("end_date") or f["end_date"] >= date_str
    ]
    if not valid_files:
        raise FileNotFoundError(
            f"No ancillary file found for {instrument=}, {descriptor=}, {date_str=}"
        )

    ancillary_file = sorted(
        valid_files, key=lambda x: (x["start_date"], x["version"]), reverse=True
    )[0]
    download_path = imap_data_access.download(ancillary_file["file_path"])
    logger.info(f"Downloaded ancillary file: {download_path}")

    return download_path


def get_ancillary_ssr(*args, **kwargs):
    """Define a convenience alias for get_ancillary.

    Needed to `process_algorithms` can locally define `get_ancillary`
    while still calling the original implementation.
    """
    return get_ancillary(*args, **kwargs)


def get_latest_spice_kernels(
    url: str, day: datetime, kernel_types: set[str] = KERNELS
) -> ProcessingInputCollection:
    """Query the SPICE metakernel API for the latest kernels for a day.

    The time range covers the day before, the day and the day after, to match
    the packet files.

    Parameters
    ----------
    url : str
        The IMAP data access API URL.
    day : datetime
        The day to query.
    kernel_types : set[str]
        The kernel types to query. Defaults to all of KERNELS.

    Returns
    -------
    dependency_inputs : ProcessingInputCollection
        A collection containing a SPICEInput object with the list of kernel
        filenames returned from the metakernel API, in priority order.

    Notes
    -----
    Realtime asks for kernels covering the last week. SSR asks for
    kernels covering the day being processed, plus one day on each side.
    """
    # The end date is exclusive (midnight), so day - 1 to day + 2 covers
    # day - 1, day and day + 1.
    file_types = ",".join(kernel_types)
    metakernel_url = url + "/metakernel"

    params = {
        "start_time": (day - timedelta(days=1)).strftime("%Y%m%d"),
        "end_time": (day + timedelta(days=2)).strftime("%Y%m%d"),
        "list_files": "True",
        "file_types": file_types,
    }

    logger.info(f"Sending request to {metakernel_url} with params: {params}")
    response = requests.get(metakernel_url, params=params, timeout=10)
    response.raise_for_status()
    metakernel_files = response.json()

    logger.info(f"Found metakernel files: {metakernel_files}. Adding to collection.")
    dependency_inputs = ProcessingInputCollection()
    dependency_inputs.add(SPICEInput(*metakernel_files))

    return dependency_inputs


def download_spice_file(
    dependencies: ProcessingInputCollection, download_dir: Path
) -> list[Path]:
    """Download SPICE kernel files from the IMAP data archive.

    Parameters
    ----------
    dependencies : ProcessingInputCollection
        A collection containing a SPICEInput object with the list of kernel
        filenames returned from the metakernel API.
    download_dir : Path
        The directory where the files will be downloaded.

    Returns
    -------
    spice_files : list[Path]
        The downloaded SPICE files.

    Notes
    -----
    List is priority ordered so furnishing in order results in correct SPICE priority.

    Realtime saves kernels to EFS and loads them here. SSR saves them
    to /tmp and loads them later in lambda_handler.
    """
    imap_data_access.config["DATA_DIR"] = download_dir
    dependencies.download_all_files()

    spice_files = dependencies.get_file_paths(data_type=SPICESource.SPICE.value)
    logger.info(f"Downloaded SPICE files: {spice_files}")

    return spice_files


def query_filenames(day: datetime) -> list[str]:
    """Query the latest I-ALiRT L0 files for the day before, the day and the day after.

    Packets for a given day can land in the next day's file, and products
    near midnight are built from packets on both sides of it, so all three
    days are needed to get a complete day of data.

    Parameters
    ----------
    day : datetime
        The day to query.

    Returns
    -------
    filenames : list[str]
        File paths relative to the data archive root, one per day.

    Raises
    ------
    FileNotFoundError
        If a file is missing for any of the days.

    Notes
    -----
    Realtime reads the last 7 minutes of packet files from S3. SSR
    reads the daily L0 files from the archive.
    """
    dates = [(day + timedelta(days=i)).strftime("%Y%m%d") for i in (-1, 0, 1)]
    results = imap_data_access.query(
        table="science",
        instrument="ialirt",
        data_level="l0",
        descriptor="raw",
        start_date=dates[0],
        end_date=dates[-1],
        version="latest",
    )

    found_dates = {result["start_date"] for result in results}
    missing_dates = [date for date in dates if date not in found_dates]
    if missing_dates:
        raise FileNotFoundError(f"No I-ALiRT L0 file found for {missing_dates}")

    filenames = [result["file_path"] for result in results]
    logger.info(f"Found I-ALiRT L0 files for {dates}: {filenames}")

    return filenames


def parse_packets(
    filenames: list[str], download_dir: Path, apid: int = 478
) -> xr.Dataset:
    """Download I-ALiRT L0 files from the archive and combine into a dataset.

    Parameters
    ----------
    filenames : list[str]
        File paths relative to the data archive root.
    download_dir : Path
        The directory where the files will be downloaded.
    apid : int
        The apid of the packet to be processed.

    Returns
    -------
    combined : xr.Dataset
        Combined dataset.

    Notes
    -----
    Realtime drops duplicates by epoch. SSR drops them by epoch and
    src_seq_ctr, because the daily files overlap.
    """
    imap_data_access.config["DATA_DIR"] = download_dir
    xtce_ialirt_path = (
        imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
    )
    datasets = []

    for filename in filenames:
        local_path = imap_data_access.download(filename)
        xarray_data = packet_file_to_datasets(
            local_path, xtce_ialirt_path, use_derived_value=False
        )[apid]
        datasets.append(xarray_data)

    combined = xr.concat(datasets, dim="epoch").sortby("epoch")

    # The same packet may be in both files. Drop packets where epoch and
    # src_seq_ctr are the same, as is done within packet_file_to_datasets.
    unique_idx = (
        combined[["src_seq_ctr"]]
        .to_dataframe()
        .reset_index()
        .drop_duplicates()
        .index.values
    )
    combined = combined.isel(epoch=unique_idx)

    return combined


class IalirtDataGapError(Exception):
    """Raised when a day of I-ALiRT packets has gaps and is not processed."""


def ttj2000ns_to_utc(ttj2000ns: int) -> str:
    """Convert TTJ2000 nanoseconds (packet epoch) to a UTC string."""
    return et_to_utc(ttj2000ns_to_et(ttj2000ns))


def check_day_coverage(combined: xr.Dataset, day: datetime) -> None:
    """Check that the packets cover the whole day with no missing packets.

    Days that fail are skipped and need to be inspected and processed
    manually.

    Parameters
    ----------
    combined : xr.Dataset
        L0 parsed data for the day before, the day and the day after,
        sorted by epoch.
    day : datetime
        The day being processed.

    Raises
    ------
    IalirtDataGapError
        If the first packet is not near the start of the day, the last packet
        is not at or after the end of the day, or src_seq_ctr skips a value.
    """
    tolerance = int(COVERAGE_TOLERANCE.total_seconds() * 1e9)
    day_start = str_yyyymmdd_to_ttj2000ns(f"{day:%Y%m%d}")
    day_end = str_yyyymmdd_to_ttj2000ns(f"{day + timedelta(days=1):%Y%m%d}")

    # Include packets just after midnight to show the data continues
    # through the end of the day.
    packets = combined.sel(epoch=slice(day_start, day_end + tolerance))
    epoch = packets["epoch"].values
    src_seq_ctr = packets["src_seq_ctr"].values.astype(np.int64)

    if epoch.size == 0:
        raise IalirtDataGapError(f"{day:%Y%m%d}: no packets.")
    if epoch[0] > day_start + tolerance:
        raise IalirtDataGapError(
            f"{day:%Y%m%d}: first packet at {ttj2000ns_to_utc(epoch[0])}."
        )
    if epoch[-1] < day_end:
        raise IalirtDataGapError(
            f"{day:%Y%m%d}: last packet at {ttj2000ns_to_utc(epoch[-1])}."
        )

    gaps = np.flatnonzero(np.diff(src_seq_ctr) % SEQ_CTR_MODULO != 1)
    if gaps.size:
        raise IalirtDataGapError(
            f"{day:%Y%m%d}: {gaps.size} src_seq_ctr gap(s), "
            f"first after {ttj2000ns_to_utc(epoch[gaps[0]])}."
        )


def trim_packets(combined: xr.Dataset, day: datetime) -> xr.Dataset:
    """Keep only the packets needed to process the day.

    combined has the whole day before, the day and the day after. Only the
    day plus BOUNDARY_BUFFER on each side is needed, so products near
    midnight still have all of their packets.

    Parameters
    ----------
    combined : xr.Dataset
        L0 parsed data for the day before, the day and the day after,
        sorted by epoch.
    day : datetime
        The day being processed.

    Returns
    -------
    xr.Dataset
        Packets from BOUNDARY_BUFFER before the day to BOUNDARY_BUFFER after.
    """
    buffer = int(BOUNDARY_BUFFER.total_seconds() * 1e9)
    day_start = str_yyyymmdd_to_ttj2000ns(f"{day:%Y%m%d}")
    day_end = str_yyyymmdd_to_ttj2000ns(f"{day + timedelta(days=1):%Y%m%d}")
    return combined.sel(epoch=slice(day_start - buffer, day_end + buffer))


def process_spacecraft_status(combined: xr.Dataset) -> list[dict]:
    """Process spacecraft status, one product per minute.

    process_status gives every record the time of the middle of its input,
    so it is called on only the last packet of each minute. Each product
    then has that packet's status values and time.

    Parameters
    ----------
    combined : xr.Dataset
        L0 parsed data.

    Returns
    -------
    result : list[dict]
        Spacecraft status data products, one per minute.

    Notes
    -----
    Realtime makes one status record per run, about once a minute.
    SSR makes one record for each minute of the day.
    """
    minute = combined["epoch"] // STATUS_CADENCE_NS
    return [
        process_status(packets.isel(epoch=[-1]))[0]
        for _, packets in combined.groupby(minute)
    ]


def trim_to_day(result: list[dict], day: datetime) -> list[dict]:
    """Keep only products with met_in_utc within the day.

    Trimming happens after processing so that products built from packets
    near midnight still have all of their packets.

    Parameters
    ----------
    result : list[dict]
        Data products with a met_in_utc key.
    day : datetime
        The day being processed.

    Returns
    -------
    list[dict]
        Data products within [day, day + 1).
    """
    start = day.strftime("%Y-%m-%dT%H:%M:%S")
    end = (day + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    return [item for item in result if start <= item["met_in_utc"] < end]


def process_algorithms(
    combined: xr.Dataset,
    items: list[dict],
    instrument: str,
    day: datetime,
    version: str,
    kernel_set_key: str,
) -> dict | None:
    """Process one instrument and append its products to the pending SSR items.

    Return ancillary filenames, or None for empty or partially empty products
    so the handler can return before writing any items. Items are written only
    after the handler exits the SPICE kernel pool successfully.

    Only the requested instrument is selected from the processors list.
    Ancillary files are chosen for the day being processed.
    """
    processors = [
        ("mag", process_packet),
        ("hit", process_hit),
        ("swe", process_swe),
        ("codice_lo", process_codice),
        ("codice_hi", process_codice),
        ("swapi", process_swapi_ialirt),
        ("spacecraft_status", process_spacecraft_status),
    ]
    process_func = dict(processors).get(instrument)

    if process_func is None:
        raise ValueError(f"Unknown instrument: {instrument}")

    def get_ancillary(the_instrument, name):
        return get_ancillary_ssr(the_instrument, name, day, DOWNLOAD_DIR)

    # These `if True` blocks reduce the diff with corresponding realtime processing code
    if True:
        if True:
            if instrument == "swe":
                download_path = get_ancillary(instrument, "l1b-in-flight-cal")
                result = process_func(combined, [download_path])
                ancillary_files = {"swe_l1b-in-flight-cal": download_path.name}
            elif instrument == "mag":
                ialirt_cal_path = get_ancillary(instrument, "ialirt-calibration")
                l1b_cal_path = get_ancillary(instrument, "l1b-calibration")
                l1b_calibration_data = load_cdf(l1b_cal_path)
                ialirt_calibration_data = load_cdf(ialirt_cal_path)
                result = process_func(
                    combined,
                    l1b_calibration_data,
                    ialirt_calibration_data,
                )
                ancillary_files = {
                    "mag_ialirt-calibration": ialirt_cal_path.name,
                    "mag_l1b-calibration": l1b_cal_path.name,
                }
            elif instrument == "codice_lo":
                # CoDICE-Lo uses the same efficiency and geometric factor tables as
                # regular processing.
                l1a_download_path = get_ancillary("codice", "l1a-sci-lut")
                l2_efficiency_download_path = get_ancillary(
                    "codice", "l2-lo-efficiency"
                )
                l2_geometric_download_path = get_ancillary("codice", "l2-lo-gfactor")
                result, _ = process_func(
                    combined,
                    l1a_download_path,
                    l2_efficiency_download_path,
                    "codice_lo",
                    l2_geometric_download_path,
                )
                ancillary_files = {
                    "codice_l1a-sci-lut": l1a_download_path.name,
                    "codice_l2-lo-efficiency": l2_efficiency_download_path.name,
                    "codice_l2-lo-gfactor": l2_geometric_download_path.name,
                }
            elif instrument == "codice_hi":
                # CoDICE-Hi uses its own I-ALiRT efficiency table
                # and no geometric factor.
                l1a_download_path = get_ancillary("codice", "l1a-sci-lut")
                l2_efficiency_download_path = get_ancillary(
                    "codice", "l2-hi-ialirt-efficiency"
                )
                _, result = process_func(
                    combined,
                    l1a_download_path,
                    l2_efficiency_download_path,
                    "codice_hi",
                )
                ancillary_files = {
                    "codice_l1a-sci-lut": l1a_download_path.name,
                    "codice_l2-hi-ialirt-efficiency": l2_efficiency_download_path.name,
                }
            elif instrument == "swapi":
                download_path = get_ancillary(instrument, "esa-unit-conversion")
                calibration_data = pd.read_csv(download_path)
                result = process_func(combined, calibration_data)
                ancillary_files = {"swapi_esa-unit-conversion": download_path.name}
            elif instrument == "hit":
                # HIT has no ancillary files.
                ancillary_files = {}
                result = process_func(combined)
            else:
                # Spacecraft status has no ancillary files.
                ancillary_files = {}
                result = process_func(combined)
            if not (any(result) and all(result)):
                logger.info("No %s products for %s %s", instrument, day, version)
                return
        result = trim_to_day(result, day)
        logger.info("[%s] results populated for [%s]", len(result), instrument)
        insert_formatted_data(result, items, instrument, version, kernel_set_key)
        return ancillary_files


def reformat_data(
    data: list[dict], instrument: str, version: str
) -> tuple[list[dict], list[dict]]:
    """Reformat science and housekeeping data.

    Parameters
    ----------
    data : list[dict]
        Data product.
    instrument : str
        The prefix for the product name.
    version : str
        The data version, e.g. "v001".

    Returns
    -------
    science_data : list[dict]
        Reformatted science data.
    hk_data : list[dict]
        Reformatted housekeeping data.

    Notes
    -----
    SSR adds the instrument_version key and puts it and time_utc first.
    Housekeeping is only made for MAG, chosen by instrument rather than
    by each item's instrument field.
    """
    exclude_keys = {"apid", "met", "met_in_utc", "mag_hk_status"}
    science_data = [
        {
            "instrument_version": f"{instrument}#{version}",
            "time_utc": item["met_in_utc"],
            **{k: v for k, v in item.items() if k not in exclude_keys},
        }
        for item in data
    ]
    hk_data = (
        [
            {
                "instrument_version": f"mag_hk#{version}",
                "time_utc": item["met_in_utc"],
                "instrument": "mag_hk",
                "mag_hk_status": item["mag_hk_status"],
            }
            for item in data
        ]
        if instrument == "mag"
        else []
    )

    return science_data, hk_data


def insert_formatted_data(
    data: list[dict],
    items: list[dict],
    instrument: str,
    version: str,
    kernel_set_key: str,
) -> None:
    """Append science and housekeeping records to the pending SSR items.

    Parameters
    ----------
    data : list[dict]
        Data product produced from processing respectively instrument.
    items : list[dict]
        The pending items, written by the handler.
    instrument : str
        The prefix for the product name.
    version : str
        The data version, e.g. "v001".
    kernel_set_key : str
        The kernel set identifier.

    Notes
    -----
    Unlike realtime's immediate inserts, SSR defers database writes until
    the handler exits the SPICE kernel pool successfully. Spacecraft
    position is calculated separately, for the whole day.
    """
    science_data, hk_data = reformat_data(data, instrument, version)

    # Insert science data
    for record in science_data:
        record["kernel_set_key"] = kernel_set_key
        items.append(record)

    # Insert hk data
    if hk_data:
        for record in hk_data:
            record["kernel_set_key"] = kernel_set_key
            items.append(record)


def spacecraft_items(day: datetime, version: str, kernel_set_key: str) -> list[dict]:
    """Calculate spacecraft position and velocity over the day.

    Parameters
    ----------
    day : datetime
        The day being processed.
    version : str
        The data version, e.g. "v001".
    kernel_set_key : str
        time_utc of the spice item used for processing.

    Returns
    -------
    items : list[dict]
        GSE/GSM position and velocity items at SPACECRAFT_CADENCE.

    Notes
    -----
    Realtime stores one position per instrument per run. SSR stores
    one position for every minute of the day.
    """
    n_steps = int(timedelta(days=1) / SPACECRAFT_CADENCE)
    times = [day + i * SPACECRAFT_CADENCE for i in range(n_steps)]
    time_strs = [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times]
    ets = np.array([str_to_et(t) for t in time_strs])

    # Calculate the spacecraft position and velocity in GSE/GSM coordinates.
    gsm_state = imap_state(ets, ref_frame=SpiceFrame.IMAP_GSM, observer=SpiceBody.EARTH)
    gse_state = imap_state(ets, ref_frame=SpiceFrame.IMAP_GSE, observer=SpiceBody.EARTH)

    return [
        {
            "instrument_version": f"spacecraft#{version}",
            "time_utc": time_str,
            "instrument": "spacecraft",
            "ttj2000ns": int(et_to_ttj2000ns(et)),
            "sc_position_GSM": [Decimal(str(val)) for val in gsm[:3]],
            "sc_velocity_GSM": [Decimal(str(val)) for val in gsm[3:]],
            "sc_position_GSE": [Decimal(str(val)) for val in gse[:3]],
            "sc_velocity_GSE": [Decimal(str(val)) for val in gse[3:]],
            "kernel_set_key": kernel_set_key,
        }
        for time_str, et, gsm, gse in zip(
            time_strs, ets, gsm_state, gse_state, strict=True
        )
    ]


def spice_item(
    dependency_inputs: ProcessingInputCollection,
    version: str,
    kernel_set_key: str,
    kernel_set_key_ttj2000ns: int,
) -> dict:
    """Create the item for the SPICE kernels used.

    Parameters
    ----------
    dependency_inputs : ProcessingInputCollection
        A collection containing a SPICEInput object with the kernels used.
    version : str
        The data version, e.g. "v001".
    kernel_set_key : str
        The kernel set identifier, used as time_utc.
    kernel_set_key_ttj2000ns : int
        The kernel set key in TTJ2000 ns.

    Returns
    -------
    item : dict
        The spice item, keyed by kernel type.

    Notes
    -----
    Same as realtime, but kernel_set_key is passed in instead of
    created here.
    """
    spice_input = dependency_inputs.processing_input[0]
    # All kernels are loaded, and more than one of a type can be returned
    # (e.g. two ephemeris files each covering part of the window), so keep
    # every filename per type, in load order.
    spice_kernels = {}
    for path, filename in zip(
        spice_input.imap_file_paths, spice_input.filename_list, strict=True
    ):
        kernel_type = path.spice_metadata["type"]
        if kernel_type not in spice_kernels:
            spice_kernels[kernel_type] = []
        spice_kernels[kernel_type].append(filename)

    kernel_item = {
        "instrument_version": f"spice#{version}",
        "time_utc": kernel_set_key,
        "instrument": "spice",
        "spice_kernels": spice_kernels,
        "ttj2000ns": kernel_set_key_ttj2000ns,
    }
    return kernel_item


def ancillary_item(
    ancillary_files: dict,
    version: str,
    kernel_set_key: str,
    kernel_set_key_ttj2000ns: int,
) -> dict:
    """Create the item for the ancillary files used.

    Parameters
    ----------
    ancillary_files : dict
        Ancillary filenames used, keyed by instrument_descriptor.
    version : str
        The data version, e.g. "v001".
    kernel_set_key : str
        The kernel set identifier, used as time_utc.
    kernel_set_key_ttj2000ns : int
        The kernel set key in TTJ2000 ns.

    Returns
    -------
    item : dict
        The ancillary item.

    Notes
    -----
    Same fields as realtime, plus the instrument_version key.
    """
    return {
        "instrument_version": f"ancillary#{version}",
        "time_utc": kernel_set_key,
        "instrument": "ancillary",
        "ancillary_files": ancillary_files,
        "software_versions": SOFTWARE_VERSIONS,
        "ttj2000ns": kernel_set_key_ttj2000ns,
    }


def lambda_handler(event, context):
    """Process one day of I-ALiRT SSR data for one instrument.

    Parameters
    ----------
    event : dict
        {"day": "YYYYMMDD", "version": "v001"}
    context : LambdaContext
        This object provides methods and properties that provide
        information about the invocation, function,
        and runtime environment.
    """
    logger.info("Received event: %s", json.dumps(event))

    instrument = os.environ["INSTRUMENT"]
    url = os.environ["IMAP_DATA_ACCESS_URL"]
    dynamodb = boto3.resource("dynamodb")
    data_table = dynamodb.Table(os.environ["SSR_TABLE"])

    day = datetime.strptime(event["day"], "%Y%m%d")
    version = event["version"]

    try:
        # Unreleased L0 files are only returned to queries with an API key.
        ssm_response = boto3.client("ssm").get_parameter(
            Name=os.environ["SSM_API_KEY_PARAMETER"], WithDecryption=True
        )
        imap_data_access.config["API_KEY"] = ssm_response["Parameter"]["Value"]
        # /tmp persists between warm invocations. Clear it first in case a
        # previous invocation failed or timed out before cleaning up.
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        DOWNLOAD_DIR.mkdir(parents=True)

        start = time.perf_counter()
        kernel_types = KERNELS if instrument in GEOMETRY_INSTRUMENTS else TIME_KERNELS
        dependency_inputs = get_latest_spice_kernels(url, day, kernel_types)
        spice_files = download_spice_file(dependency_inputs, DOWNLOAD_DIR)
        size_mb = sum(path.stat().st_size for path in spice_files) / 1e6
        logger.info(
            "Downloaded %d SPICE files (%.1f MB) in %.1f s",
            len(spice_files),
            size_mb,
            time.perf_counter() - start,
        )

        kernel_set_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")

        # Only these kernels are loaded inside the block. They are unloaded when
        # the block exits, even on an error, so none carry over to the next
        # invocation in a warm Lambda.
        with spiceypy.KernelPool([str(path) for path in spice_files]):
            kernel_set_key_ttj2000ns = int(et_to_ttj2000ns(str_to_et(kernel_set_key)))
            items = [
                spice_item(
                    dependency_inputs, version, kernel_set_key, kernel_set_key_ttj2000ns
                )
            ]
            if instrument == "spacecraft":
                # The spacecraft has no ancillary files.
                ancillary_files = {}
                items += spacecraft_items(day, version, kernel_set_key)
            else:
                filenames = query_filenames(day)
                combined = parse_packets(filenames, DOWNLOAD_DIR)
                check_day_coverage(combined, day)
                combined = trim_packets(combined, day)
                ancillary_files = process_algorithms(
                    combined, items, instrument, day, version, kernel_set_key
                )
                if ancillary_files is None:
                    return
            # Always add the ancillary item so every run records its software versions.
            items.append(
                ancillary_item(
                    ancillary_files,
                    version,
                    kernel_set_key,
                    kernel_set_key_ttj2000ns,
                )
            )

        with data_table.batch_writer(
            overwrite_by_pkeys=["instrument_version", "time_utc"]
        ) as batch:
            for record in items:
                batch.put_item(Item=record)

        logger.info(
            "Inserted %d %s items for %s %s", len(items), instrument, day, version
        )
    except Exception as e:
        logger.error(
            "FAILED %s %s %s: %s: %s",
            instrument,
            event["day"],
            version,
            type(e).__name__,
            e,
        )
        raise
