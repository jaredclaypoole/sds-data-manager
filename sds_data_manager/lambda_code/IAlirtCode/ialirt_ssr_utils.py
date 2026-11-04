"""I-ALiRT SSR helpers with no counterpart in realtime processing.

ialirt_ssr_processing.py mirrors realtime (ialirt_ingest.py) where it can.
The SSR-only steps live here: checking and trimming a day of packets,
per-minute spacecraft status and position, and the ancillary item.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from importlib.metadata import version as package_version

import numpy as np
import xarray as xr
from imap_processing.ialirt.l0.process_status import process_status
from imap_processing.spice.geometry import SpiceBody, SpiceFrame, imap_state
from imap_processing.spice.time import (
    et_to_ttj2000ns,
    et_to_utc,
    str_to_et,
    str_yyyymmdd_to_ttj2000ns,
    ttj2000ns_to_et,
)

# Spacecraft position/velocity cadence over the day.
SPACECRAFT_CADENCE = timedelta(minutes=1)
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
