"""Test the I-ALiRT SSR processing lambda."""

# ruff: noqa: E402
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import imap_data_access
import numpy as np
import pytest
import xarray as xr
from imap_data_access.processing_input import ProcessingInputCollection, SPICEInput

from sds_data_manager.lambda_code.IAlirtCode import ialirt_ssr_utils
from sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils import (
    SOFTWARE_VERSIONS,
    IalirtDataGapError,
    ancillary_item,
    check_day_coverage,
    process_spacecraft_status,
    spacecraft_items,
    trim_to_day,
)

# The Lambda imports ialirt_ssr_utils by its plain name.
sys.modules["ialirt_ssr_utils"] = ialirt_ssr_utils

from sds_data_manager.lambda_code.IAlirtCode import ialirt_ssr_processing as processing
from sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing import (
    KERNELS,
    insert_formatted_data,
    lambda_handler,
    process_day,
    spice_item,
)

DAY_NS = 86400 * 10**9


@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils.str_yyyymmdd_to_ttj2000ns"
)
def test_check_day_coverage(mock_str_to_ttj2000ns):
    """Test that a complete day passes."""
    mock_str_to_ttj2000ns.side_effect = [0, DAY_NS]
    # A packet every 30 seconds until just after midnight.
    epoch = np.arange(0, 86430, 30) * 10**9
    combined = xr.Dataset(
        {"src_seq_ctr": ("epoch", np.arange(len(epoch)))}, coords={"epoch": epoch}
    )

    check_day_coverage(combined, datetime(2026, 10, 8))


@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils.ttj2000ns_to_utc")
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils.str_yyyymmdd_to_ttj2000ns"
)
def test_check_day_coverage_gap(mock_str_to_ttj2000ns, mock_to_utc):
    """Test that a missing packet raises."""
    mock_str_to_ttj2000ns.side_effect = [0, DAY_NS]
    epoch = np.arange(0, 86430, 30) * 10**9
    src_seq_ctr = np.arange(len(epoch))
    src_seq_ctr[100:] += 1
    combined = xr.Dataset(
        {"src_seq_ctr": ("epoch", src_seq_ctr)}, coords={"epoch": epoch}
    )

    with pytest.raises(IalirtDataGapError):
        check_day_coverage(combined, datetime(2026, 10, 8))


def test_trim_to_day():
    """Test that products outside the day are dropped."""
    result = [
        {"met_in_utc": "2026-10-07T23:59:59"},
        {"met_in_utc": "2026-10-08T12:00:00"},
        {"met_in_utc": "2026-10-09T00:00:00"},
    ]

    assert trim_to_day(result, datetime(2026, 10, 8)) == [
        {"met_in_utc": "2026-10-08T12:00:00"}
    ]


@pytest.mark.parametrize("instrument", ["mag", "hit"])
def test_insert_formatted_data(instrument):
    """Append science and optional MAG housekeeping without altering input records."""
    kernel_set_key = "2026-10-19T12:00:00.000001"
    product = {
        "apid": 478,
        "met": 1,
        "met_in_utc": "2026-10-08T12:00:00",
        "instrument": instrument,
        "x": 5,
        "mag_hk_status": {"hk": 1},
    }
    original = product.copy()
    existing = {"instrument_version": "spice#v001"}
    items = [existing]

    insert_formatted_data([product], items, instrument, "v001", kernel_set_key)

    expected = [
        existing,
        {
            "instrument_version": f"{instrument}#v001",
            "time_utc": product["met_in_utc"],
            "instrument": instrument,
            "x": 5,
            "kernel_set_key": kernel_set_key,
        },
    ]
    if instrument == "mag":
        expected.append(
            {
                "instrument_version": "mag_hk#v001",
                "time_utc": product["met_in_utc"],
                "instrument": "mag_hk",
                "mag_hk_status": {"hk": 1},
                "kernel_set_key": kernel_set_key,
            }
        )
    assert items == expected
    assert items[0] is existing
    assert product == original


def test_insert_formatted_data_missing_housekeeping():
    """MAG housekeeping formatting fails before any items are appended."""
    product = {"met_in_utc": "2026-10-08T12:00:00", "instrument": "mag"}
    items = []
    with pytest.raises(KeyError, match="mag_hk_status"):
        insert_formatted_data([product], items, "mag", "v001", "kernels")
    assert items == []


def test_spice_item():
    """Test the DynamoDB item for the SPICE kernels used."""
    dependencies = ProcessingInputCollection()
    dependencies.add(
        SPICEInput(
            "naif0012.tls",
            "imap_pred_20260922_20261020_v01.bsp",
            "imap_pred_20261020_20261117_v01.bsp",
        )
    )

    item = spice_item(dependencies, "v001", "2026-10-19T12:00:00.000001", 5)

    # Both ephemeris files are kept, in load order.
    assert item == {
        "instrument_version": "spice#v001",
        "time_utc": "2026-10-19T12:00:00.000001",
        "instrument": "spice",
        "spice_kernels": {
            "leapseconds": ["naif0012.tls"],
            "ephemeris_predicted": [
                "imap_pred_20260922_20261020_v01.bsp",
                "imap_pred_20261020_20261117_v01.bsp",
            ],
        },
        "ttj2000ns": 5,
    }


def test_ancillary_item():
    """Test the DynamoDB item for the ancillary files used."""
    ancillary_files = {"mag_l1d-calibration": "imap_mag_l1d-calibration.cdf"}

    item = ancillary_item(ancillary_files, "v001", "2026-10-19T12:00:00.000001", 5)

    assert item == {
        "instrument_version": "ancillary#v001",
        "time_utc": "2026-10-19T12:00:00.000001",
        "instrument": "ancillary",
        "ancillary_files": ancillary_files,
        "software_versions": SOFTWARE_VERSIONS,
        "ttj2000ns": 5,
    }


@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils.process_status")
def test_process_spacecraft_status(mock_process_status):
    """Test that there is one spacecraft status product per minute."""
    mock_process_status.return_value = [{"met_in_utc": "2026-10-08T12:00:00"}]
    # 3 minutes of packets, 1 per second.
    combined = xr.Dataset(coords={"epoch": np.arange(180) * 10**9})

    result = process_spacecraft_status(combined)

    assert len(result) == 3


@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils.et_to_ttj2000ns")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils.str_to_et")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils.imap_state")
def test_spacecraft_items(mock_imap_state, mock_str_to_et, mock_et_to_ttj2000ns):
    """Test that there is one spacecraft position item per minute."""
    mock_imap_state.return_value = np.ones((1440, 6))

    items = spacecraft_items(datetime(2026, 10, 8), "v001", "2026-10-19T12:00:00")

    assert len(items) == 1440
    assert items[0]["time_utc"] == "2026-10-08T00:00:00"
    assert items[0]["kernel_set_key"] == "2026-10-19T12:00:00"


@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.et_to_ttj2000ns")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.str_to_et")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.process_hit")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.trim_packets")
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.check_day_coverage"
)
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.parse_packets")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.query_filenames")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.spiceypy")
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.download_spice_file"
)
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.get_latest_spice_kernels"
)
def test_process_day_empty_product(
    mock_get_kernels,
    mock_download_spice,
    mock_spiceypy,
    mock_query,
    mock_parse,
    mock_check,
    mock_trim,
    mock_process_hit,
    mock_str_to_et,
    mock_et_to_ttj2000ns,
):
    """Test that nothing is written if a product is empty."""
    mock_process_hit.return_value = [{"met_in_utc": "2026-10-08T12:00:00"}, {}]
    table = MagicMock()

    process_day("hit", datetime(2026, 10, 8), "v001", "https://url", table)

    table.batch_writer.assert_not_called()


@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.et_to_ttj2000ns")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.str_to_et")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.process_hit")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.trim_packets")
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.check_day_coverage"
)
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.parse_packets")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.query_filenames")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.spiceypy")
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.download_spice_file"
)
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.get_latest_spice_kernels"
)
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.boto3")
def test_lambda_handler_empty_product(
    mock_boto3,
    mock_get_kernels,
    mock_download_spice,
    mock_spiceypy,
    mock_query,
    mock_parse,
    mock_check,
    mock_trim,
    mock_process_hit,
    mock_str_to_et,
    mock_et_to_ttj2000ns,
    monkeypatch,
    tmp_path,
):
    """Test that nothing is written if a product is empty."""
    mock_process_hit.return_value = [{"met_in_utc": "2026-10-08T12:00:00"}, {}]
    table = MagicMock()

    monkeypatch.setenv("INSTRUMENT", "hit")
    monkeypatch.setenv("SSR_TABLE", "ssr-table")
    monkeypatch.setenv("IMAP_DATA_ACCESS_URL", "https://url")
    monkeypatch.setenv("SSM_API_KEY_PARAMETER", "/ssr/api-key")
    monkeypatch.setattr(
        "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.DOWNLOAD_DIR",
        tmp_path / "ssr",
    )
    monkeypatch.setitem(imap_data_access.config, "API_KEY", None)
    mock_boto3.resource.return_value.Table.return_value = table
    lambda_handler({"day": "20261008", "version": "v001"}, {})

    table.batch_writer.assert_not_called()


@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.et_to_ttj2000ns")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.str_to_et")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.boto3.client")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.load_cdf")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.get_ancillary")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.process_packet")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.trim_packets")
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.check_day_coverage"
)
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.parse_packets")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.query_filenames")
@patch("sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.spiceypy")
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.download_spice_file"
)
@patch(
    "sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing.get_latest_spice_kernels"
)
def test_lambda_handler(
    mock_get_kernels,
    mock_download_spice,
    mock_spiceypy,
    mock_query,
    mock_parse,
    mock_check,
    mock_trim,
    mock_process_packet,
    mock_get_ancillary,
    mock_load_cdf,
    mock_ssm_client,
    mock_str_to_et,
    mock_et_to_ttj2000ns,
    setup_ssr_table,
    monkeypatch,
):
    """Test that lambda_handler writes MAG, MAG HK, spice and ancillary items."""
    monkeypatch.setenv("INSTRUMENT", "mag")
    monkeypatch.setenv("SSM_API_KEY_PARAMETER", "/imap-sdc/batch-jobs/api-key")
    # Restore the global imap_data_access config after the test.
    monkeypatch.setitem(imap_data_access.config, "API_KEY", None)
    mock_ssm_client.return_value.get_parameter.return_value = {
        "Parameter": {"Value": "test-api-key"}
    }
    mock_process_packet.return_value = [
        {"met_in_utc": "2026-10-08T12:00:00", "mag_hk_status": {"hk": 1}}
    ]
    mock_get_ancillary.side_effect = [
        Path("imap_mag_ialirt-calibration.cdf"),
        Path("imap_mag_l1b-calibration.cdf"),
    ]

    lambda_handler({"day": "20261008", "version": "v001"}, {})

    # MAG uses SPICE geometry, so it gets all kernel types.
    assert mock_get_kernels.call_args.args[2] == KERNELS
    assert imap_data_access.config["API_KEY"] == "test-api-key"
    # Each item is written, and the MAG item points to the spice and
    # ancillary items by kernel_set_key.
    table = setup_ssr_table["ssr_table"]
    mag = table.get_item(
        Key={"instrument_version": "mag#v001", "time_utc": "2026-10-08T12:00:00"}
    )["Item"]
    kernel_set_key = mag["kernel_set_key"]
    for instrument_version, time_utc in [
        ("mag_hk#v001", "2026-10-08T12:00:00"),
        ("spice#v001", kernel_set_key),
        ("ancillary#v001", kernel_set_key),
    ]:
        key = {"instrument_version": instrument_version, "time_utc": time_utc}
        assert "Item" in table.get_item(Key=key)


@pytest.mark.parametrize(
    ("instrument", "processor_name", "has_ancillary"),
    [
        ("mag", "process_packet", True),
        ("hit", "process_hit", False),
        ("swe", "process_swe", True),
        ("codice_lo", "process_codice", True),
        ("codice_hi", "process_codice", True),
        ("swapi", "process_swapi_ialirt", True),
        ("spacecraft_status", "process_spacecraft_status", False),
    ],
)
@pytest.mark.parametrize(
    "scenario", ["success", "empty", "mixed", "outside_day", "error"]
)
def test_process_algorithms(  # noqa: PLR0912, PLR0915
    monkeypatch, tmp_path, instrument, processor_name, has_ancillary, scenario
):
    """Check dispatch, trimming, pending items, provenance, and failure behavior."""
    day = datetime(2026, 10, 8)
    combined = xr.Dataset()
    kernel_set_key = "2026-11-01T00:00:00.000000"
    product = {
        "instrument": instrument,
        "met_in_utc": "2026-10-08T00:00:00",
        "apid": 478,
        "met": 1,
        "value": 5,
        "mag_hk_status": {"hk": 1},
    }
    outside = [
        dict(product, met_in_utc="2026-10-07T23:59:59"),
        dict(product, met_in_utc="2026-10-09T00:00:00"),
    ]
    results = {
        "success": [outside[0], product, outside[1]],
        "empty": [],
        "mixed": [product, {}],
        "outside_day": outside,
        "error": [product],
    }
    descriptors = {
        "mag": [("mag", "ialirt-calibration"), ("mag", "l1b-calibration")],
        "swe": [("swe", "l1b-in-flight-cal")],
        "codice_lo": [
            ("codice", "l1a-sci-lut"),
            ("codice", "l2-lo-efficiency"),
            ("codice", "l2-lo-gfactor"),
        ],
        "codice_hi": [("codice", "l1a-sci-lut"), ("codice", "l2-hi-ialirt-efficiency")],
        "swapi": [("swapi", "esa-unit-conversion")],
    }.get(instrument, [])
    ancillary = {f"{inst}_{desc}": f"{inst}_{desc}.cdf" for inst, desc in descriptors}
    assert bool(ancillary) == has_ancillary
    get_ancillary = MagicMock(
        side_effect=lambda inst, desc, day, directory: directory / f"{inst}_{desc}.cdf"
    )
    load_cdf = MagicMock(side_effect=lambda path: f"cdf:{path.name}")
    read_csv = MagicMock(return_value={"calibration": "swapi"})
    monkeypatch.setattr(processing, "get_ancillary", get_ancillary)
    monkeypatch.setattr(processing, "load_cdf", load_cdf)
    monkeypatch.setattr(processing.pd, "read_csv", read_csv)
    processors = {}
    for name in (
        "process_packet",
        "process_hit",
        "process_swe",
        "process_codice",
        "process_swapi_ialirt",
        "process_spacecraft_status",
    ):
        processors[name] = MagicMock()
        monkeypatch.setattr(processing, name, processors[name])
    selected = processors[processor_name]
    selected.return_value = results[scenario]
    if instrument == "codice_lo":
        selected.return_value = (results[scenario], [{"wrong_output": True}])
    elif instrument == "codice_hi":
        selected.return_value = ([{"wrong_output": True}], results[scenario])
    monkeypatch.setattr(processing, "DOWNLOAD_DIR", tmp_path)
    existing = {"instrument_version": "spice#v001", "time_utc": kernel_set_key}
    items = [existing]

    if scenario == "error":
        failure = RuntimeError("processor failed")
        selected.side_effect = failure
        with pytest.raises(RuntimeError, match="processor failed") as caught:
            processing.process_algorithms(
                combined, items, instrument, day, "v001", kernel_set_key
            )
        assert caught.value is failure
        assert items == [existing]
    else:
        provenance = processing.process_algorithms(
            combined, items, instrument, day, "v001", kernel_set_key
        )
        if scenario in {"empty", "mixed"}:
            assert provenance is None
            assert items == [existing]
        else:
            assert provenance == ancillary
            expected = [existing]
            if scenario == "success":
                expected.append(
                    {
                        "instrument_version": f"{instrument}#v001",
                        "time_utc": product["met_in_utc"],
                        "instrument": instrument,
                        "value": 5,
                        "kernel_set_key": kernel_set_key,
                    }
                )
                if instrument == "mag":
                    expected.append(
                        {
                            "instrument_version": "mag_hk#v001",
                            "time_utc": product["met_in_utc"],
                            "instrument": "mag_hk",
                            "mag_hk_status": {"hk": 1},
                            "kernel_set_key": kernel_set_key,
                        }
                    )
            assert items == expected
    assert items[0] is existing
    assert selected.call_count == 1
    assert selected.call_args.args[0] is combined
    expected_args = {
        "mag": ("cdf:mag_l1b-calibration.cdf", "cdf:mag_ialirt-calibration.cdf"),
        "swe": ([tmp_path / "swe_l1b-in-flight-cal.cdf"],),
        "codice_lo": (
            tmp_path / "codice_l1a-sci-lut.cdf",
            tmp_path / "codice_l2-lo-efficiency.cdf",
            "codice_lo",
            tmp_path / "codice_l2-lo-gfactor.cdf",
        ),
        "codice_hi": (
            tmp_path / "codice_l1a-sci-lut.cdf",
            tmp_path / "codice_l2-hi-ialirt-efficiency.cdf",
            "codice_hi",
        ),
        "swapi": ({"calibration": "swapi"},),
    }.get(instrument, ())
    assert selected.call_args.args[1:] == expected_args
    assert get_ancillary.call_args_list == [
        call(inst, desc, day, tmp_path) for inst, desc in descriptors
    ]
    if instrument == "mag":
        assert load_cdf.call_args_list == [
            call(tmp_path / "mag_ialirt-calibration.cdf"),
            call(tmp_path / "mag_l1b-calibration.cdf"),
        ]
    else:
        load_cdf.assert_not_called()
    if instrument == "swapi":
        read_csv.assert_called_once_with(tmp_path / "swapi_esa-unit-conversion.cdf")
    else:
        read_csv.assert_not_called()
    assert selected.call_args.kwargs == {}
    for name, processor in processors.items():
        if name != processor_name:
            processor.assert_not_called()


@pytest.mark.parametrize("instrument", ["unknown", "spacecraft"])
def test_process_algorithms_unknown_instrument(instrument):
    """Reject unsupported instruments without modifying pending items."""
    items = [{"instrument_version": "spice#v001"}]
    with pytest.raises(ValueError, match=f"^Unknown instrument: {instrument}$"):
        processing.process_algorithms(
            xr.Dataset(), items, instrument, datetime(2026, 10, 8), "v001", "kernels"
        )
    assert items == [{"instrument_version": "spice#v001"}]
