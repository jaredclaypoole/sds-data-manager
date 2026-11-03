# I-ALiRT SSR refactoring tests

`tests/lambda_endpoints/test_ialirt_ssr_contract.py` runs `lambda_handler` through
both `ialirt_ssr_processing.py` and `ialirt_ssr_utils.py`. Neither module's
functions are mocked. The committed JSON files in
`tests/lambda_endpoints/ssr_traces/` record the original implementation's ordered
external calls, positional and keyword arguments, return/exception outcome, and
final data-access configuration. Arrays that fit within 100 columns are formatted
on one line, including short call records; larger structures use two-space
indentation. These are characterization tests: existing
behavior is the baseline, including any existing bugs.

## Running during a refactor

```bash
poetry run pytest tests/lambda_endpoints/test_ialirt_ssr_contract.py \
  --cov=sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing \
  --cov=sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils \
  --cov-branch --cov-report=term-missing --cov-fail-under=100
```

Keep the JSON baselines unchanged during the refactor. The normal test run reads
them without generating or updating them. Added, removed, reordered, or changed
recorded calls fail exact list equality. Arguments are copied when each call
happens, including full dataset values and DynamoDB items, so later mutations
cannot change the evidence. Set ordering in the metakernel `file_types` query
parameter is normalized because Python hash randomization changes it between
processes; all other argument list order is retained.

The 21 scenarios cover all eight instruments, unknown instruments, empty and
partially empty products, all four packet-coverage failures, missing L0 and
ancillary files, and failures in SSM, HTTP, science processing, and DynamoDB
writes. Each packet file contributes distinct data, with duplicates across files,
unsorted input, and a sequence-counter rollover. Packets at both trimming-buffer
boundaries are retained, while packets just outside them are removed. Gap
scenarios modify every overlapping copy consistently. Ancillary fixtures include
expired files and an end date equal to the processing day. Both modules reach
100% statement and branch coverage from these handler invocations alone.

## Recorded boundary and limits

Mocks record AWS resource/client construction and methods, HTTP response methods,
data-access queries and downloads, processing-input construction and methods,
packet decoding, calibration loading, science processors, SPICE pool lifetime,
time conversions, spacecraft geometry, directory cleanup/creation, file size,
and performance timing. One shared trace preserves order across dependencies
and both modules. Exception paths record pool/writer exit and the propagated
exception type and message. The tests use deterministic external results and
never contact AWS or download files.

NumPy, xarray, pandas dataset methods, Python built-ins, datetime parsing and
formatting, and logging run normally and their individual calls are **not**
compared. Their effects are reflected in subsequent recorded arguments. This
boundary therefore checks service/science interactions and data passed between
them; it does not promise literal equality of every call to code defined outside
the two modules. Extend the recording boundary if that stricter contract is
needed. It also does not detect a newly introduced external dependency that has
not been added to the fixture.

The clock, software-version metadata, and download directory are fixed. The
spacecraft cadence is set to twelve hours to keep the baseline readable while
exercising multiple iterations; the existing `test_spacecraft_items` unit test
checks the default 1,440 samples. Mock time conversions use a simple scale relative
to the fixture day, with different outputs for different timestamps. Mock geometry
varies by timestamp and coordinate frame, and CoDICE-Lo/Hi return distinct products
so selecting the wrong output changes the trace. Coverage measures branches taken
by these fixtures, not equivalence for every possible input or correctness of the science
algorithms. These tests intentionally preserve the current implementation's
behavior.

## Capturing an intentional new baseline

Only regenerate against a trusted implementation, before refactoring or after
reviewing an intentional behavior change:

```bash
SSR_UPDATE_TRACES=1 poetry run pytest tests/lambda_endpoints/test_ialirt_ssr_contract.py
```

Review the JSON diff and commit the test, fixture, and baseline changes together.
Do not regenerate simply to make a refactor pass: that would replace the evidence
of the old behavior with the new behavior. To add a scenario during a refactor,
first capture its baseline using the original two modules in a separate checkout.
