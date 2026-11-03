# Instrument-only SSR contract discovery

This harness supplies only eight instrument names and one unknown name. Event
fields, other environment variables, and dependency results are synthetic values.
There are no packet fixtures, calibration tables, expected exceptions, processor
outputs, or per-instrument setup. The handler and both modules' helpers run normally.

`test_ialirt_ssr_contract.py` defines the input names and mock boundary.
`utils.py` contains reusable discovery tools. The formatter is shared with the
behavior-aware harness in `../utils.py`.

Run against the committed baseline:

```bash
poetry run pytest tests/lambda_endpoints/ssr_refactor/behavior_agnostic
```

Capture baselines only against a trusted implementation, before refactoring:

```bash
SSR_UPDATE_TRACES=1 poetry run pytest tests/lambda_endpoints/ssr_refactor/behavior_agnostic
```

Normal runs never update baselines. Review the generated diff before committing;
regenerating after a refactor would replace the evidence being compared.

## Discovery

`external_bindings()` identifies imported functions, constructors, modules, and
external instances in the two target modules. It excludes functions and modules
defined within that pair. This includes logging, NumPy, xarray, datetime, paths,
AWS, HTTP, science processing, and SPICE. Callers can instead pass an explicit list
of `(module, attribute)` pairs to `discover()` to choose a narrower boundary.
Aliases for the same external object share a substitute. Patches are local to the
target modules and are restored after every execution. Actual dependencies are
never called during discovery; initial module imports still run normally.

Each dependency returns a symbolic value that supports calls, attributes, indexing,
arithmetic, comparisons, iteration, and context managers. Every operation receives
a distinct reference, so changing the selected output or argument changes the
trace. Arguments are copied at the time of each operation. Context exits record
exception type and message without traceback addresses.

Discovery first executes with generic boolean defaults, then reruns with opposite
choices at encountered boolean sites. Bytecode locations are used only to plan
exploration; source filenames and line numbers are not stored in the baseline.
Repeated tests at one site use the same choice within an execution, allowing a
filter to exclude all synthetic elements without a hand-written empty-data case.
The explorer aims to observe both outcomes at each encountered site, rather than
enumerate every combination. Run, operation, and decision limits fail discovery
instead of silently saving incomplete exploration.

Top-level set constants retain set membership semantics but iterate in canonical
order, avoiding Python hash-seed differences. No mission-specific normalization
rules or downloaded file paths are needed.

## Trace format

Each JSON file contains the explored runs for one instrument. Every execution
is self-contained, with `choices`, its `result` or propagated `error`, and the
complete ordered `steps`. There is no call index or shared-prefix indirection.

Steps have named fields. For example:

```json
{
  "call": "boto3.client.get_parameter",
  "args": [],
  "kwargs": {"Name": {"ref": "v18"}, "WithDecryption": true},
  "result": {"ref": "v19"}
}
```

The call name is a display path, with no receiver or callable-object metadata.
It may include a synthetic reference, such as `v32.strftime`, when the originating
class is unknown. `args` and `kwargs` contain only explicitly passed arguments.
Calls contain exactly `call`, `args`, `kwargs`, and `result`.

Attribute lookups used only as call targets are omitted; their names are shown
in the calls themselves. Lookups used as arguments, as operands of other
operations, or as returned values remain, as do unused lookups. Invocation no
longer records its exact callable reference, and the count and timing of omitted
attribute lookups are not compared. Thus the raw discovery records cannot always
be reconstructed unambiguously from the readable format.

Other operations also name their operands:

```json
{"getattr": "get_parameter", "object": {"ref": "v16"}, "result": {"ref": "v17"}}
{"getitem": {"ref": "v19"}, "key": "Parameter", "result": {"ref": "v20"}}
{"setitem": {"ref": "v22"}, "key": "API_KEY", "value": {"ref": "v21"}, "result": {"ref": "v23"}}
{"bool": {"ref": "v40"}, "value": true, "result": {"ref": "v41"}}
{"op": "add", "left": {"ref": "v10"}, "right": 1, "result": {"ref": "v11"}}
```

Iteration names its `index`, formatting its `spec`, and scalar conversions their
chosen `value`. Context entry uses `enter`; exit uses `exit`, `exception_type`,
and `exception`. Seeded lookups include a `value` as well as the key. Every result
reference remains attached to its operation; `{"ref": "v42"}` points to an earlier
result in the same execution. External imports and module variables use their
local names in root references, such as `imap_data_access` and `DOWNLOAD_DIR`.
Only ambiguous names retain enough module qualification to distinguish them;
generated references such as `v22` remain unchanged. Literal argument strings
are not shortened.
Types defined in the target modules also use local names, such as
`IalirtDataGapError`. Imported types retain their qualification, such as
`builtins.FileNotFoundError`.

Discovery still compresses prefixes internally. `readable_runs()` expands them
and replaces positional records with named fields only for comparison and baseline
generation. Operation order, explicit arguments, result references, and outcomes are
preserved; callable and receiver metadata are omitted from calls. The formatter
preserves field order so the operation appears first, keeping containers inline
when the full line fits within 100 columns.

## Limits

These are synthetic protocol contracts. They do not establish that mocked values
are valid service responses or scientific data, or that explored paths are
feasible with real dependencies. Symbolic comparisons are independent choices,
not mathematical constraints. Some discovered exceptions can therefore arise from
the synthetic values themselves. The trace includes synthetic operations as well
as calls, so an equivalent rewrite of argument expressions may require inspection.

Iteration and integer conversion use two elements/steps; floating-point conversion
uses a fixed value. Empty iterators, other tuple sizes, numeric edge cases, arbitrary
dependency exceptions, and all path combinations are not explored. A new Python
protocol may require extending `Symbol`, without adding SSR-specific return data.
Built-in operations on concrete values and internal constants still execute
normally; coverage does not imply equivalence for every input. Keep the sibling
behavior-aware harness for realistic data manipulation and explicit failure cases.

Measure coverage of the two target modules separately from the discovery tools:

```bash
poetry run pytest tests/lambda_endpoints/ssr_refactor/behavior_agnostic/test_ialirt_ssr_contract.py \
  --cov=sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_processing \
  --cov=sds_data_manager.lambda_code.IAlirtCode.ialirt_ssr_utils \
  --cov-branch --cov-report=term-missing
```
