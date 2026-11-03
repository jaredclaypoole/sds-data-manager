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

Each JSON file contains the explored runs for one instrument. A run stores its
boolean `choices`, ordered `steps`, and returned `result` or propagated `error`.
A step is `[operation, positional_arguments, keyword_arguments, result_reference]`.
`{"ref": "v42"}` refers to the result produced by an earlier step. External roots
use qualified names; references do not depend on object addresses.

To avoid repeating common setup, `prefix: [parent_run, step_count]` means prepend
the first `step_count` fully expanded steps of that earlier run to the stored
`steps`. A zero count means no prefix. This is lossless: call order, arguments,
operations, and outcomes are retained. The shared formatter keeps containers
inline when the entire line fits within 100 columns.

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
