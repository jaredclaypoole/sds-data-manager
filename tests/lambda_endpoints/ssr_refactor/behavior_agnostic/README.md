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

## Development next steps

The following changes are proposed, not implemented. They aim to reduce repeated
iteration work while preserving evidence that checks occurred and exploring
symbolic truth outcomes within configurable resource limits.

### Summary

1. Keep traces flat and retain the readable format described above.
2. Use CPython bytecode to choose iteration length: one element for ordinary
   iteration, exactly enough for fixed unpacking, and required elements plus one
   starred element for starred unpacking.
3. Explore symbolic truth tests through `__bool__`. Let Python handle
   short-circuiting, `any()`/`all()`, walrus assignments, `break`, and `continue`.
   No AST classification is needed.
4. Bound exploration with configurable limits on decisions per execution and
   total executions.
5. Support two limit policies: fail the test, or warn and explicitly report
   incomplete exploration while retaining completed executions.
6. Explore both truth outcomes of all comparisons involving symbols, including
   comparisons with concrete values. Symbol version IDs identify values only;
   they do not determine ordering or equality.
7. Validate the implementation and regenerate agnostic traces, keeping
   behavior-aware traces unchanged.

Runtime scope tracking and per-scope budgets remain optional extensions.
Mutable-container proxies and sorting-specific mocks are outside the core plan.

### Details

**Consumer-aware iteration.** Inspect the consuming instruction from
`Symbol.__iter__` using the caller's frame and `dis.get_instructions()`. For
`UNPACK_SEQUENCE`, yield the instruction's required number of elements. For
`UNPACK_EX`, decode its leading and trailing target counts and yield those
elements plus one for the starred target. Other consumers receive one symbolic
element. This supports nested unpacking without source parsing. Instruction
lookup must accommodate inline caches and be tested on supported CPython
versions. CPython-specific behavior is an accepted dependency.

**Flat execution records.** Record yielded elements and subsequent operations
in their actual execution order. Do not infer comprehension or loop block
boundaries: an iterator receives no reliable notification when a consumer stops
early. `break` and short-circuiting consumers may stop requesting elements;
`continue` requests the next element and reaches the ordinary one-element bound.
These paths remain visible through their recorded operations and outcomes.

**Truth-test exploration.** Make symbolic truth tests eligible for exploration
without distinguishing explicit `if` conditions from comprehension filters or
comparisons inside sorting and container operations. Explore component truth
tests rather than forcing only the final outcome of a compound condition. Python
continues to execute assignments and calls in conditions, preserving walrus side
effects and the evaluation order of `and`, `or`, and chained comparisons.

`any()` and `all()` need no special mocks: they truth-test yielded symbolic
elements through `__bool__`. For a reusable one-element iterable,
`if any(result) and all(result)` can explore three paths: false `any()` with
`all()` skipped, true `any()` followed by false `all()`, and both true. Actual
consumption must still govern single-use iterators. Empty inputs and mixed-element
sequences are not covered by the one-element policy.

**Exploration limits.** Configure decisions per execution separately from total
executions: limiting decision depth alone does not prevent a large number of
paths. Retain the existing operation limit as an execution guard. Define whether
the decision limit counts distinct sites or repeated evaluations, and document
the exploration strategy; eligibility of every truth test does not itself imply
enumeration of every decision combination.

Limit exceptions must escape the target code and reach the discovery runner,
rather than being recorded as application failures. In fail mode, discovery
propagates the harness exception so the test fails. In warn mode, discovery prunes
the interrupted path, retains completed executions, and continues other queued
paths where resources permit. Reaching the total execution limit stops further
runs. Emit a warning and persist an explicit incomplete-exploration status with
the limit reached, so updating baselines cannot silently conceal truncation.
Any completeness claim applies only to the configured symbolic model and search
strategy, not all real inputs.

**Comparisons.** All comparisons involving symbols produce symbolic booleans.
When truth-tested, equality, inequality, and ordering comparisons are eligible
for exploration of both outcomes, whether the other operand is symbolic or
concrete. Version IDs serve only as reference identities and never determine
comparison outcomes. This policy also applies to comparisons encountered inside
sorting and dictionary/set operations; configured limits bound their exploration.
Hashing and container identity shortcuts still need explicit handling and tests:
Python may bypass equality when hashes differ or keys are the same object.
Exploring comparison outcomes does not enforce mathematical consistency between
separate comparisons.

**Validation.** Add focused tests for fixed and starred unpacking, trailing
targets, nested unpacking, ordinary loops, and `list()`. Exercise short-circuiting,
walrus side effects, `any()`/`all()`, single-use iterators, `break`, and `continue`.
Verify both limit policies, incomplete-result reporting, symbol-to-symbol and
symbol-to-concrete comparisons, and stable traces across processes. Regenerate the agnostic
baselines against the trusted implementation and inspect the differences. Verify
that extracting or modifying shared helpers leaves all behavior-aware traces
byte-for-byte unchanged. Statement and branch coverage remain useful checks,
but do not prove exhaustive symbolic paths or realistic data correctness.
