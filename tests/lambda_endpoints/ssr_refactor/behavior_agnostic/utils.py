"""Discover call contracts using synthetic values, without dependency fixtures.

These values model Python protocols, not physical data or service responses.
Boolean choices are explored independently, so some paths may be infeasible
with real dependencies. No wrapped dependency is ever executed.
"""

from __future__ import annotations

import inspect
import json
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import CodeType, ModuleType, TracebackType
from typing import Literal, TypeAlias, TypedDict, cast

import pytest

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
# Steps retain their JSON list representation: operation, args, kwargs, result.
Step: TypeAlias = list[JSONValue]
Binding: TypeAlias = tuple[ModuleType, str]
DecisionSite: TypeAlias = tuple[CodeType, int]
Decision: TypeAlias = tuple[DecisionSite, bool]


class Run(TypedDict):
    """One complete discovered execution before prefix compression."""

    choices: list[bool]
    steps: list[Step]
    result: JSONValue
    error: JSONValue


class PackedRun(Run):
    """An execution whose steps omit the shared prefix [parent, length]."""

    prefix: list[int]


class ReadableRun(TypedDict):
    """A self-contained execution with named operation fields."""

    choices: list[bool]
    steps: list[dict[str, JSONValue]]
    result: JSONValue
    error: JSONValue


class ExplorationLimitError(RuntimeError):
    """Discovery exceeded a configured resource limit."""


class UnsupportedValueError(TypeError):
    """The recorder needs a representation for a new concrete argument type."""


def snapshot(value: object) -> JSONValue:  # noqa: PLR0911
    """Serialize synthetic references and ordinary Python arguments."""
    if isinstance(value, Symbol):
        return {"ref": value._label}
    if isinstance(value, slice):
        return {
            "slice": [snapshot(value.start), snapshot(value.stop), snapshot(value.step)]
        }
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: snapshot(item) for key, item in value.items()}
        return {
            "mapping": [[snapshot(key), snapshot(item)] for key, item in value.items()]
        }
    if isinstance(value, tuple | list):
        return [snapshot(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((snapshot(item) for item in value), key=repr)
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, BaseException):
        return {"exception": type(value).__name__, "message": str(value)}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise UnsupportedValueError(f"Unsupported discovery argument: {type(value)}")


class Recorder:
    """Record operations and choose booleans from one exploration prefix."""

    def __init__(
        self,
        choices: Sequence[bool] = (),
        max_steps: int = 3000,
        max_decisions: int = 128,
    ) -> None:
        """Start one isolated execution."""
        self.choices = choices
        self.max_steps = max_steps
        self.max_decisions = max_decisions
        self.steps: list[Step] = []
        self.decisions: list[bool] = []
        self.sites: list[DecisionSite] = []
        self.branch_choices: dict[DecisionSite, bool] = {}

    def operation(self, operation: str, /, *args: object, **kwargs: object) -> Symbol:
        """Produce a distinct synthetic result for each operation."""
        if len(self.steps) >= self.max_steps:
            raise ExplorationLimitError("Maximum operations per execution exceeded")
        result = Symbol(self, f"v{len(self.steps)}")
        self.steps.append(
            [operation, snapshot(args), snapshot(kwargs), snapshot(result)]
        )
        return result

    def choose(self, value: Symbol) -> bool:
        """Record a truth test; unsupplied choices use the value's default."""
        frame = inspect.currentframe()
        assert frame is not None
        assert frame.f_back is not None
        assert frame.f_back.f_back is not None
        frame = frame.f_back.f_back
        site = (frame.f_code, frame.f_lasti)
        del frame
        if site not in self.branch_choices:
            index = len(self.decisions)
            if index >= self.max_decisions:
                raise ExplorationLimitError(
                    "Maximum boolean decisions per execution exceeded"
                )
            choice = (
                self.choices[index] if index < len(self.choices) else value._default
            )
            self.decisions.append(choice)
            self.sites.append(site)
            self.branch_choices[site] = choice
        choice = self.branch_choices[site]
        self.operation("bool", value, choice)
        return choice


class Symbol:
    """A generic callable, container, scalar, and context-manager substitute."""

    def __init__(self, recorder: Recorder, label: str, default: bool = True) -> None:
        """Keep only a stable reference and an exploration policy."""
        self._recorder = recorder
        self._label = label
        self._default = default

    def __getattr__(self, name: str) -> Symbol:
        """Represent an attribute read without touching the real dependency."""
        return self._recorder.operation("getattr", self, name)

    def __call__(self, /, *args: object, **kwargs: object) -> Symbol:
        """Record the function and its arguments."""
        return self._recorder.operation("call", self, *args, **kwargs)

    def __getitem__(self, key: object) -> object:
        """Represent indexing or a mapping lookup."""
        return self._recorder.operation("getitem", self, key)

    def __setitem__(self, key: object, value: object) -> None:
        """Record writes to external mappings."""
        self._recorder.operation("setitem", self, key, value)

    def __iter__(self) -> Iterator[Symbol]:
        """Provide a fixed, bounded number of synthetic elements."""
        # Two elements support ordinary loops and pair unpacking without knowing
        # the dependency's return schema. The fixed bound prevents infinite loops.
        for index in range(2):
            yield self._recorder.operation("iter", self, index)

    def __len__(self) -> int:
        """Use a fixed container size."""
        self._recorder.operation("len", self, 2)
        return 2

    def __bool__(self) -> bool:
        """Let discovery choose this branch."""
        return self._recorder.choose(self)

    def __int__(self) -> int:
        """Use a bounded integer suitable for loop counts."""
        self._recorder.operation("int", self, 2)
        return 2

    __index__ = __int__

    def __float__(self) -> float:
        """Use a fixed synthetic floating-point value."""
        self._recorder.operation("float", self, 1.25)
        return 1.25

    def __str__(self) -> str:
        """Avoid memory addresses and environment-specific representations."""
        return f"<{self._label}>"

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        """Record formatting without requiring a real scalar or datetime."""
        result = self._recorder.operation("format", self, spec)
        return str(result)

    def __hash__(self) -> int:
        """Allow symbolic keys to participate in container comparisons."""
        # A common hash lets set/dict lookups exercise symbolic equality.
        return 0

    def __enter__(self) -> Symbol:
        """Represent context-manager entry."""
        return self._recorder.operation("enter", self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Record exceptions, exclude tracebacks, and propagate failures."""
        self._recorder.operation("exit", self, exc_type, exc)
        return False


def install_operators() -> None:
    """Implement arithmetic and comparisons with one recording adapter."""
    for name in (
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "pow",
        "and",
        "or",
        "xor",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ):

        def binary(self: Symbol, other: object, operation: str = name) -> Symbol:
            default = operation not in {"eq", "ne", "lt", "gt"}
            if operation == "eq":
                default = isinstance(other, Symbol)
            elif operation == "lt":
                default = not isinstance(other, Symbol)
            result = self._recorder.operation(operation, self, other)
            result._default = default
            return result

        def reflected(self: Symbol, other: object, operation: str = name) -> Symbol:
            return self._recorder.operation(operation, other, self)

        setattr(Symbol, f"__{name}__", binary)
        if name not in {"eq", "ne", "lt", "le", "gt", "ge"}:
            setattr(Symbol, f"__r{name}__", reflected)


install_operators()


class StableSet(set[object]):
    """Give unordered constant sets deterministic iteration across processes."""

    def __iter__(self) -> Iterator[object]:
        """Return a canonical iteration order."""
        return iter(sorted(super().__iter__(), key=repr))


class Inputs(Symbol):
    """A synthetic mapping with optional concrete input keys."""

    def __init__(
        self,
        recorder: Recorder,
        label: str,
        values: Mapping[object, object] | None = None,
    ) -> None:
        """Seed only inputs the caller explicitly wants to vary."""
        super().__init__(recorder, label)
        self._values = values or {}

    def __getitem__(self, key: object) -> object:
        """Return a concrete seed when supplied, otherwise a synthetic value."""
        if key in self._values:
            self._recorder.operation("getitem", self, key, self._values[key])
            return self._values[key]
        return super().__getitem__(key)


def external_bindings(modules: Sequence[ModuleType]) -> list[Binding]:
    """Find external imports/instances, preserving functions in the target modules."""
    local_modules = {module.__name__ for module in modules}
    bindings: list[Binding] = []
    for module in modules:
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            origin = (
                value.__name__
                if isinstance(value, ModuleType)
                else getattr(value, "__module__", type(value).__module__)
            )
            if origin not in local_modules | {"builtins"}:
                bindings.append((module, name))
    return bindings


def pack_runs(runs: Sequence[Run]) -> list[PackedRun]:
    """Share identical operation prefixes without discarding execution details."""
    packed: list[PackedRun] = []
    for run_index, run in enumerate(runs):
        steps = run["steps"]
        best_length, best_parent = 0, 0
        for parent in range(run_index):
            previous = runs[parent]["steps"]
            length = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(previous, steps, strict=False)
                    )
                    if left != right
                ),
                min(len(previous), len(steps)),
            )
            if length > best_length:
                best_length, best_parent = length, parent
        packed.append(
            {**run, "prefix": [best_parent, best_length], "steps": steps[best_length:]}
        )
    return packed


def expand_runs(runs: Sequence[PackedRun]) -> list[Run]:
    """Restore each execution's complete steps without changing any records."""
    expanded: list[Run] = []
    for run in runs:
        parent, length = run["prefix"]
        prefix = expanded[parent]["steps"][:length] if length else []
        expanded.append(
            {
                "choices": run["choices"],
                "steps": prefix + run["steps"],
                "result": run["result"],
                "error": run["error"],
            }
        )
    return expanded


def _describe(value: JSONValue, names: Mapping[str, str]) -> str:
    """Resolve a display name while leaving the underlying reference intact."""
    if isinstance(value, dict) and "ref" in value:
        reference = cast(str, value["ref"])
        return names.get(reference, reference.rsplit(".", 1)[-1])
    return json.dumps(value, sort_keys=True)


# Non-call operands receive names in their original order. Optional trailing
# operands (for example a seeded getitem value) remain optional in the JSON.
OPERATION_FIELDS: dict[str, tuple[str, ...]] = {
    "getattr": ("object", "getattr"),
    "getitem": ("getitem", "key", "value"),
    "setitem": ("setitem", "key", "value"),
    "iter": ("iter", "index"),
    "format": ("format", "spec"),
    "bool": ("bool", "value"),
    "len": ("len", "value"),
    "int": ("int", "value"),
    "float": ("float", "value"),
    "enter": ("enter",),
    "exit": ("exit", "exception_type", "exception"),
}


def _references(value: JSONValue) -> Iterator[str]:
    """Find explicit symbolic references in nested operands and outcomes."""
    if isinstance(value, dict):
        if isinstance(value.get("ref"), str):
            yield cast(str, value["ref"])
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _omit_callable_lookups(
    steps: list[dict[str, JSONValue]], run: Run
) -> list[dict[str, JSONValue]]:
    """Omit lookup definitions used exclusively as named call targets."""
    targets = {
        cast(dict[str, str], cast(list[JSONValue], args)[0])["ref"]
        for operation, args, _, _ in run["steps"]
        if operation == "call"
    }
    used = set(_references(run["result"])) | set(_references(run["error"]))
    for step in steps:
        for field, value in step.items():
            if field != "result":
                used.update(_references(value))
    redundant = targets - used
    return [
        step
        for step in steps
        if not (
            "getattr" in step
            and cast(dict[str, str], step["result"])["ref"] in redundant
        )
    ]


def _rename_references(
    value: JSONValue, aliases: Mapping[str, str], local_types: Mapping[str, str]
) -> JSONValue:
    """Shorten references and local types without changing literal strings."""
    if isinstance(value, dict):
        return {
            key: aliases[item]
            if key == "ref" and isinstance(item, str)
            else local_types.get(item, item)
            if key == "type" and isinstance(item, str)
            else _rename_references(item, aliases, local_types)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_references(item, aliases, local_types) for item in value]
    return value


def _shorten_roots(
    runs: list[ReadableRun], local_types: Mapping[str, str]
) -> list[ReadableRun]:
    """Use the shortest unique suffix for each reference across executions."""
    value = cast(JSONValue, runs)
    references = set(_references(value))
    depths = dict.fromkeys(references, 1)
    while True:
        aliases = {
            reference: ".".join(reference.split(".")[-depths[reference] :])
            for reference in references
        }
        counts: dict[str, int] = {}
        for alias in aliases.values():
            counts[alias] = counts.get(alias, 0) + 1
        collisions = {
            reference for reference, alias in aliases.items() if counts[alias] > 1
        }
        if not collisions:
            return cast(
                list[ReadableRun], _rename_references(value, aliases, local_types)
            )
        for reference in collisions:
            depths[reference] += 1


def readable_runs(  # noqa: PLR0912
    runs: Sequence[PackedRun], local_modules: Sequence[ModuleType] = ()
) -> list[ReadableRun]:
    """Expand executions and display names, arguments, and results directly."""
    readable: list[ReadableRun] = []
    for run in expand_runs(runs):
        names: dict[str, str] = {}
        steps: list[dict[str, JSONValue]] = []
        for operation_value, arguments, keywords, result in run["steps"]:
            operation = cast(str, operation_value)
            args = cast(list[JSONValue], arguments)
            reference = cast(dict[str, str], result)["ref"]
            owner = _describe(args[0], names)
            if operation == "getattr":
                names[reference] = f"{owner}.{args[1]}"
            elif operation == "getitem":
                names[reference] = f"{owner}[{_describe(args[1], names)}]"
            elif operation == "iter":
                names[reference] = owner
            elif operation == "call":
                # The result keeps the callable's path, without parentheses.
                # Object identity is represented separately by its result ref.
                names[reference] = owner
            elif operation == "enter":
                names[reference] = f"{owner}.__enter__"

            if operation == "call":
                step: dict[str, JSONValue] = {
                    "call": owner,
                    "args": args[1:],
                    "kwargs": keywords,
                    "result": result,
                }
            elif operation in OPERATION_FIELDS:
                fields = OPERATION_FIELDS[operation]
                step = dict(zip(fields, args, strict=False))
                if len(args) > len(fields):
                    raise ValueError(f"Unexpected operands for {operation}")
                step = {operation: step[operation], **step}
                step["result"] = result
                if keywords:
                    step["kwargs"] = keywords
            else:
                step = {
                    "op": operation,
                    "left": args[0],
                    "right": args[1],
                    "result": result,
                }
                if len(args) != 2:
                    raise ValueError(f"Unexpected operands for {operation}")
                if keywords:
                    step["kwargs"] = keywords
            steps.append(step)
        readable.append(
            {
                "choices": run["choices"],
                "result": run["result"],
                "error": run["error"],
                "steps": _omit_callable_lookups(steps, run),
            }
        )
    local_types = {
        f"{value.__module__}.{value.__qualname__}": value.__qualname__
        for module in local_modules
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__
    }
    return _shorten_roots(readable, local_types)


def discover(
    invoke: Callable[[Recorder], object],
    bindings: Sequence[Binding],
    max_runs: int = 128,
) -> list[PackedRun]:
    """Explore truth choices with fresh mocks and return all observed contracts.

    The caller supplies only a callable and mock locations. Explore each observed
    boolean site both ways, holding repeated tests at that site consistent within
    an execution. This is branch exploration, not exhaustive path enumeration.
    Queued paths are never silently dropped: exhausting the budget fails discovery.
    """
    pending: deque[tuple[tuple[bool, ...], Decision | None]] = deque([((), None)])
    scheduled: set[Decision] = set()
    observed: set[Decision] = set()
    runs: list[Run] = []
    while pending:
        prefix, goal = pending.popleft()
        if goal in observed:
            continue
        if len(runs) >= max_runs:
            raise ExplorationLimitError(
                f"Maximum executions exceeded ({max_runs}); {len(pending)} pending"
            )
        recorder = Recorder(prefix)
        with pytest.MonkeyPatch.context() as patch:
            roots: dict[int, Symbol] = {}
            for module, name in bindings:
                original = getattr(module, name)
                # Shared imports remain shared across the two modules.
                root = roots.setdefault(
                    id(original), Symbol(recorder, f"{module.__name__}.{name}")
                )
                patch.setattr(module, name, root)
            for module in {module for module, _ in bindings}:
                for name, value in vars(module).copy().items():
                    if isinstance(value, set | frozenset):
                        patch.setattr(module, name, StableSet(value))
            try:
                result = snapshot(invoke(recorder))
                error = None
            except (ExplorationLimitError, UnsupportedValueError):
                raise
            except Exception as exc:
                result, error = None, snapshot(exc)
        runs.append(
            {
                "choices": recorder.decisions,
                "steps": recorder.steps,
                "result": result,
                "error": error,
            }
        )
        observed.update(zip(recorder.sites, recorder.decisions, strict=True))
        for index, site in enumerate(recorder.sites):
            goal = (site, not recorder.decisions[index])
            alternative = (*recorder.decisions[:index], not recorder.decisions[index])
            if goal not in observed and goal not in scheduled:
                scheduled.add(goal)
                pending.append((alternative, goal))
    return pack_runs(runs)
