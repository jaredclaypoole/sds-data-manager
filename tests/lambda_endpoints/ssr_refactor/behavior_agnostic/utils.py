"""Discover call contracts using synthetic values, without dependency fixtures.

These values model Python protocols, not physical data or service responses.
Boolean choices are explored independently, so some paths may be infeasible
with real dependencies. No wrapped dependency is ever executed.
"""

import inspect
from collections import deque
from types import ModuleType

import pytest


class ExplorationLimitError(RuntimeError):
    """Discovery exceeded a configured resource limit."""


class UnsupportedValueError(TypeError):
    """The recorder needs a representation for a new concrete argument type."""


def snapshot(value):  # noqa: PLR0911
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

    def __init__(self, choices=(), max_steps=3000, max_decisions=128):
        """Start one isolated execution."""
        self.choices = choices
        self.max_steps = max_steps
        self.max_decisions = max_decisions
        self.steps = []
        self.decisions = []
        self.sites = []
        self.branch_choices = {}

    def operation(self, operation, /, *args, **kwargs):
        """Produce a distinct synthetic result for each operation."""
        if len(self.steps) >= self.max_steps:
            raise ExplorationLimitError("Maximum operations per execution exceeded")
        result = Symbol(self, f"v{len(self.steps)}")
        self.steps.append(
            [operation, snapshot(args), snapshot(kwargs), snapshot(result)]
        )
        return result

    def choose(self, value):
        """Record a truth test; unsupplied choices use the value's default."""
        frame = inspect.currentframe().f_back.f_back
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

    def __init__(self, recorder, label, default=True):
        """Keep only a stable reference and an exploration policy."""
        self._recorder = recorder
        self._label = label
        self._default = default

    def __getattr__(self, name):
        """Represent an attribute read without touching the real dependency."""
        return self._recorder.operation("getattr", self, name)

    def __call__(self, /, *args, **kwargs):
        """Record the function and its arguments."""
        return self._recorder.operation("call", self, *args, **kwargs)

    def __getitem__(self, key):
        """Represent indexing or a mapping lookup."""
        return self._recorder.operation("getitem", self, key)

    def __setitem__(self, key, value):
        """Record writes to external mappings."""
        self._recorder.operation("setitem", self, key, value)

    def __iter__(self):
        """Provide a fixed, bounded number of synthetic elements."""
        # Two elements support ordinary loops and pair unpacking without knowing
        # the dependency's return schema. The fixed bound prevents infinite loops.
        for index in range(2):
            yield self._recorder.operation("iter", self, index)

    def __len__(self):
        """Use a fixed container size."""
        self._recorder.operation("len", self, 2)
        return 2

    def __bool__(self):
        """Let discovery choose this branch."""
        return self._recorder.choose(self)

    def __int__(self):
        """Use a bounded integer suitable for loop counts."""
        self._recorder.operation("int", self, 2)
        return 2

    __index__ = __int__

    def __float__(self):
        """Use a fixed synthetic floating-point value."""
        self._recorder.operation("float", self, 1.25)
        return 1.25

    def __str__(self):
        """Avoid memory addresses and environment-specific representations."""
        return f"<{self._label}>"

    __repr__ = __str__

    def __format__(self, spec):
        """Record formatting without requiring a real scalar or datetime."""
        result = self._recorder.operation("format", self, spec)
        return str(result)

    def __hash__(self):
        """Allow symbolic keys to participate in container comparisons."""
        # A common hash lets set/dict lookups exercise symbolic equality.
        return 0

    def __enter__(self):
        """Represent context-manager entry."""
        return self._recorder.operation("enter", self)

    def __exit__(self, exc_type, exc, traceback):
        """Record exceptions, exclude tracebacks, and propagate failures."""
        self._recorder.operation("exit", self, exc_type, exc)
        return False


def install_operators():
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

        def binary(self, other, operation=name):
            default = operation not in {"eq", "ne", "lt", "gt"}
            if operation == "eq":
                default = isinstance(other, Symbol)
            elif operation == "lt":
                default = not isinstance(other, Symbol)
            result = self._recorder.operation(operation, self, other)
            result._default = default
            return result

        def reflected(self, other, operation=name):
            return self._recorder.operation(operation, other, self)

        setattr(Symbol, f"__{name}__", binary)
        if name not in {"eq", "ne", "lt", "le", "gt", "ge"}:
            setattr(Symbol, f"__r{name}__", reflected)


install_operators()


class StableSet(set):
    """Give unordered constant sets deterministic iteration across processes."""

    def __iter__(self):
        """Return a canonical iteration order."""
        return iter(sorted(super().__iter__(), key=repr))


class Inputs(Symbol):
    """A synthetic mapping with optional concrete input keys."""

    def __init__(self, recorder, label, values=None):
        """Seed only inputs the caller explicitly wants to vary."""
        super().__init__(recorder, label)
        self._values = values or {}

    def __getitem__(self, key):
        """Return a concrete seed when supplied, otherwise a synthetic value."""
        if key in self._values:
            self._recorder.operation("getitem", self, key, self._values[key])
            return self._values[key]
        return super().__getitem__(key)


def external_bindings(modules):
    """Find external imports/instances, preserving functions in the target modules."""
    local_modules = {module.__name__ for module in modules}
    bindings = []
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


def pack_runs(runs):
    """Share identical operation prefixes without discarding execution details."""
    packed = []
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


def discover(invoke, bindings, max_runs=128):
    """Explore truth choices with fresh mocks and return all observed contracts.

    The caller supplies only a callable and mock locations. Explore each observed
    boolean site both ways, holding repeated tests at that site consistent within
    an execution. This is branch exploration, not exhaustive path enumeration.
    Queued paths are never silently dropped: exhausting the budget fails discovery.
    """
    pending = deque([((), None)])
    scheduled = set()
    observed = set()
    runs = []
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
            roots = {}
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
