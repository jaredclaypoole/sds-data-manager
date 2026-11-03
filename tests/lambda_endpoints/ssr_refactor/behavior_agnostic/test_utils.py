"""Check discovery independently of SSR's implementation and expected behavior."""

import json
from types import ModuleType

import pytest

from tests.lambda_endpoints.ssr_refactor.behavior_agnostic.utils import (
    OPERATION_FIELDS,
    ExplorationLimitError,
    Inputs,
    Recorder,
    Symbol,
    UnsupportedValueError,
    discover,
    expand_runs,
    external_bindings,
    pack_runs,
    readable_runs,
)


def unpack(runs):
    """Reconstruct complete operation lists from shared prefixes."""
    expanded = []
    for run in runs:
        parent, length = run["prefix"]
        prefix = expanded[parent]["steps"][:length] if length else []
        expanded.append({key: value for key, value in run.items() if key != "prefix"})
        expanded[-1]["steps"] = prefix + run["steps"]
    return expanded


def test_discovery_explores_both_branches_and_records_errors():
    """Discover nested branches and exception outcomes without a return fixture."""

    def invoke(recorder):
        dependency = Symbol(recorder, "dependency")
        if dependency.ready():
            if dependency.valid():
                return dependency.process("payload")
            raise ValueError("invalid")
        return dependency.skip()

    runs = unpack(discover(invoke, []))
    assert len(runs) == 3
    assert [run["choices"] for run in runs] == [[True, True], [False], [True, False]]
    assert runs[-1]["error"] == {"exception": "ValueError", "message": "invalid"}
    assert any(step[0] == "call" for step in runs[0]["steps"])


def test_call_arguments_are_frozen_at_call_time():
    """Mutating a container after a call must not alter its recorded arguments."""
    recorder = Recorder()
    payload = {"items": [1]}
    Symbol(recorder, "dependency")(payload)
    payload["items"].append(2)
    assert recorder.steps[0][1] == [{"ref": "dependency"}, {"items": [1]}]


def test_dependency_keywords_do_not_collide_with_recorder_parameters():
    """Record all dependency keywords, including names used by the recorder."""
    recorder = Recorder()
    Symbol(recorder, "dependency")(default=False, operation="read", self="value")
    assert recorder.steps[0][2] == {
        "default": False,
        "operation": "read",
        "self": "value",
    }


def test_bindings_preserve_helpers_aliases_and_restore_state():
    """Patch external objects locally and leave both modules' own code runnable."""
    first, second = ModuleType("first"), ModuleType("second")

    def helper():
        return "local"

    def external():
        pytest.fail("A real dependency was called")

    helper.__module__ = first.__name__
    first.helper = helper
    second.helper = helper
    first.external = external
    second.external = external
    first.options = {"z", "a"}
    original_options = first.options
    first.target_module = second
    bindings = external_bindings((first, second))
    assert bindings == [(first, "external"), (second, "external")]

    def invoke(recorder):
        assert first.helper() == "local"
        assert second.helper is helper
        assert first.external is second.external
        assert list(first.options) == ["a", "z"]
        return first.external()

    discover(invoke, bindings)
    assert first.external is second.external is external
    assert first.options is original_options


def test_context_exit_propagates_exceptions_without_tracebacks():
    """An external context manager must not swallow an application exception."""

    def invoke(recorder):
        with Symbol(recorder, "context"):
            raise LookupError("failed")

    run = unpack(discover(invoke, []))[0]
    assert run["error"] == {"exception": "LookupError", "message": "failed"}
    assert [step[0] for step in run["steps"]] == ["enter", "exit"]
    assert run["steps"][-1][1][-1] == run["error"]


def test_budget_exhaustion_is_not_captured_as_an_application_error():
    """An incomplete exploration must fail the test, even in update mode."""
    with pytest.raises(ExplorationLimitError, match="Maximum executions"):
        discover(lambda recorder: bool(Symbol(recorder, "value")), [], max_runs=1)

    recorder = Recorder(max_steps=1)
    value = Symbol(recorder, "value")
    value()
    with pytest.raises(ExplorationLimitError, match="Maximum operations"):
        value()


def test_unsupported_arguments_fail_discovery_instead_of_becoming_baselines():
    """A serialization gap is a harness failure, not a discovered code outcome."""
    with pytest.raises(UnsupportedValueError, match="Unsupported discovery argument"):
        discover(lambda recorder: Symbol(recorder, "dependency")(object()), [])


def test_packing_is_lossless():
    """Sharing a prefix retains every operation and each outcome."""
    runs = [
        {"steps": [["a"], ["b"], ["c"]], "error": None},
        {"steps": [["a"], ["b"], ["d"]], "error": "failure"},
        {"steps": [["a"], ["b"], ["c"], ["e"]], "error": None},
    ]
    packed = pack_runs(runs)
    assert packed[1]["prefix"] == [0, 2]
    assert packed[2]["prefix"] == [0, 3]
    assert unpack(packed) == runs


def test_only_seeded_inputs_are_concrete():
    """Unspecified event and environment keys remain synthetic."""
    recorder = Recorder()
    inputs = Inputs(recorder, "environment", {"instrument": "example"})
    assert inputs["instrument"] == "example"
    assert isinstance(inputs["unknown"], Symbol)


def retained_operands(runs):
    """Project retained operands, excluding the deliberately omitted callable ref."""
    raw = []
    for run in runs:
        steps = []
        for step in run["steps"]:
            if "call" in step:
                operation, args = "call", step["args"]
            elif "op" in step:
                operation, args = step["op"], [step["left"], step["right"]]
            else:
                operation = next(name for name in OPERATION_FIELDS if name in step)
                args = [
                    step[field]
                    for field in OPERATION_FIELDS[operation]
                    if field in step
                ]
            steps.append([operation, args, step.get("kwargs", {}), step["result"]])
        raw.append({**run, "steps": steps})
    return raw


def test_readable_runs_preserve_records_and_resolve_chained_calls():
    """Call records retain display names, explicit arguments, and results."""
    module = ModuleType("example")
    module.service = ModuleType("service")
    bindings = [(module, "service")]

    def invoke(recorder):
        client = module.service.client("db")
        for path in client.paths():
            path.stat()
        if client.ready():
            with client.session() as session:
                session.write(value=client.read())

    packed = discover(invoke, bindings)
    original = json.dumps(packed)
    readable = readable_runs(packed)
    assert json.dumps(packed) == original
    steps = [step for run in readable for step in run["steps"]]
    calls = [step for step in steps if "call" in step]
    assert all(set(step) == {"call", "args", "kwargs", "result"} for step in calls)
    assert any(step["call"] == "service.client.ready" for step in calls)
    assert any(step["call"] == "service.client.paths.stat" for step in calls)
    assert any(
        step["call"] == "service.client.session.__enter__.write" for step in calls
    )
    assert all("prefix" not in run and "calls" not in run for run in readable)
    expected = expand_runs(packed)
    for run, rendered in zip(expected, readable, strict=True):
        retained = {step["result"]["ref"] for step in rendered["steps"]}
        run["steps"] = [
            [operation, args[1:] if operation == "call" else args, kwargs, result]
            for operation, args, kwargs, result in run["steps"]
            if operation != "getattr" or result["ref"] in retained
        ]
    assert retained_operands(readable) == expected


def test_callable_lookups_are_omitted_only_when_no_other_use_exists():
    """Keep references passed as arguments, used as objects, or returned."""

    def invoke(recorder):
        dependency = Symbol(recorder, "dependency")
        dependency.only_called()
        shared = dependency.shared
        shared()
        dependency.accept(shared, nested={"callback": shared})
        _ = shared.attribute
        returned = dependency.returned
        returned()
        _ = dependency.unused
        return returned

    run = readable_runs(discover(invoke, []))[0]
    attributes = [step["getattr"] for step in run["steps"] if "getattr" in step]
    assert attributes == ["shared", "attribute", "returned", "unused"]
    assert any(step.get("call") == "dependency.only_called" for step in run["steps"])
    defined = {step["result"]["ref"] for step in run["steps"]}
    assert run["result"]["ref"] in defined


def test_root_references_use_local_names_without_changing_literals():
    """External imports and module variables need no target-module prefix."""
    prefix = "example.processing."

    def invoke(recorder):
        data_access = Symbol(recorder, prefix + "imap_data_access")
        data_access.config["API_KEY"] = "key"
        Symbol(recorder, "shutil").rmtree(
            Symbol(recorder, prefix + "DOWNLOAD_DIR"), prefix + "DOWNLOAD_DIR"
        )

    run = readable_runs(discover(invoke, []))[0]
    config = next(step for step in run["steps"] if step.get("getattr") == "config")
    assert config["object"] == {"ref": "imap_data_access"}
    cleanup = next(step for step in run["steps"] if step.get("call") == "shutil.rmtree")
    assert cleanup["args"] == [{"ref": "DOWNLOAD_DIR"}, prefix + "DOWNLOAD_DIR"]


def test_shortened_roots_remain_distinct_and_do_not_collide_with_results():
    """Matching names in different modules retain enough scope to distinguish them."""

    def invoke(recorder):
        return Symbol(recorder, "consume")(
            Symbol(recorder, "example.first.value"),
            Symbol(recorder, "example.second.value"),
            Symbol(recorder, "example.v0"),
        )

    call = readable_runs(discover(invoke, []))[0]["steps"][0]
    assert call["args"] == [
        {"ref": "first.value"},
        {"ref": "second.value"},
        {"ref": "example.v0"},
    ]
    assert call["result"] == {"ref": "v0"}


def test_only_module_local_exception_types_are_unqualified():
    """Local classes lose their module prefix; imported classes retain theirs."""
    module = ModuleType("example.processing")
    module.LocalError = type(
        "LocalError", (Exception,), {"__module__": module.__name__}
    )
    module.ImportedError = ValueError
    for exception, expected in (
        (module.LocalError, "LocalError"),
        (ValueError, "builtins.ValueError"),
    ):

        def invoke(recorder, exception=exception):
            with Symbol(recorder, "context"):
                raise exception("failed")

        run = readable_runs(discover(invoke, []), (module,))[0]
        exit_step = next(step for step in run["steps"] if "exit" in step)
        assert exit_step["exception_type"] == {"type": expected}
