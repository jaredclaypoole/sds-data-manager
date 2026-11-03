"""Check discovery independently of SSR's implementation and expected behavior."""

from types import ModuleType

import pytest

from tests.lambda_endpoints.ssr_refactor.behavior_agnostic.utils import (
    ExplorationLimitError,
    Inputs,
    Recorder,
    Symbol,
    UnsupportedValueError,
    discover,
    external_bindings,
    pack_runs,
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
