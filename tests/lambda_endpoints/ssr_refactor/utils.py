"""Shared serialization and recording helpers for SSR refactoring tests."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import DEFAULT, Mock

import numpy as np
import xarray as xr


def format_trace(value, indent=0, width=100):
    """Render JSON with two-space indentation and short containers inline."""
    compact = json.dumps(value, sort_keys=True)
    if indent + len(compact) <= width:
        return compact
    if isinstance(value, dict) and value:
        entries = []
        for index, key in enumerate(sorted(value)):
            prefix = json.dumps(key) + ": "
            comma = int(index < len(value) - 1)
            entries.append(
                " " * (indent + 2)
                + prefix
                + format_trace(value[key], indent + 2, 100 - len(prefix) - comma)
            )
        return "{\n" + ",\n".join(entries) + "\n" + " " * indent + "}"
    if isinstance(value, list) and value:
        entries = [
            " " * (indent + 2)
            + format_trace(item, indent + 2, 100 - int(index < len(value) - 1))
            for index, item in enumerate(value)
        ]
        return "[\n" + ",\n".join(entries) + "\n" + " " * indent + "]"
    return compact


def freeze(value):  # noqa: PLR0911, PLR0912
    """Copy arguments at call time, before callers can mutate them."""
    if isinstance(value, xr.Dataset):
        return {"dataset": freeze(value.to_dict())}
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(k): sorted(v.split(",")) if k == "file_types" else freeze(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [freeze(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(freeze(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, BaseException):
        return {"exception": type(value).__name__, "message": str(value)}
    if isinstance(value, type):
        return value.__name__
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "filename_list"):
        return {"filenames": freeze(value.filename_list)}
    if hasattr(value, "name"):
        return {"name": value.name}
    raise TypeError(f"Add an explicit trace representation for {type(value)}")


class Boundary:
    """Provide deterministic external dependencies sharing one ordered trace."""

    def __init__(self, monkeypatch):
        """Start a fresh trace for one invocation."""
        self.monkeypatch = monkeypatch
        self.calls = []

    def function(self, name, result=None, effect=None):
        """Record each call before returning a value or invoking an effect."""

        def invoke(*args, **kwargs):
            self.calls.append([name, freeze(args), freeze(kwargs)])
            if effect is not None:
                return effect(*args, **kwargs)
            return DEFAULT

        return Mock(return_value=result, side_effect=invoke)

    def context_exit(self, name):
        """Record exception type/message without the nondeterministic traceback."""

        def exit_context(exc_type, exc, tb):
            self.calls.append([name, freeze([exc_type, exc]), {}])
            return False

        return Mock(side_effect=exit_context)

    def patch(self, owner, attribute, result=None, effect=None, name=None):
        """Replace one dependency with a recording mock."""
        mock = self.function(name or attribute, result, effect)
        self.monkeypatch.setattr(owner, attribute, mock)
        return mock
