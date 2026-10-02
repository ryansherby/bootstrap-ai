from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from src.utils.extract_called_names import extract_called_names


class FunctionRegistry:
    def __init__(self, roots: list[str | Path]):
        self.roots = [Path(r) for r in roots]
        self.path_by_name: dict[str, Path] = {}
        self.cache: dict[str, dict] = {}

    def scan(self) -> None:
        """Refresh discovered files."""
        discovered: dict[str, Path] = {}

        for root in self.roots:
            if not root.exists():
                continue

            for path in root.glob("*.py"):
                if path.name.startswith("_"):
                    continue

                fn_name = path.stem
                if fn_name in discovered:
                    raise ValueError(f"Duplicate function name found: {fn_name}")

                discovered[fn_name] = path

        self.path_by_name = discovered

    def load(self, fn_name: str) -> Callable:
        """Load or reload a function if its file changed."""
        if fn_name not in self.path_by_name:
            raise KeyError(f"Unknown function: {fn_name}")

        path = self.path_by_name[fn_name]
        mtime = path.stat().st_mtime
        cached = self.cache.get(fn_name)

        if cached and cached["mtime"] == mtime:
            return cached["callable"]

        module_name = f"_dynamic_fn_{fn_name}_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        fn = getattr(module, fn_name, None)
        if fn is None or not callable(fn):
            raise ValueError(
                f"Expected callable named '{fn_name}' in file {path}, but did not find one."
            )

        self.cache[fn_name] = {
            "mtime": mtime,
            "callable": fn,
            "path": path,
        }
        return fn

    def build_namespace(self, names: set[str]) -> dict[str, Callable]:
        ns: dict[str, Callable] = {}
        for name in names:
            if name in self.path_by_name:
                ns[name] = self.load(name)
        return ns


def exec_solve(
    source: str,
    registry: FunctionRegistry,
    fn_name: str | None = None,
    fn_args: list = [],
    fn_kwargs: dict = {},
) -> object:
    """
    Execute a trusted solve() source string against a dynamically built namespace.

    Args:
        source (str): The source code containing the fn_name function definition.
        registry (FunctionRegistry): The function registry to resolve dynamic functions.
        fn_name (str | None): The name of the function to execute. Defaults to "solve".
        *fn_args: Positional arguments to pass to the function.
        **fn_kwargs: Keyword arguments to pass to the function.

    Returns:
        object: The result of the function execution.
    """
    registry.scan()

    needed_names = extract_called_names(source)
    dynamic_fns = registry.build_namespace(needed_names)

    # Minimal builtins. Expand only as needed.
    safe_builtins = {
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "min": min,
        "max": max,
        "sum": sum,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sorted": sorted,
        "abs": abs,
        "any": any,
        "all": all,
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "warnings": __import__("warnings"),
    }

    namespace = {
        "__builtins__": safe_builtins,
        **dynamic_fns,
    }

    exec(source, namespace, namespace)

    solve = namespace.get(fn_name or "solve")
    if solve is None or not callable(solve):
        raise ValueError(f"Source did not define a callable {fn_name or 'solve'}()")

    return solve(*fn_args, **fn_kwargs)