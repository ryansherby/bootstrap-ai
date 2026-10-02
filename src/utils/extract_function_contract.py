import inspect
import hashlib
import textwrap
from collections.abc import Callable
from typing import Any


def extract_function_contract(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Extracts the public contract of a function, including its name, signature, parameters, return annotation, docstring, whether it is asynchronous, and its source code.
    """
    if not callable(func):
        raise TypeError(f"Expected a callable, got {type(func).__name__}")

    try:
        signature = inspect.signature(
            func,
            follow_wrapped=True,
            eval_str=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not inspect the signature of {func!r}"
        ) from exc

    parameters: list[dict[str, Any]] = []

    for parameter in signature.parameters.values():
        parameters.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "annotation": (
                    None
                    if parameter.annotation is inspect.Parameter.empty
                    else inspect.formatannotation(parameter.annotation)
                ),
                "required": parameter.default is inspect.Parameter.empty,
                "default": (
                    None
                    if parameter.default is inspect.Parameter.empty
                    else repr(parameter.default)
                ),
            }
        )

    return_annotation = (
        None
        if signature.return_annotation is inspect.Signature.empty
        else inspect.formatannotation(signature.return_annotation)
    )

    return {
        "name": getattr(func, "__name__", type(func).__name__),
        "qualified_name": getattr(func, "__qualname__", None),
        "module": getattr(func, "__module__", None),
        "signature": str(signature),
        "parameters": parameters,
        "return_annotation": return_annotation,
        "docstring": inspect.getdoc(func) or "",
        "is_async": inspect.iscoroutinefunction(func),
        "source": textwrap.dedent(inspect.getsource(func)) if inspect.isfunction(func) else None,
        "hash": (
            hashlib.sha256(inspect.getsource(func).encode("utf-8")).hexdigest()
            if inspect.isfunction(func)
            else None
        ),
    }