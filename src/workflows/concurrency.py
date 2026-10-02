import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ConcurrencyPolicy:
    max_workers: int = 16

    resource_limits: Mapping[str, int] = field(
        default_factory=dict
    )


class ConcurrencyRegistry:
    def __init__(
        self,
        policy: ConcurrencyPolicy,
    ):
        self.policy = policy

        self._semaphores = {
            "workers": asyncio.Semaphore(
                policy.max_workers
            ),
            **{
                name: asyncio.Semaphore(limit)
                for name, limit
                in policy.resource_limits.items()
            },
        }

    @asynccontextmanager
    async def acquire(
        self,
        name: str,
    ):
        semaphore = self._semaphores.get(name)

        if semaphore is None:
            raise KeyError(
                f"No concurrency resource "
                f"registered as {name!r}"
            )

        async with semaphore:
            yield
            
            
import asyncio
import inspect
from functools import wraps


def limited_node(resource: str | None = None):
    """
    Converts either a synchronous or async node into
    an async LangGraph node.

    If resource is provided, execution is protected
    by the corresponding shared semaphore.
    """

    def decorator(fn):

        @wraps(fn)
        async def wrapper(
            state,
            runtime,
            *args,
            **kwargs,
        ):
            async def execute():
                if inspect.iscoroutinefunction(fn):
                    return await fn(
                        state,
                        runtime,
                        *args,
                        **kwargs,
                    )

                return await asyncio.to_thread(
                    fn,
                    state,
                    runtime,
                    *args,
                    **kwargs,
                )

            if resource is None:
                return await execute()

            async with (
                runtime.context
                .concurrency
                .acquire(resource)
            ):
                return await execute()

        return wrapper

    return decorator