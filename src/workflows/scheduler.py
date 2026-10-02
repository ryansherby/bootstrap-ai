import operator

from typing import (
    Annotated,
    Any,
    Callable,
    TypedDict,
)

from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
)

from langgraph.graph import (
    START,
    END,
    StateGraph,
)

from langgraph.runtime import Runtime
from langgraph.types import Send


class ScheduledItem(TypedDict):
    index: int
    value: Any


class ScheduledResult(TypedDict):
    index: int
    value: Any


class SchedulerState(TypedDict):
    inputs: list[Any]

    pending_results: Annotated[
        list[ScheduledResult],
        operator.add,
    ]

    results: list[Any]


def create_scheduler(
    worker: Runnable,
    *,
    context_schema,
    worker_context_adapter: Callable[
        [Any], Any
    ] = lambda context: context,
):
    """
    Create a generic map/fan-out scheduler around
    any async-compatible LangChain/LangGraph Runnable.
    """

    def fan_out(state: SchedulerState):

        return [
            Send(
                "execute_worker",
                {
                    "index": index,
                    "value": value,
                },
            )
            for index, value
            in enumerate(state["inputs"])
        ]

    async def execute_worker(
        state: ScheduledItem,
        runtime: Runtime,
        config: RunnableConfig,
    ):
        concurrency = (
            runtime.context.concurrency
        )

        async with concurrency.acquire(
            "workers"
        ):
            result = await worker.ainvoke(
                state["value"],
                config=config,
                context=worker_context_adapter(
                    runtime.context
                ),
            )

        return {
            "pending_results": [
                {
                    "index": state["index"],
                    "value": result,
                }
            ]
        }

    def finalize(
        state: SchedulerState,
    ):
        ordered = sorted(
            state["pending_results"],
            key=lambda x: x["index"],
        )

        return {
            "results": [
                x["value"]
                for x in ordered
            ]
        }

    builder = StateGraph(
        SchedulerState,
        context_schema=context_schema,
    )

    builder.add_node(
        "execute_worker",
        execute_worker,
        input_schema=ScheduledItem,
    )

    builder.add_node(
        "finalize",
        finalize,
    )

    builder.add_conditional_edges(
        START,
        fan_out,
        ["execute_worker"],
    )

    builder.add_edge(
        "execute_worker",
        "finalize",
    )

    builder.add_edge(
        "finalize",
        END,
    )

    return builder.compile()