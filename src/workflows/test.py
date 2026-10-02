from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.workflows.concurrency import ConcurrencyRegistry
from src.workflows.scheduler import create_scheduler
from src.workflows.test_worker import (
	build_test_worker,
	WorkflowContext as WorkerContext,
)

class WorkflowContext(WorkerContext):
	concurrency: ConcurrencyRegistry

class WorkflowState(BaseModel):
	function_texts: list[str] = Field(default_factory=list)
	test_results: list[Any] = Field(default_factory=list)


def build_test_workflow(
):
	"""
	Build a scheduler-driven test workflow.

	Expected inbound state keys:
	- function_texts: list[str]
	"""

	test_worker = build_test_worker()
	scheduler = create_scheduler(
		test_worker,
		context_schema=WorkflowContext,
		worker_context_adapter=lambda context: WorkerContext(
			**context.dict(exclude={"process_name", "concurrency"}),
			process_name=context.workflow_name,
		),
	)

	async def dispatch_test_workers(
		state: WorkflowState,
		runtime: Runtime[WorkflowContext],
		config: RunnableConfig,
	):
		function_texts = state.function_texts

		scheduler_state = {
			"inputs": [
				{
					"function_text": function_text,
				}
				for function_text in function_texts
			],
			"pending_results": [],
			"results": [],
		}

		scheduler_result = await scheduler.ainvoke(
			scheduler_state,
			config=config,
			context=runtime.context,
		)

		return {
			"function_texts": function_texts,
			"test_results": scheduler_result["results"],
		}

	builder = StateGraph(
		WorkflowState,
		context_schema=WorkflowContext,
	)

	builder.add_node(
		"dispatch_test_workers",
		dispatch_test_workers,
	)

	builder.add_edge(
		START,
		"dispatch_test_workers",
	)
	builder.add_edge(
		"dispatch_test_workers",
		END,
	)

	return builder.compile()
