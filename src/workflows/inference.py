from __future__ import annotations

from typing import Any
from src.types import ComparisonResult

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.types import Data, FileMetadata
from src.workflows.concurrency import ConcurrencyRegistry
from src.workflows.inference_worker import (
	WorkflowContext as WorkerContext,
	build_inference_worker,
)
from src.workflows.scheduler import create_scheduler


class WorkflowContext(WorkerContext):
	concurrency: ConcurrencyRegistry


class WorkflowState(BaseModel):
	library_path: str | None = None
	data: list[Data] = Field(default_factory=list)
	file_metadata: list[FileMetadata] = Field(default_factory=list)
	execute: bool = True
	compare: bool = True
	execution_results: list[Any] = Field(default_factory=list)
	comparison_results: list[ComparisonResult] = Field(default_factory=list)
	code_results: list[str] = Field(default_factory=list)

def build_inference_workflow():
	inference_worker = build_inference_worker()
	scheduler = create_scheduler(
		inference_worker,
		context_schema=WorkflowContext,
		worker_context_adapter=lambda context: WorkerContext(
			**context.dict(exclude={"process_name", "concurrency"}),
			process_name=context.workflow_name,
		),
	)

	async def dispatch_inference_workers(
		state: WorkflowState,
		runtime: Runtime[WorkflowContext],
		config: RunnableConfig,
	):
		file_metadata = state.file_metadata

		scheduler_state = {
			"inputs": [
				{
					"data": data_item,
					"execute": state.execute,
					"compare": state.compare,
					"functions": file_metadata,
				}
				for data_item in state.data
			],
			"pending_results": [],
			"results": [],
		}

		scheduler_result = await scheduler.ainvoke(
			scheduler_state,
			config=config,
			context=runtime.context,
		)

		worker_results = scheduler_result["results"]

		return {
			"file_metadata": file_metadata,
			"execution_results": [
				result.get("execution_result")
				for result in worker_results
			],
			"comparison_results": [
				result.get("comparison_result")
				for result in worker_results
			],
			"code_results": [
				result.get("code", "")
				for result in worker_results
			],
		}

	builder = StateGraph(
		WorkflowState,
		context_schema=WorkflowContext,
	)

	builder.add_node(
		"dispatch_inference_workers",
		dispatch_inference_workers,
	)

	builder.add_edge(
		START,
		"dispatch_inference_workers",
	)
	builder.add_edge(
		"dispatch_inference_workers",
		END,
	)

	return builder.compile()
