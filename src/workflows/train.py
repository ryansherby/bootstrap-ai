


from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.execution.executor import FunctionRegistry
from src.model_definitions.base_model import BaseModel as NNBaseModel
from src.types import Data, FileMetadata
from src.vectordb import VectorDB
from src.workflows.bootstrap_worker import build_bootstrap_worker, WorkflowContext as BootstrapWorkflowContext
from src.workflows.concurrency import ConcurrencyRegistry
from src.workflows.inference_worker import build_inference_worker, WorkflowContext as InferenceWorkflowContext
from src.workflows.facts import SQLiteWorkflowFactRepository, workflow_fact_repository_factory
from src.workflows.scheduler import create_scheduler
from src.workflows.test_worker import build_test_worker, WorkflowContext as TestWorkflowContext
from src.workflows.write_embed_batch import (
    WorkflowContext as WriteEmbedBatchWorkflowContext,
    WorkflowState as WriteEmbedBatchWorkflowState,
    write_embed_batch_workflow_builder,
)


class WorkflowContext(BaseModel):
    transformer_model: NNBaseModel
    embedding_model: NNBaseModel
    shared_cache: Any
    concurrency: ConcurrencyRegistry
    perm_vdb: VectorDB
    temp_vdb: VectorDB
    perm_library_path: str
    temp_library_path: str
    workflow_fact_repository: SQLiteWorkflowFactRepository = Field(
        default_factory=workflow_fact_repository_factory
    )
    run_id: str
    epoch: str
    batch_id: str
    workflow_name: str = "train"
    process_name: str = "train"


class WorkflowState(BaseModel):
    dataset: list[Data] = Field(default_factory=list)
    inference_results: list[Any] = Field(default_factory=list)
    solve_write_embed_result: dict[str, Any] = Field(default_factory=dict)
    bootstrap_results: list[Any] = Field(default_factory=list)
    tested_function_texts: list[str] = Field(default_factory=list)
    test_results: list[Any] = Field(default_factory=list)
    merge_write_embed_result: dict[str, Any] = Field(default_factory=dict)


_MERGE_NO_OP_TOKENS = {
    "none",
    "null",
    "no_merge",
}


def _build_function_registry(root: str) -> FunctionRegistry:
    return FunctionRegistry([root])


def _build_batch_path(base_path: str, run_id: str, epoch: str, batch_id: str) -> Path:
    return Path(base_path) / f"{run_id}:{epoch}:{batch_id}"


def _extract_function_name(code: str, fallback_id: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"function_{fallback_id}"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name

    return f"function_{fallback_id}"


def _extract_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _comparison_matches(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("matches"))

    return bool(getattr(result, "matches", False))


def _result_text(result: Any, key: str) -> str:
    if isinstance(result, dict):
        value = result.get(key, "")
    else:
        value = getattr(result, key, "")

    return value if isinstance(value, str) else ""


def _result_metadata_list(result: Any) -> list[FileMetadata]:
    if isinstance(result, dict):
        metadata = result.get("file_metadata", [])
    else:
        metadata = getattr(result, "file_metadata", [])

    return list(metadata or [])


def _normalize_merged_function_text(result: Any) -> str:
    candidate = _result_text(result, "merged_function").strip()
    if not candidate:
        return ""

    if candidate.casefold() in _MERGE_NO_OP_TOKENS:
        return ""

    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return ""

    has_function = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    )

    return candidate if has_function else ""


def build_train_workflow():
    inference_worker = build_inference_worker()
    bootstrap_worker = build_bootstrap_worker()
    test_worker = build_test_worker()
    write_embed_batch_workflow = write_embed_batch_workflow_builder()

    inference_scheduler = create_scheduler(
        inference_worker,
        context_schema=InferenceWorkflowContext,
        worker_context_adapter=lambda context: InferenceWorkflowContext(
            **context.dict(exclude={"process_name", "temp_solves_vdb_path", "temp_library_path", "perm_library_path"}),
            function_registry=_build_function_registry(context.perm_library_path),
            process_name="inference",
        ),
    )

    bootstrap_scheduler = create_scheduler(
        bootstrap_worker,
        context_schema=BootstrapWorkflowContext,
        worker_context_adapter=lambda context: BootstrapWorkflowContext(
            **context.dict(exclude={"process_name", "temp_solves_vdb_path", "temp_library_path", "perm_library_path"}),
            function_registry=_build_function_registry(context.temp_library_path),
            vdb=context.temp_vdb,
            process_name="bootstrap",
        ),
    )

    test_scheduler = create_scheduler(
        test_worker,
        context_schema=TestWorkflowContext,
        worker_context_adapter=lambda context: TestWorkflowContext(
            **context.dict(exclude={"process_name", "temp_solves_vdb_path", "temp_library_path", "perm_library_path"}),
            function_registry=_build_function_registry(context.perm_library_path),
            process_name="test",
        ),
    )

    async def dispatch_inference_workers(
        state: WorkflowState,
        runtime: Runtime[WorkflowContext],
        config: RunnableConfig,
    ):
        scheduler_state = {
            "inputs": [{"data": data_item, "execute": True, "compare": True} for data_item in state.dataset],
            "pending_results": [],
            "results": [],
        }

        scheduler_result = await inference_scheduler.ainvoke(
            scheduler_state,
            config=config,
            context=runtime.context,
        )

        return {"inference_results": scheduler_result["results"]}

    async def solve_write_embed(
        state: WorkflowState,
        runtime: Runtime[WorkflowContext],
        config: RunnableConfig,
    ):
        selected_items: list[tuple[Data, str]] = []

        for data_item, result in zip(state.dataset, state.inference_results):
            if not _comparison_matches(result):
                continue

            code = _result_text(result, "code")
            if not code:
                continue

            selected_items.append((data_item, code))

        if not selected_items:
            return {"solve_write_embed_result": {"file_metadata": []}}

        batch_path = _build_batch_path(
            runtime.context.temp_library_path,
            runtime.context.run_id,
            runtime.context.epoch,
            runtime.context.batch_id,
        )

        texts = [code for _, code in selected_items]
        ids = [_extract_code_hash(code) for code in texts]
        write_embed_state = WriteEmbedBatchWorkflowState(
            ids=ids,
            names=[],
            texts=texts,
            type="build_index",
        )
        write_embed_context = WriteEmbedBatchWorkflowContext(
            embedding_model=runtime.context.embedding_model,
            library_path=str(batch_path),
            function_registry=_build_function_registry(str(batch_path)),
            vdb=runtime.context.temp_vdb,
            run_id=runtime.context.run_id,
            epoch=runtime.context.epoch,
            workflow_name=runtime.context.workflow_name,
            process_name="solve_write_embed",
            batch_id=runtime.context.batch_id,
            workflow_fact_repository=runtime.context.workflow_fact_repository,
        )

        write_embed_result = await write_embed_batch_workflow.ainvoke(
            write_embed_state,
            config=config,
            context=write_embed_context,
        )

        return {"solve_write_embed_result": write_embed_result}

    async def dispatch_bootstrap_workers(
        state: WorkflowState,
        runtime: Runtime[WorkflowContext],
        config: RunnableConfig,
    ):
        file_metadata = _result_metadata_list(state.solve_write_embed_result)

        if not file_metadata:
            return {"bootstrap_results": []}

        scheduler_state = {
            "inputs": [{"filemetadata": metadata} for metadata in file_metadata],
            "pending_results": [],
            "results": [],
        }

        scheduler_result = await bootstrap_scheduler.ainvoke(
            scheduler_state,
            config=config,
            context=runtime.context,
        )

        return {"bootstrap_results": scheduler_result["results"]}

    async def dispatch_test_workers(
        state: WorkflowState,
        runtime: Runtime[WorkflowContext],
        config: RunnableConfig,
    ):
        function_texts = [
            normalized
            for normalized in (
                _normalize_merged_function_text(result)
                for result in state.bootstrap_results
            )
            if normalized
        ]

        if not function_texts:
            return {
                "tested_function_texts": [],
                "test_results": [],
            }

        scheduler_state = {
            "inputs": [{"function_text": function_text} for function_text in function_texts],
            "pending_results": [],
            "results": [],
        }

        scheduler_result = await test_scheduler.ainvoke(
            scheduler_state,
            config=config,
            context=runtime.context,
        )

        return {
            "tested_function_texts": function_texts,
            "test_results": scheduler_result["results"],
        }

    async def merge_write_embed_batch(
        state: WorkflowState,
        runtime: Runtime[WorkflowContext],
        config: RunnableConfig,
    ):
        passed_codes: list[str] = []

        for code, result in zip(state.tested_function_texts, state.test_results):
            if bool(getattr(result, "passed", False) if not isinstance(result, dict) else result.get("passed", False)):
                passed_codes.append(code)

        if not passed_codes:
            return {"merge_write_embed_result": {"file_metadata": []}}

        merge_library_path = Path(runtime.context.perm_library_path) / "gen"
        merge_registry = _build_function_registry(str(merge_library_path))
        merge_vdb = runtime.context.perm_vdb

        ids = [_extract_code_hash(code) for code in passed_codes]
        merge_write_state = WriteEmbedBatchWorkflowState(
            ids=ids,
            names=[],
            texts=passed_codes,
            type="update_index",
        )
        merge_write_context = WriteEmbedBatchWorkflowContext(
            embedding_model=runtime.context.embedding_model,
            library_path=str(merge_library_path),
            function_registry=merge_registry,
            vdb=merge_vdb,
            run_id=runtime.context.run_id,
            epoch=runtime.context.epoch,
            workflow_name=runtime.context.workflow_name,
            process_name="merge_write_embed_batch",
            batch_id=runtime.context.batch_id,
            workflow_fact_repository=runtime.context.workflow_fact_repository,
        )

        merge_write_result = await write_embed_batch_workflow.ainvoke(
            merge_write_state,
            config=config,
            context=merge_write_context,
        )

        return {"merge_write_embed_result": merge_write_result}

    builder = StateGraph(WorkflowState, context_schema=WorkflowContext)
    builder.add_node("dispatch_inference_workers", dispatch_inference_workers)
    builder.add_node("solve_write_embed", solve_write_embed)
    builder.add_node("dispatch_bootstrap_workers", dispatch_bootstrap_workers)
    builder.add_node("dispatch_test_workers", dispatch_test_workers)
    builder.add_node("merge_write_embed_batch", merge_write_embed_batch)

    builder.add_edge(START, "dispatch_inference_workers")
    builder.add_edge("dispatch_inference_workers", "solve_write_embed")
    builder.add_edge("solve_write_embed", "dispatch_bootstrap_workers")
    builder.add_edge("dispatch_bootstrap_workers", "dispatch_test_workers")
    builder.add_edge("dispatch_test_workers", "merge_write_embed_batch")
    builder.add_edge("merge_write_embed_batch", END)

    return builder.compile()