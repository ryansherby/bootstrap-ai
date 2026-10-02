

import ast
from dataclasses import asdict
from pathlib import Path
import re
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.workflows.events import BootstrapEvents
from src.workflows.attributes import (
    FunctionReference,
    MergedFunctionGeneratedAttributes,
    SimilarSolvesRetrievedAttributes,
)
from src.workflows.facts import (
    SQLiteWorkflowFactRepository,
    WorkflowFact,
    workflow_fact_repository_factory,
)
from src.workflows.cache import SharedCache

from langgraph.graph import START, END, StateGraph

from langgraph.runtime import Runtime
from torch import tensor
import pandas as pd


from configs import CONFIG
from src.execution.executor import FunctionRegistry, exec_solve
from src.types import ComparisonResult, ComparisonRules, Data, FileMetadata
from src.model_definitions.base_model import BaseModel as NNBaseModel
from src.vectordb import VectorDB
from src.utils.render_function_stub import render_function_stub

from src.workflows.concurrency import limited_node



class WorkflowContext(BaseModel):
    transformer_model: NNBaseModel
    function_registry: FunctionRegistry
    shared_cache: SharedCache
    vdb: VectorDB
    run_id: str
    epoch: int
    workflow_name: str = "train"
    process_name: str | None = None
    batch_id: str
    workflow_fact_repository: SQLiteWorkflowFactRepository = Field(
        default_factory=workflow_fact_repository_factory
    )
    
    
class WorkflowState(BaseModel):
    filemetadata: FileMetadata
    retrieval_threshold: float = CONFIG.vector_db.merge_retrieval_threshold
    top_k: int = CONFIG.vector_db.merge_retrieval_top_k
    similar_functions: list[FileMetadata] = Field(default_factory=list)
    merged_function: str|None = None


_MERGE_NO_OP_TOKENS = {
    "none",
    "null",
    "no_merge",
}


def _source_for_merge(metadata: FileMetadata) -> str:
    try:
        return Path(metadata.filepath).read_text(encoding="utf-8")
    except Exception:
        return None


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()

    if "```" not in cleaned:
        return cleaned

    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return cleaned.replace("```", "").strip()


def _normalize_merge_output(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    candidate = _strip_markdown_fences(value)
    if not candidate:
        return None

    if candidate.casefold() in _MERGE_NO_OP_TOKENS:
        return None

    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None

    has_function = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    )

    return candidate if has_function else None
    
    
async def get_similar_functions(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """
    Retrieves similar functions from the vector database based on the embedding of the function in the state.
    """
    vdb = runtime.context.vdb
    embedding = state.filemetadata.embedding
    results = vdb.search(embedding, top_k=state.top_k, search_type="basic", similarity_threshold=state.retrieval_threshold)
    
    similar_functions = [result["metadata"] for result in results[1:]]  # Skip the first result as it is the function itself

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=BootstrapEvents.SIMILAR_SOLVES_RETRIEVED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                SimilarSolvesRetrievedAttributes(
                    source_file_id=state.filemetadata.id,
                    source_function_name=state.filemetadata.funcname,
                    source_filepath=state.filemetadata.filepath,
                    retrieval_threshold=state.retrieval_threshold,
                    top_k=state.top_k,
                    retrieved_count=len(similar_functions),
                    retrieved_functions=[
                        FunctionReference(
                            id=function_metadata.id,
                            funcname=function_metadata.funcname,
                            filepath=function_metadata.filepath,
                        )
                        for function_metadata in similar_functions
                    ],
                )
            ),
        )
    )
    
    return {"similar_functions": similar_functions}

@limited_node("transformer")
async def merge_functions(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """
    Merges the function in the state with the similar functions retrieved from the vector database.
    """
    merged_function = None
    selected_metadata = None
    selected_cache_hit = False
    source_function = _source_for_merge(state.filemetadata)
    
    for md in state.similar_functions:
        cache_id = ":".join(sorted([state.filemetadata.id, md.id]))

        target_function = _source_for_merge(md)

        cache_result = runtime.context.shared_cache.get_or_compute(
            cache_id,
            lambda source_function=source_function, target_function=target_function: runtime.context.transformer_model.forward(
                process_name="merge",
                code_snippets=[source_function, target_function],
            ),
        )

        # A cache hit means this pair was already merged previously.
        # Skip it to avoid scheduling duplicate merge outcomes.
        if cache_result.cache_hit:
            continue

        normalized_merge = _normalize_merge_output(cache_result.value)
        if normalized_merge is None:
            continue

        merged_function = normalized_merge
        selected_metadata = md
        selected_cache_hit = cache_result.cache_hit
        
        break

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=BootstrapEvents.MERGED_FUNCTION_GENERATED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                MergedFunctionGeneratedAttributes(
                    source_file_id=state.filemetadata.id,
                    source_function_name=state.filemetadata.funcname,
                    considered_count=len(state.similar_functions),
                    considered_functions=[
                        FunctionReference(
                            id=function_metadata.id,
                            funcname=function_metadata.funcname,
                            filepath=function_metadata.filepath,
                        )
                        for function_metadata in state.similar_functions
                    ],
                    selected_function=(
                        None
                        if selected_metadata is None
                        else FunctionReference(
                            id=selected_metadata.id,
                            funcname=selected_metadata.funcname,
                            filepath=selected_metadata.filepath,
                        )
                    ),
                    cache_hit=selected_cache_hit,
                    merged_function_present=merged_function is not None,
                    merged_function_length=len(merged_function) if merged_function else 0,
                )
            ),
        )
    )

    return {"merged_function": merged_function}


def build_bootstrap_worker():
    workflow_builder = StateGraph(WorkflowState)
    workflow_builder.add_node("get_similar_functions")
    workflow_builder.add_node("merge_functions")
    
    workflow_builder.add_edge(START, "get_similar_functions")
    workflow_builder.add_edge("get_similar_functions", "merge_functions")
    workflow_builder.add_edge("merge_functions", END)
    
    return workflow_builder.compile()
    