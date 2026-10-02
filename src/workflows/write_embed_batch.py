

import ast
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from langgraph.runtime import Runtime
from torch import Tensor, tensor
import pandas as pd


from configs import CONFIG
from src.execution.executor import FunctionRegistry, exec_solve
from src.types import ComparisonResult, ComparisonRules, Data, FileMetadata
from src.model_definitions.base_model import BaseModel as NNBaseModel
from src.vectordb import VectorDB
from src.workflows.attributes import (
    FilesEmbeddedAttributes,
    FilesMetadataCreatedAttributes,
    FilesStoredAttributes,
    FilesWrittenAttributes,
)
from src.workflows.events import WriteEmbedBatchEvents
from src.workflows.facts import (
    SQLiteWorkflowFactRepository,
    WorkflowFact,
    workflow_fact_repository_factory,
)

from src.utils.render_function_stub import render_function_stub
from src.utils.compare_dataframes import compare_dataframes

from src.workflows.concurrency import limited_node
from src.utils.classify_file_mode import classify_file_mode_from_path
from utils.extract_function_contract import extract_function_contract



class WorkflowContext(BaseModel):

    embedding_model: NNBaseModel
    library_path: str
    function_registry: FunctionRegistry
    vdb: VectorDB
    run_id: str
    epoch: str
    workflow_name: str
    process_name: str = "write_embed_batch"
    batch_id: str = ""
    workflow_fact_repository: SQLiteWorkflowFactRepository = Field(
        default_factory=workflow_fact_repository_factory
    )

    


class WorkflowState(BaseModel):
    ids: list[str]
    names: list[str]
    texts: list[str]
    file_paths: list[str] = Field(default_factory=list)
    file_metadata: list[FileMetadata] = Field(default_factory=list)
    storage_write_count: int = 0
    storage_written_ids: list[str] = Field(default_factory=list)
    vector_db_write_count: int = 0
    vector_db_written_ids: list[str] = Field(default_factory=list)
    embedding_matrix: Tensor = Field(default_factory=lambda: tensor([]))
    type: Literal["update_index", "build_index"] = "update_index"
    
    
    
async def embed_texts(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """
    Embeds the texts in the state using the embedding model from the runtime context.
    """
    embedding_model = runtime.context.embedding_model
    embeddings = embedding_model.forward(process_name="embed", texts=state.texts)

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=WriteEmbedBatchEvents.FILES_EMBEDDED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                FilesEmbeddedAttributes(
                    data_ids=state.ids,
                    item_count=len(state.texts),
                    ids=state.ids,
                    text_lengths=[len(text) for text in state.texts],
                    embedding_shape=list(getattr(embeddings, "shape", [])),
                )
            ),
        )
    )
    
    return {"embedding_matrix": embeddings}
    
    
async def write_files(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """
    Writes the texts in the state to files in the library path.
    """
    library_path = runtime.context.library_path

    Path(library_path).mkdir(parents=True, exist_ok=True)
    file_paths = []
    
    for i, text in enumerate(state.texts):
        file_id = state.ids[i]
        name = state.names[i] if i < len(state.names) else f"function_{file_id}"
        filename = f"{name}.py"
        filepath = f"{library_path}/{filename}"
        
        with open(filepath, "w") as file:
            file.write(text)
        file_paths.append(filepath)

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=WriteEmbedBatchEvents.FILES_WRITTEN,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                FilesWrittenAttributes(
                    data_ids=state.ids,
                    item_count=len(state.texts),
                    ids=state.ids,
                    file_paths=file_paths,
                    filenames=[path.split("/")[-1] for path in file_paths],
                    storage_write_count=len(file_paths),
                    storage_written_ids=list(state.ids),
                )
            ),
        )
    )
    
    return {
        "file_paths": file_paths,
        "storage_write_count": len(file_paths),
        "storage_written_ids": list(state.ids),
    }


async def create_file_metadata(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """
    Creates file metadata for the texts in the state.
    """
    
    file_metadatas = []
    
    for i,path in enumerate(state.file_paths):
        source = Path(path).read_text()
        tree = ast.parse(source)
        function_nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if not function_nodes:
            raise ValueError(f"No top-level function definition found in {path}.")

        function_name = function_nodes[0].name
        namespace: dict[str, object] = {"pd": pd}
        exec(source, namespace, namespace)
        func = namespace.get(function_name)

        if func is None or not callable(func):
            raise ValueError(f"Function '{function_name}' was not defined by {path}.")

        contract = extract_function_contract(func)
        
        file_metadata = FileMetadata(
            id=state.ids[i],
            filename=path.split("/")[-1],
            filepath=path,
            filetype="py",
            funcname=contract["name"],
            func_docstring=contract["docstring"],
            func_is_async=contract["is_async"],
            func_signature=contract["signature"],
            func_parameters=contract["parameters"],
            embedding=state.embedding_matrix[i] if state.embedding_matrix.size(0) > i else [],
            mode=classify_file_mode_from_path(path),
        )
        
        file_metadatas.append(file_metadata)

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=WriteEmbedBatchEvents.FILES_METADATA_CREATED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                FilesMetadataCreatedAttributes(
                    data_ids=state.ids,
                    item_count=len(file_metadatas),
                    ids=state.ids,
                    file_paths=state.file_paths,
                    function_names=[metadata.funcname for metadata in file_metadatas],
                    file_metadata_ids=[metadata.id for metadata in file_metadatas],
                )
            ),
        )
    )
    
    return {"file_metadata": file_metadatas}
    
async def store_embeddings_in_vector_db(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """
    Stores the embeddings in the vector database.
    """
    vdb = runtime.context.vdb
    
    if state.type == "update_index":
        vdb.add(state.embedding_matrix, state.file_metadata)
    elif state.type == "build_index":
        vdb.build_index(state.embedding_matrix, state.file_metadata)
    else:
        raise ValueError(f"Unknown type: {state.type}")

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=WriteEmbedBatchEvents.FILES_STORED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                FilesStoredAttributes(
                    data_ids=state.ids,
                    type=state.type,
                    file_metadata_count=len(state.file_metadata),
                    vector_db_path=str(getattr(vdb, "path", "")),
                    file_metadata_ids=[metadata.id for metadata in state.file_metadata],
                    vector_db_write_count=len(state.file_metadata),
                    vector_db_written_ids=[metadata.id for metadata in state.file_metadata],
                )
            ),
        )
    )
    
    return {
        "vector_db_write_count": len(state.file_metadata),
        "vector_db_written_ids": [metadata.id for metadata in state.file_metadata],
    }


def write_embed_batch_workflow_builder():
    workflow_builder = StateGraph(WorkflowState)
    workflow_builder.add_node("embed_texts")
    workflow_builder.add_node("write_files")
    workflow_builder.add_node("create_file_metadata")
    workflow_builder.add_node("store_embeddings_in_vector_db")
    
    workflow_builder.add_edge(START,"embed_texts")
    workflow_builder.add_edge("embed_texts", "write_files")
    workflow_builder.add_edge("write_files", "create_file_metadata")
    workflow_builder.add_edge("create_file_metadata", "store_embeddings_in_vector_db")
    workflow_builder.add_edge("store_embeddings_in_vector_db", END)
    
    return workflow_builder.compile()