

import hashlib
from dataclasses import asdict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from torch import tensor

from langgraph.graph import StateGraph, START,END
from langgraph.runtime import Runtime

from src.types import FileMetadata

from src.model_definitions.base_model import BaseModel as NNBaseModel
from configs import CONFIG
from src.execution.executor import FunctionRegistry
from src.workflows.attributes import (
    EmbeddingsStoredAttributes,
    FileMetadataConstructedAttributes,
    FunctionCodeEmbeddedAttributes,
)
from src.workflows.events import InitializationEvents
from src.workflows.facts import (
    SQLiteWorkflowFactRepository,
    WorkflowFact,
    workflow_fact_repository_factory,
)
from src.utils.classify_file_mode import classify_file_mode_from_path
from utils.extract_function_contract import extract_function_contract
from src.vectordb import VectorDB


class WorkflowContext(BaseModel):

    embedding_model: NNBaseModel
    function_registry: FunctionRegistry
    vdb: VectorDB
    run_id: str = ""
    epoch: int = 0
    workflow_name: str = "initialize"
    process_name: str = "initialize"
    batch_id: str | None = None
    workflow_fact_repository: SQLiteWorkflowFactRepository = Field(
        default_factory=workflow_fact_repository_factory
    )


class WorkflowState(BaseModel):
    """
    A TypedDict representing the structure of the InitializeWorkflow.
    This workflow is responsible for initializing the system, setting up necessary configurations,
    and preparing the environment for further operations.
    """
    # Define the expected keys and their types here
    file_metadata: list[FileMetadata] = Field(default_factory=list)
    function_code: list[str] = Field(default_factory=list)
    embedding_matrix: tensor = Field(default_factory=lambda: tensor([]))
    vector_db_path: str = Field(default="")


async def construct_file_metadata(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    registry = runtime.context.function_registry
    registry.scan()
    
    file_metadatas = []
    function_codes = []
    
    for name,path in registry.path_by_name.items():
        func = registry.load(name)
        contract = extract_function_contract(func)
        
        file_id = hashlib.sha256(contract["source"].encode('utf-8')).hexdigest()
        
        file_metadata = FileMetadata(
            id=file_id,
            filename=path.split("/")[-1],
            filepath=str(path),
            filetype=path.split(".")[-1],
            funcname=contract["name"],
            func_docstring=contract["docstring"],
            func_is_async=contract["is_async"],
            func_signature=contract["signature"],
            func_parameters={param["name"]: param for param in contract["parameters"]},
            mode=classify_file_mode_from_path(str(path)),
        )
        
        file_metadatas.append(file_metadata)
        function_codes.append(contract["source"] if contract["source"] else "")

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InitializationEvents.FILE_METADATA_CONSTRUCTED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                FileMetadataConstructedAttributes(
                    file_count=len(file_metadatas),
                    file_metadata_ids=[metadata.id for metadata in file_metadatas],
                    filenames=[metadata.filename for metadata in file_metadatas],
                    filepaths=[metadata.filepath for metadata in file_metadatas],
                )
            ),
        )
    )
        
        
    return {"file_metadata": file_metadatas, "function_code": function_codes}


async def embed_function_code(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    vectors = runtime.context.embedding_model.forward(texts=state.function_code)

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InitializationEvents.FUNCTION_CODE_EMBEDDED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                FunctionCodeEmbeddedAttributes(
                    function_code_count=len(state.function_code),
                    embedding_shape=list(getattr(vectors, "shape", [])),
                    file_metadata_ids=[metadata.id for metadata in state.file_metadata],
                )
            ),
        )
    )
    
    return {"embedding_matrix": vectors}


async def store_embeddings_in_vector_db(state: WorkflowState, runtime: Runtime[WorkflowContext]):

    
    vdb = runtime.context.vdb
    vdb.build_index(state.embedding_matrix, state.file_metadata)

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InitializationEvents.EMBEDDINGS_STORED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                EmbeddingsStoredAttributes(
                    file_metadata_count=len(state.file_metadata),
                    vector_db_path=str(vdb.path),
                    file_metadata_ids=[metadata.id for metadata in state.file_metadata],
                )
            ),
        )
    )
    
    return {"vector_db_path": str(vdb.path)}



def build_initialize_workflow():
    workflow_builder = StateGraph(WorkflowState)
    workflow_builder.add_node("construct_file_metadata")
    workflow_builder.add_node("embed_function_code")
    workflow_builder.add_node("store_embeddings_in_vector_db")
    workflow_builder.add_edge("construct_file_metadata", "embed_function_code")
    workflow_builder.add_edge("embed_function_code", "store_embeddings_in_vector_db")
    workflow_builder.add_edge(START, "construct_file_metadata")
    workflow_builder.add_edge("store_embeddings_in_vector_db", END)
    return workflow_builder.compile()