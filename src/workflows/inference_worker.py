

import ast
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from torch import Tensor, tensor
import pandas as pd


from configs import CONFIG
from src.execution.executor import FunctionRegistry, exec_solve
from src.types import ComparisonResult, ComparisonRules, Data, FileMetadata
from src.model_definitions.base_model import BaseModel as NNBaseModel
from src.workflows.attributes import (
    ComparisonResultPayload,
    ComparisonRuleGeneratedAttributes,
    FunctionReference,
    FunctionsRetrievedAttributes,
    InterruptAttributes,
    PlanGeneratedAttributes,
    QueryEmbeddedAttributes,
    ResultsComparedAttributes,
    SolutionExecutedAttributes,
    SolutionGeneratedAttributes,
)
from src.workflows.events import InferenceEvents
from src.workflows.facts import (
    SQLiteWorkflowFactRepository,
    WorkflowFact,
    workflow_fact_repository_factory,
)
from src.vectordb import VectorDB

from src.utils.extract_called_names import extract_called_names
from src.utils.render_function_stub import render_function_stub
from src.utils.compare_dataframes import compare_dataframes

from src.workflows.concurrency import limited_node



class WorkflowContext(BaseModel):

    transformer_model: NNBaseModel
    embedding_model: NNBaseModel
    function_registry: FunctionRegistry
    vdb: VectorDB
    run_id: str
    epoch: int
    workflow_name: str = "inference"
    process_name: str | None = None
    batch_id: str
    workflow_fact_repository: SQLiteWorkflowFactRepository = Field(
        default_factory=workflow_fact_repository_factory
    )

    
    
class WorkflowState(BaseModel):
    data: Data
    functions: list[FileMetadata] = Field(default_factory=list)
    top_k: int = CONFIG.vector_db.inference_retrieval_top_k
    retrieval_threshold: float = CONFIG.vector_db.inference_retrieval_threshold

    execute: bool = True
    compare: bool = True
    plan: str = ""
    code: str = ""
    comparison_rules: ComparisonRules = Field(default_factory=ComparisonRules)
    embedding: Tensor = Field(default_factory=lambda: tensor([]))
    execution_result: dict = Field(default_factory=dict)
    comparison_result: ComparisonResult = Field(
        default_factory=lambda: ComparisonResult(matches=False, reasons=())
    )


def _coerce_literal(value: Any, fallback: Any = None) -> Any:
    if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
        return value

    if not isinstance(value, str):
        return fallback if fallback is not None else value

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return fallback if fallback is not None else value


def _count_called_functions(
    source: str,
    allowed_names: set[str] | None = None,
) -> dict[str, int]:
    candidate_names = extract_called_names(source)
    if allowed_names is not None:
        candidate_names = {
            name for name in candidate_names
            if name in allowed_names
        }

    counts: dict[str, int] = {name: 0 for name in sorted(candidate_names)}

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return counts

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if fn_name in counts:
                counts[fn_name] += 1

    return counts


@limited_node("transformer")
async def generate_plan(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    plan = runtime.context.transformer_model.forward(
        process_name="plan",
        table_metadata=state.data.metadata,
        query=state.data.query
    )

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.PLAN_GENERATED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                PlanGeneratedAttributes(
                    data_id=state.data.id,
                    query=state.data.query,
                    table_count=len(state.data.tables),
                    table_names=[
                        table.get("table_name")
                        for table in state.data.tables
                        if isinstance(table, dict)
                    ],
                    metadata=state.data.metadata,
                    plan=plan,
                )
            ),
        )
    )
    
    
    
    return {"plan": plan}

@limited_node("embedding")
async def embed_query_metadata(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    prompt = open(CONFIG.embedding_model.inference_retrieval_prompt_path, "r").read()
    
    prompt = prompt.replace("{{TABLE_METADATA}}", str(state.data.metadata))
    prompt = prompt.replace("{{QUERY}}", str(state.data.query))
    prompt = prompt.replace("{{PLAN}}", str(state.plan))
    
    
    embedding = runtime.context.embedding_model.forward(
        process_name="embed",
        texts=[prompt]
    )

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.QUERY_EMBEDDED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                QueryEmbeddedAttributes(
                    data_id=state.data.id,
                    query=state.data.query,
                    plan=state.plan,
                    table_metadata=state.data.metadata,
                    prompt_length=len(prompt),
                    embedding_shape=list(getattr(embedding, "shape", [])),
                )
            ),
        )
    )
    
    return {"embedding": embedding}


@limited_node("vdb_search")
async def retrieve_relevant_functions(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    results = runtime.context.vdb.search(state.embedding, top_k=state.top_k, search_type="mmr")  # Adjust top_k as needed
    
    functions = []
    for result in results:
        
        if result["score"] < state.retrieval_threshold:
            continue
        
        functions.append(result["metadata"])

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.FUNCTIONS_RETRIEVED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                FunctionsRetrievedAttributes(
                    data_id=state.data.id,
                    retrieval_top_k=state.top_k,
                    retrieval_threshold=state.retrieval_threshold,
                    retrieved_count=len(functions),
                    retrieved_functions=[
                        FunctionReference(
                            id=function_metadata.id,
                            funcname=function_metadata.funcname,
                            filepath=function_metadata.filepath,
                        )
                        for function_metadata in functions
                    ],
                )
            ),
        )
    )
    
    return {"functions": functions}





@limited_node("transformer")
async def generate_solution(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    code = runtime.context.transformer_model.forward(
        process_name="code",
        plan=state.plan,
        functions=[
            render_function_stub(
                is_async=func.func_is_async,
                name=func.funcname,
                signature=func.func_signature,
                docstring=func.func_docstring,
            )
            for func in state.functions
        ],
    )

    listed_function_names = {
        function_metadata.funcname
        for function_metadata in state.functions
    }
    functions_called = _count_called_functions(
        code,
        allowed_names=listed_function_names,
    )

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.SOLUTION_GENERATED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                SolutionGeneratedAttributes(
                    data_id=state.data.id,
                    plan=state.plan,
                    function_count=len(state.functions),
                    functions_listed=[
                        FunctionReference(
                            id=function_metadata.id,
                            funcname=function_metadata.funcname,
                            filepath=function_metadata.filepath,
                        )
                        for function_metadata in state.functions
                    ],
                    functions_called=functions_called,
                    code_length=len(code),
                )
            ),
        )
    )
    
    
    
    return {"code": code}



def execution_routing(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    if state.execute:
        return "execute_solution"
    else:
        return "interrupt_handler"
    


async def execute_solution(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    
    res = exec_solve(source=state.code, registry=runtime.context.function_registry, fn_kwargs={"tables": [pd.DataFrame(table) for table in state.data.tables]})

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.SOLUTION_EXECUTED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                SolutionExecutedAttributes(
                    data_id=state.data.id,
                    code_length=len(state.code),
                    table_count=len(state.data.tables),
                    execution_result_type=type(res).__name__,
                    execution_result_rows=(len(res) if isinstance(res, pd.DataFrame) else None),
                )
            ),
        )
    )
    
    return {"execution_result": res.to_dict(orient="records") if isinstance(res, pd.DataFrame) else res}

def comparison_routing(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    if state.compare:
        return "generate_comparison_rules"
    else:
        return "interrupt_handler"
    
@limited_node("transformer")
async def generate_comparison_rules(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    content = f"""
    ### Query
    {state.data.query}
    
    ### Table Metadata
    {state.data.metadata}
    
    ### Plan
    {state.plan}
    
    ### Code
    {state.code}
    """

    
    raw_output = runtime.context.transformer_model.forward(
        process_name="comparison_rules",
        comparison_basis=content
    )

    parsed_comparison_rules = _coerce_literal(raw_output, {})
    if not isinstance(parsed_comparison_rules, dict):
        parsed_comparison_rules = {}

    try:
        rules = ComparisonRules(**parsed_comparison_rules)
    except Exception:
        rules = ComparisonRules()
        parsed_comparison_rules = rules.__dict__

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.COMPARISON_RULE_GENERATED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                ComparisonRuleGeneratedAttributes(
                    data_id=state.data.id,
                    query=state.data.query,
                    plan=state.plan,
                    code_length=len(state.code),
                    comparison_rule=parsed_comparison_rules,
                )
            ),
        )
    )
    
    return {"comparison_rules": rules}
    

    
async def compare_results(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    
    output_df = pd.DataFrame(state.data.output)
    solved_df = pd.DataFrame(state.execution_result)
    
    
    res = compare_dataframes(output_df, solved_df, state.comparison_rules)

    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.RESULTS_COMPARED,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                ResultsComparedAttributes(
                    data_id=state.data.id,
                    expected_row_count=len(output_df),
                    actual_row_count=len(solved_df),
                    comparison_result=ComparisonResultPayload(
                        matches=res.matches,
                        reasons=list(res.reasons),
                    ),
                )
            ),
        )
    )
    
    return {"comparison_result": res}

async def interrupt_handler(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    await runtime.context.workflow_fact_repository.record(
        WorkflowFact(
            event_id=str(uuid4()),
            event_type=InferenceEvents.INTERRUPT,
            run_id=str(runtime.context.run_id),
            epoch=str(runtime.context.epoch),
            batch_id=runtime.context.batch_id,
            workflow_name=runtime.context.workflow_name,
            process_name=runtime.context.process_name,
            attributes=asdict(
                InterruptAttributes(
                    data_id=state.data.id,
                    execute=state.execute,
                    compare=state.compare,
                    plan_present=bool(state.plan),
                    code_present=bool(state.code),
                    function_count=len(state.functions),
                )
            ),
        )
    )

    return {}



def build_inference_worker() -> StateGraph:
    graph = StateGraph(
        WorkflowState,
        context_schema=WorkflowContext,
    )

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    graph.add_node("generate_plan", generate_plan)
    graph.add_node("embed_query_metadata", embed_query_metadata)
    graph.add_node("retrieve_relevant_functions", retrieve_relevant_functions)
    graph.add_node("generate_solution", generate_solution)

    graph.add_node("execute_solution", execute_solution)
    graph.add_node("generate_comparison_rules", generate_comparison_rules)
    graph.add_node("compare_results", compare_results)

    graph.add_node("interrupt_handler", interrupt_handler)

    # ------------------------------------------------------------------
    # Main inference path
    # ------------------------------------------------------------------

    graph.add_edge(START, "generate_plan")
    graph.add_edge("generate_plan", "embed_query_metadata")
    graph.add_edge("embed_query_metadata", "retrieve_relevant_functions")
    graph.add_edge("retrieve_relevant_functions", "generate_solution")

    # ------------------------------------------------------------------
    # Execution routing
    # ------------------------------------------------------------------

    graph.add_conditional_edges(
        "generate_solution",
        execution_routing,
        {
            "execute_solution": "execute_solution",
            "interrupt_handler": "interrupt_handler",
        },
    )

    # ------------------------------------------------------------------
    # Comparison routing
    # ------------------------------------------------------------------

    graph.add_conditional_edges(
        "execute_solution",
        comparison_routing,
        {
            "generate_comparison_rules": "generate_comparison_rules",
            "interrupt_handler": "interrupt_handler",
        },
    )

    graph.add_edge(
        "generate_comparison_rules",
        "compare_results",
    )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    graph.add_edge("compare_results", END)
    graph.add_edge("interrupt_handler", END)

    return graph.compile()
    
    