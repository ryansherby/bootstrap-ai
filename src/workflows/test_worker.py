import ast
from dataclasses import asdict
import inspect
from typing import Any
from uuid import uuid4

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.execution.executor import FunctionRegistry, exec_solve
from src.model_definitions.base_model import BaseModel as NNBaseModel
from src.types import ComparisonResult, ComparisonRules, TestParams
from src.utils.extract_function_contract import extract_function_contract
from src.utils.compare_dataframes import compare_dataframes
from src.utils.render_function_stub import render_function_stub
from src.workflows.attributes import (
    TestComparisonRuleGeneratedAttributes,
    TestExecutedAttributes,
    TestGeneratedAttributes,
    TestResultsComparedAttributes,
)
from src.workflows.events import TestEvents
from src.workflows.facts import (
    SQLiteWorkflowFactRepository,
    WorkflowFact,
    workflow_fact_repository_factory,
)
from src.workflows.concurrency import limited_node
from src.vectordb import VectorDB


class WorkflowContext(BaseModel):
    transformer_model: NNBaseModel
    function_registry: FunctionRegistry
    vdb: VectorDB | None = None
    run_id: str = ""
    epoch: int = 0
    workflow_name: str = "test"
    process_name: str | None = None
    batch_id: str = ""
    workflow_fact_repository: SQLiteWorkflowFactRepository = Field(
        default_factory=workflow_fact_repository_factory
    )


class WorkflowState(BaseModel):
    function_text: str
    test_params: TestParams | None = None
    comparison_basis: str = ""
    comparison_rules: ComparisonRules = Field(default_factory=ComparisonRules)
    execution_result: Any = None
    actual_output_data: list[dict[str, list[Any]]] = Field(default_factory=list)
    comparison_result: ComparisonResult = Field(
        default_factory=lambda: ComparisonResult(matches=False, reasons=())
    )
    passed: bool = False
    error_message: str = ""


def _resolve_function_contract_from_text(function_text: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"pd": pd}
    exec(function_text, namespace, namespace)

    function_candidates = [
        value
        for name, value in namespace.items()
        if callable(value) and not name.startswith("__") and name != "pd"
    ]
    if not function_candidates:
        raise ValueError("No callable definition found in function_text.")

    contract = extract_function_contract(function_candidates[0])
    contract["function_hash"] = contract["hash"]
    contract["function_text"] = contract["source"] or function_text
    contract["function_parameters"] = {
        parameter["name"]: parameter
        for parameter in contract["parameters"]
    }
    return contract


def _coerce_literal(value: Any, fallback: Any = None) -> Any:
    """Safely convert a model response into a Python literal."""
    if isinstance(value, (dict, list, tuple, int, float, bool)) or value is None:
        return value

    if not isinstance(value, str):
        return fallback if fallback is not None else value

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return fallback if fallback is not None else value


def _normalize_execution_output(value: Any) -> list[dict[str, list[Any]]]:
    if isinstance(value, pd.DataFrame):
        return [value.to_dict(orient="list")]

    if isinstance(value, (list, tuple)):
        output: list[dict[str, list[Any]]] = []
        for item in value:
            if isinstance(item, pd.DataFrame):
                output.append(item.to_dict(orient="list"))
            elif isinstance(item, dict):
                output.append(item)
            else:
                output.append({"value": [item]})
        return output

    if isinstance(value, dict):
        return [value]

    return [{"value": [value]}]


def _is_dataframe_parameter(param: inspect.Parameter | dict[str, Any]) -> bool:
    if isinstance(param, inspect.Parameter):
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            return False

        if annotation is pd.DataFrame:
            return True

        annotation_text = str(annotation)
        return "DataFrame" in annotation_text or "pd.DataFrame" in annotation_text

    annotation = param.get("annotation")
    if annotation is None:
        return False

    annotation_text = str(annotation)
    return "DataFrame" in annotation_text or "pd.DataFrame" in annotation_text


def _coerce_test_params(value: Any) -> TestParams:
    if not isinstance(value, dict):
        raise ValueError("Model output for test params is not a dictionary.")

    if "input_data" not in value or "parameters" not in value or "output_data" not in value:
        raise ValueError("Test params must include input_data, parameters, and output_data.")

    return TestParams(
        input_data=value["input_data"],
        parameters=value["parameters"],
        output_data=value["output_data"],
    )


def _build_call_arguments(
    func_parameters: dict[str, Any],
    test_params: TestParams,
) -> tuple[list[Any], dict[str, Any]]:
    input_frames = [pd.DataFrame(table) for table in test_params.input_data]
    parameters = test_params.parameters or {}

    ordered_params = list(func_parameters.values())
    dataframe_params = [param for param in ordered_params if _is_dataframe_parameter(param)]

    if len(input_frames) < len(dataframe_params):
        raise ValueError(
            "Not enough input_data entries for dataframe parameters: "
            f"expected at least {len(dataframe_params)}, got {len(input_frames)}"
        )

    # Dataframe inputs are always passed first as positional args.
    fn_args: list[Any] = input_frames[: len(dataframe_params)]

    # Any additional model-provided dataframe inputs are passed positionally after
    # required dataframe parameters to preserve input_data ordering.
    if len(input_frames) > len(dataframe_params):
        fn_args.extend(input_frames[len(dataframe_params) :])

    fn_kwargs: dict[str, Any] = {}
    for param in ordered_params:
        if isinstance(param, inspect.Parameter):
            param_name = param.name
            param_kind = param.kind
        else:
            param_name = str(param.get("name", ""))
            param_kind = str(param.get("kind", "POSITIONAL_OR_KEYWORD"))

        if _is_dataframe_parameter(param):
            continue

        is_var_positional = (
            param_kind is inspect.Parameter.VAR_POSITIONAL
            if isinstance(param_kind, inspect._ParameterKind)
            else param_kind == "VAR_POSITIONAL"
        )
        if is_var_positional:
            extra_args = parameters.get(param_name)
            if isinstance(extra_args, (list, tuple)):
                fn_args.extend(extra_args)
            continue

        is_var_keyword = (
            param_kind is inspect.Parameter.VAR_KEYWORD
            if isinstance(param_kind, inspect._ParameterKind)
            else param_kind == "VAR_KEYWORD"
        )
        if is_var_keyword:
            extra_kwargs = parameters.get(param_name)
            if isinstance(extra_kwargs, dict):
                fn_kwargs.update(extra_kwargs)
            continue

        is_positional_only = (
            param_kind is inspect.Parameter.POSITIONAL_ONLY
            if isinstance(param_kind, inspect._ParameterKind)
            else param_kind == "POSITIONAL_ONLY"
        )
        if is_positional_only and param_name in parameters:
            fn_args.append(parameters[param_name])
            continue

        if param_name in parameters:
            fn_kwargs[param_name] = parameters[param_name]

    return fn_args, fn_kwargs


@limited_node("transformer")
async def generate_test(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """Generate parameter values for the target function."""
    transformer_model = runtime.context.transformer_model
    function_contract = _resolve_function_contract_from_text(state.function_text)
    data_id = function_contract["function_hash"]
    function_stub = render_function_stub(
        is_async=function_contract["is_async"],
        name=function_contract["name"],
        signature=function_contract["signature"],
        docstring=function_contract["docstring"],
    )

    try:
        raw_output = transformer_model.forward(
            process_name="test",
            code_snippets=[function_stub],
        )
        parsed_output = _coerce_literal(raw_output, {})
        test_params = _coerce_test_params(parsed_output)

        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.TEST_GENERATED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestGeneratedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        signature=function_contract["signature"],
                        parameter_names=list(function_contract["function_parameters"].keys()),
                        stub_length=len(function_stub),
                        input_data_count=len(test_params.input_data),
                        output_data_count=len(test_params.output_data),
                        parameter_keys=list(test_params.parameters.keys()),
                    )
                ),
            )
        )

        return {
            "test_params": test_params,
            "error_message": "",
        }
    except Exception as exc:
        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.TEST_GENERATED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestGeneratedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        signature=function_contract["signature"],
                        parameter_names=list(function_contract["function_parameters"].keys()),
                        error=str(exc),
                    )
                ),
            )
        )
        return {
            "test_params": None,
            "error_message": f"generate_test failed: {exc}",
        }


@limited_node("transformer")
async def generate_comparison_rules(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """Generate comparison rules for the target function tests."""
    if state.error_message:
        return {}

    transformer_model = runtime.context.transformer_model
    function_contract = _resolve_function_contract_from_text(state.function_text)
    data_id = function_contract["function_hash"]
    source_code = function_contract["function_text"]

    comparison_basis = f"""
    ### Function
    {source_code}

    ### Test Parameters
    {state.test_params}

    ### Execution Result
    {state.actual_output_data}
    """

    try:
        raw_output = transformer_model.forward(
            process_name="comparison_rules",
            comparison_basis=comparison_basis,
        )
        rules_literal = _coerce_literal(raw_output, {})
        rules = ComparisonRules(**rules_literal) if isinstance(rules_literal, dict) else ComparisonRules()

        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.COMPARISON_RULE_GENERATED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestComparisonRuleGeneratedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        comparison_basis_length=len(comparison_basis),
                        comparison_rule=rules.__dict__,
                        input_data_count=0 if state.test_params is None else len(state.test_params.input_data),
                        output_data_count=0 if state.test_params is None else len(state.test_params.output_data),
                    )
                ),
            )
        )

        return {
            "comparison_basis": comparison_basis,
            "comparison_rules": rules,
            "error_message": "",
        }
    except Exception as exc:
        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.COMPARISON_RULE_GENERATED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestComparisonRuleGeneratedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        comparison_basis_length=len(comparison_basis),
                        error=str(exc),
                    )
                ),
            )
        )
        return {
            "comparison_basis": comparison_basis,
            "comparison_rules": ComparisonRules(),
            "error_message": f"generate_comparison_rules failed: {exc}",
        }


async def execute_test(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """Execute the target function using state values produced by the workflow."""
    if state.error_message:
        return {}

    if state.test_params is None:
        function_contract = _resolve_function_contract_from_text(state.function_text)
        data_id = function_contract["function_hash"]
        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.TEST_EXECUTED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestExecutedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        error="test_params missing",
                    )
                ),
            )
        )
        return {
            "execution_result": None,
            "actual_output_data": [],
            "error_message": "execute_test failed: test_params missing.",
        }

    function_contract = _resolve_function_contract_from_text(state.function_text)
    registry = runtime.context.function_registry
    data_id = function_contract["function_hash"]

    try:

        fn_args, fn_kwargs = _build_call_arguments(
            function_contract["function_parameters"],
            state.test_params,
        )

        execution_result = exec_solve(
            source=function_contract["function_text"],
            registry=registry,
            fn_name=function_contract["name"],
            fn_args=fn_args,
            fn_kwargs=fn_kwargs,
        )

        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.TEST_EXECUTED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestExecutedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        input_data_count=len(state.test_params.input_data),
                        parameter_keys=list(state.test_params.parameters.keys()),
                        execution_result_type=type(execution_result).__name__,
                        output_frame_count=(
                            len(execution_result)
                            if isinstance(execution_result, tuple)
                            else (1 if isinstance(execution_result, pd.DataFrame) else None)
                        ),
                    )
                ),
            )
        )

        return {
            "execution_result": execution_result,
            "actual_output_data": _normalize_execution_output(execution_result),
            "error_message": "",
        }
    except Exception as exc:
        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.TEST_EXECUTED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestExecutedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        input_data_count=len(state.test_params.input_data),
                        parameter_keys=list(state.test_params.parameters.keys()),
                        error=str(exc),
                    )
                ),
            )
        )
        return {
            "execution_result": None,
            "actual_output_data": [],
            "error_message": f"execute_test failed: {exc}",
        }


async def compare_results(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    """Compare normalized actual output against expected output from generated tests."""
    function_contract = _resolve_function_contract_from_text(state.function_text)
    data_id = function_contract["function_hash"]

    if state.error_message:
        return {
            "comparison_result": ComparisonResult(matches=False, reasons=(state.error_message,)),
            "passed": False,
        }

    if state.test_params is None:
        return {
            "comparison_result": ComparisonResult(matches=False, reasons=("test_params missing",)),
            "passed": False,
        }

    expected_output_data = state.test_params.output_data

    try:
        actual_frames = [pd.DataFrame(data) for data in state.actual_output_data]
        expected_frames = [pd.DataFrame(data) for data in expected_output_data]

        if len(actual_frames) != len(expected_frames):
            reason = (
                "Output frame count differs: "
                f"actual={len(actual_frames)}, expected={len(expected_frames)}"
            )
            await runtime.context.workflow_fact_repository.record(
                WorkflowFact(
                    event_id=str(uuid4()),
                    event_type=TestEvents.RESULTS_COMPARED,
                    run_id=str(runtime.context.run_id),
                    epoch=str(runtime.context.epoch),
                    batch_id=runtime.context.batch_id,
                    workflow_name=runtime.context.workflow_name,
                    process_name=runtime.context.process_name,
                    attributes=asdict(
                        TestResultsComparedAttributes(
                            data_id=data_id,
                            function_hash=function_contract["function_hash"],
                            function_name=function_contract["name"],
                            function_text_length=len(function_contract["function_text"]),
                            actual_output_count=len(actual_frames),
                            expected_output_count=len(expected_frames),
                            passed=False,
                            reasons=[reason],
                        )
                    ),
                )
            )
            return {
                "comparison_result": ComparisonResult(matches=False, reasons=(reason,)),
                "passed": False,
            }

        reasons: list[str] = []
        for index, (actual_df, expected_df) in enumerate(zip(actual_frames, expected_frames)):
            result = compare_dataframes(actual_df, expected_df, state.comparison_rules)
            if not result.matches:
                reason = result.reasons[0] if result.reasons else "DataFrames differ"
                reasons.append(f"output[{index}]: {reason}")

        if reasons:
            await runtime.context.workflow_fact_repository.record(
                WorkflowFact(
                    event_id=str(uuid4()),
                    event_type=TestEvents.RESULTS_COMPARED,
                    run_id=str(runtime.context.run_id),
                    epoch=str(runtime.context.epoch),
                    batch_id=runtime.context.batch_id,
                    workflow_name=runtime.context.workflow_name,
                    process_name=runtime.context.process_name,
                    attributes=asdict(
                        TestResultsComparedAttributes(
                            data_id=data_id,
                            function_hash=function_contract["function_hash"],
                            function_name=function_contract["name"],
                            function_text_length=len(function_contract["function_text"]),
                            actual_output_count=len(actual_frames),
                            expected_output_count=len(expected_frames),
                            passed=False,
                            reasons=reasons,
                        )
                    ),
                )
            )
            return {
                "comparison_result": ComparisonResult(matches=False, reasons=tuple(reasons)),
                "passed": False,
            }

        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.RESULTS_COMPARED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestResultsComparedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        actual_output_count=len(actual_frames),
                        expected_output_count=len(expected_frames),
                        passed=True,
                        reasons=[],
                    )
                ),
            )
        )
        return {
            "comparison_result": ComparisonResult(matches=True, reasons=()),
            "passed": True,
        }
    except Exception as exc:
        message = f"compare_results failed: {exc}"
        await runtime.context.workflow_fact_repository.record(
            WorkflowFact(
                event_id=str(uuid4()),
                event_type=TestEvents.RESULTS_COMPARED,
                run_id=str(runtime.context.run_id),
                epoch=str(runtime.context.epoch),
                batch_id=runtime.context.batch_id,
                workflow_name=runtime.context.workflow_name,
                process_name=runtime.context.process_name,
                attributes=asdict(
                    TestResultsComparedAttributes(
                        data_id=data_id,
                        function_hash=function_contract["function_hash"],
                        function_name=function_contract["name"],
                        function_text_length=len(function_contract["function_text"]),
                        error=message,
                    )
                ),
            )
        )
        return {
            "comparison_result": ComparisonResult(matches=False, reasons=(message,)),
            "passed": False,
            "error_message": message,
        }


def test_routing(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    if state.error_message:
        return "error_handler"
    if state.passed:
        return "success_handler"
    return "failure_handler"


async def success_handler(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    return {"passed": True}


async def failure_handler(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    return {"passed": False}


async def error_handler(state: WorkflowState, runtime: Runtime[WorkflowContext]):
    return {"passed": False}


def build_test_worker() -> StateGraph:
    """Build and compile the test worker graph."""
    graph = StateGraph(
        WorkflowState,
        context_schema=WorkflowContext,
    )

    graph.add_node("generate_test", generate_test)
    graph.add_node("execute_test", execute_test)
    graph.add_node("generate_comparison_rules", generate_comparison_rules)
    graph.add_node("compare_results", compare_results)
    graph.add_node("success_handler", success_handler)
    graph.add_node("failure_handler", failure_handler)
    graph.add_node("error_handler", error_handler)

    graph.add_edge(START, "generate_test")
    graph.add_edge("generate_test", "execute_test")
    graph.add_edge("execute_test", "generate_comparison_rules")
    graph.add_edge("generate_comparison_rules", "compare_results")
    graph.add_conditional_edges(
        "compare_results",
        test_routing,
        {
            "success_handler": "success_handler",
            "failure_handler": "failure_handler",
            "error_handler": "error_handler",
        },
    )

    graph.add_edge("success_handler", END)
    graph.add_edge("failure_handler", END)
    graph.add_edge("error_handler", END)

    return graph.compile()

