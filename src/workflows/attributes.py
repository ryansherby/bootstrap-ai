from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionReference:
    id: str
    funcname: str
    filepath: str


@dataclass
class SimilarSolvesRetrievedAttributes:
    source_file_id: str
    source_function_name: str
    source_filepath: str
    retrieval_threshold: float
    top_k: int
    retrieved_count: int
    retrieved_functions: list[FunctionReference] = field(default_factory=list)


@dataclass
class MergedFunctionGeneratedAttributes:
    source_file_id: str
    source_function_name: str
    considered_count: int
    considered_functions: list[FunctionReference] = field(default_factory=list)
    selected_function: FunctionReference | None = None
    cache_hit: bool = False
    merged_function_present: bool = False
    merged_function_length: int = 0


@dataclass
class PlanGeneratedAttributes:
    data_id: str | None
    query: str
    table_count: int
    table_names: list[str | None]
    metadata: Any
    plan: str


@dataclass
class QueryEmbeddedAttributes:
    data_id: str | None
    query: str
    plan: str
    table_metadata: Any
    prompt_length: int
    embedding_shape: list[int]


@dataclass
class FunctionsRetrievedAttributes:
    data_id: str | None
    retrieval_top_k: int
    retrieval_threshold: float
    retrieved_count: int
    retrieved_functions: list[FunctionReference] = field(default_factory=list)


@dataclass
class SolutionGeneratedAttributes:
    data_id: str | None
    plan: str
    function_count: int
    functions_listed: list[FunctionReference] = field(default_factory=list)
    functions_called: dict[str, int] = field(default_factory=dict)
    code_length: int = 0


@dataclass
class SolutionExecutedAttributes:
    data_id: str | None
    code_length: int
    table_count: int
    execution_result_type: str
    execution_result_rows: int | None


@dataclass
class ComparisonRuleGeneratedAttributes:
    data_id: str | None
    query: str
    plan: str
    code_length: int
    comparison_rule: Any


@dataclass
class ComparisonResultPayload:
    matches: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class ResultsComparedAttributes:
    data_id: str | None
    expected_row_count: int
    actual_row_count: int
    comparison_result: ComparisonResultPayload


@dataclass
class InterruptAttributes:
    data_id: str | None
    execute: bool
    compare: bool
    plan_present: bool
    code_present: bool
    function_count: int


@dataclass
class FileMetadataConstructedAttributes:
    file_count: int
    file_metadata_ids: list[str] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    filepaths: list[str] = field(default_factory=list)


@dataclass
class FunctionCodeEmbeddedAttributes:
    function_code_count: int
    embedding_shape: list[int]
    file_metadata_ids: list[str] = field(default_factory=list)


@dataclass
class EmbeddingsStoredAttributes:
    file_metadata_count: int
    vector_db_path: str
    file_metadata_ids: list[str] = field(default_factory=list)


@dataclass
class FilesEmbeddedAttributes:
    item_count: int
    ids: list[str] = field(default_factory=list)
    text_lengths: list[int] = field(default_factory=list)
    embedding_shape: list[int] = field(default_factory=list)
    data_ids: list[str] = field(default_factory=list)


@dataclass
class FilesWrittenAttributes:
    item_count: int
    ids: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    data_ids: list[str] = field(default_factory=list)
    storage_write_count: int = 0
    storage_written_ids: list[str] = field(default_factory=list)


@dataclass
class FilesMetadataCreatedAttributes:
    item_count: int
    ids: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)
    function_names: list[str] = field(default_factory=list)
    file_metadata_ids: list[str] = field(default_factory=list)
    data_ids: list[str] = field(default_factory=list)


@dataclass
class FilesStoredAttributes:
    type: str
    file_metadata_count: int
    vector_db_path: str
    file_metadata_ids: list[str] = field(default_factory=list)
    data_ids: list[str] = field(default_factory=list)
    vector_db_write_count: int = 0
    vector_db_written_ids: list[str] = field(default_factory=list)


@dataclass
class TestGeneratedAttributes:
    data_id: str | None
    function_hash: str
    function_name: str
    function_text_length: int
    signature: str
    parameter_names: list[str] = field(default_factory=list)
    stub_length: int | None = None
    input_data_count: int | None = None
    output_data_count: int | None = None
    parameter_keys: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class TestComparisonRuleGeneratedAttributes:
    data_id: str | None
    function_hash: str
    function_name: str
    function_text_length: int
    comparison_basis_length: int
    comparison_rule: dict[str, Any] | None = None
    input_data_count: int | None = None
    output_data_count: int | None = None
    error: str | None = None


@dataclass
class TestExecutedAttributes:
    data_id: str | None
    function_hash: str
    function_name: str
    function_text_length: int
    input_data_count: int | None = None
    parameter_keys: list[str] = field(default_factory=list)
    execution_result_type: str | None = None
    output_frame_count: int | None = None
    error: str | None = None


@dataclass
class TestResultsComparedAttributes:
    data_id: str | None
    function_hash: str
    function_name: str
    function_text_length: int
    actual_output_count: int | None = None
    expected_output_count: int | None = None
    passed: bool | None = None
    reasons: list[str] = field(default_factory=list)
    error: str | None = None
