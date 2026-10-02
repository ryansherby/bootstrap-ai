from dataclasses import dataclass, field
from inspect import Parameter
from typing import Any, Literal, Optional, Dict
import pandas as pd
import uuid

from torch import Tensor, tensor





@dataclass
class Data:
    query: str
    tables: list[Dict]
    output: Optional[Dict]= None
    metadata: Optional[Dict] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    

@dataclass
class TestParams:
    input_data: list[dict[str,list[Any]]]
    parameters: dict[str,Any]
    output_data: list[dict[str,list[Any]]]
    
@dataclass
class ColumnMetadata:
    pos: int
    column_name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: list[Any] 
    
    
@dataclass
class TableMetadata:
    pos: int
    table_name: str
    row_count: int
    column_metadata: list[ColumnMetadata]
    sample_rows: list[dict]
    
    
@dataclass
class FileMetadata:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    filepath: str
    filetype: str
    funcname: str
    func_docstring: str
    func_is_async: bool
    func_signature: str
    func_parameters: dict[str, Parameter]
    embedding: Tensor = field(default_factory=lambda: tensor([]))
    mode: Literal["primtive", "generated", ""] = ""
    
    
@dataclass
class ComparisonRules:
    row_order_matters: bool = False
    column_order_matters: bool = False
    index_matters: bool = False
    column_names_matter: bool = True
    dtypes_matter: bool = False
    row_count_must_match: bool = True
    column_count_must_match: bool = True
    duplicate_counts_must_match: bool = True
    numeric_values_must_match_exactly: bool = True
    string_case_matters: bool = True
    string_whitespace_matters: bool = True
    null_representation_matters: bool = False
    
@dataclass
class ComparisonResult:
    matches: bool
    reasons: tuple[str,...] = ()