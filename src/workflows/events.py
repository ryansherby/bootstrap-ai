
from enum import StrEnum


class BootstrapEvents(StrEnum):
    SIMILAR_SOLVES_RETRIEVED = "similar_solves_retrieved"
    MERGED_FUNCTION_GENERATED = "merged_function_generated"
    
    
class InferenceEvents(StrEnum):
    PLAN_GENERATED = "plan_generated"
    SOLUTION_GENERATED = "solution_generated"
    QUERY_EMBEDDED = "query_embedded"
    FUNCTIONS_RETRIEVED = "functions_retrieved"
    INTERRUPT = "interrupt"
    SOLUTION_EXECUTED = "solution_executed"
    COMPARISON_RULE_GENERATED = "comparison_rule_generated"
    RESULTS_COMPARED = "results_compared"
    
    
class TestEvents(StrEnum):
    TEST_GENERATED = "test_generated"
    COMPARISON_RULE_GENERATED = "comparison_rule_generated"
    TEST_EXECUTED = "test_executed"
    RESULTS_COMPARED = "results_compared"
    
    
class WriteEmbedBatchEvents(StrEnum):
    FILES_METADATA_CREATED = "files_metadata_created"
    FILES_WRITTEN = "files_written"
    FILES_EMBEDDED = "files_embedded"
    FILES_STORED = "files_stored"


class InitializationEvents(StrEnum):
    FILE_METADATA_CONSTRUCTED = "file_metadata_constructed"
    FUNCTION_CODE_EMBEDDED = "function_code_embedded"
    EMBEDDINGS_STORED = "embeddings_stored"
    

    