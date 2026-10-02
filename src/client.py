from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import duckdb
import pandas as pd

from configs import CONFIG
from src.execution.executor import FunctionRegistry
from src.model_definitions.base_model import BaseModel as NNBaseModel
from src.types import ColumnMetadata, ComparisonResult, Data, TableMetadata
from src.vectordb import VectorDB
from src.workflows.cache import SharedCache
from src.workflows.concurrency import ConcurrencyPolicy, ConcurrencyRegistry
from src.workflows.facts import SQLiteWorkflowFactRepository
from src.workflows.inference import (
    WorkflowContext as InferenceWorkflowContext,
    WorkflowState as InferenceWorkflowState,
    build_inference_workflow,
)
from src.workflows.initialize import (
    WorkflowContext as InitializeWorkflowContext,
    WorkflowState as InitializeWorkflowState,
    build_initialize_workflow,
)
from src.workflows.test import (
    WorkflowContext as TestWorkflowContext,
    WorkflowState as TestWorkflowState,
    build_test_workflow,
)
from src.workflows.train import (
    WorkflowContext as TrainWorkflowContext,
    WorkflowState as TrainWorkflowState,
    build_train_workflow,
)


class DuckDb:
    """
    Read-only DuckDB interface over the SQLite facts database.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(
            db_path or CONFIG.facts_db_path
        ).expanduser().resolve()

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Facts database does not exist: {self.db_path}"
            )

        self.conn = duckdb.connect(":memory:")

        path = str(self.db_path).replace("'", "''")

        self.conn.execute(
            f"""
            ATTACH '{path}'
            AS facts
            (TYPE SQLITE, READ_ONLY)
            """
        )

    def sql(self, query: str):
        return self.conn.sql(query)

    def table(self, table_name: str):
        table_name = table_name.replace('"', '""')

        return self.conn.sql(
            f'SELECT * FROM facts."{table_name}"'
        )

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



# REMEMBER TO CLEAR THE SHARED CACHE AFTER EACH EPOCH

class Client:
    def __init__(
        self,
        transformer_model: NNBaseModel,
        embedding_model: NNBaseModel,
        permanent_library_path: str | Path | None = None,
        temp_library_root: str | Path | None = None,
        permanent_vdb_uuid: str | None = None,
        facts_db_path: str | Path | None = None,
        max_workers: int | None = None,
        resource_limits: dict[str, int] | None = None,
    ):
        self.transformer_model = transformer_model
        self.embedding_model = embedding_model

        self.permanent_library_path = str(
            Path(permanent_library_path or CONFIG.library_path).expanduser().resolve()
        )
        self.temp_library_root = Path(
            temp_library_root or CONFIG.temp_path
        ).expanduser().resolve()
        self.temp_library_root.mkdir(parents=True, exist_ok=True)

        self.permanent_vdb_uuid = permanent_vdb_uuid or str(uuid.uuid4())
        self.permanent_vdb_name = f"vdb-permanent-{self.permanent_vdb_uuid}"
        self.permanent_vdb = VectorDB(name=self.permanent_vdb_name)

        policy = ConcurrencyPolicy(
            max_workers=max_workers or int(CONFIG.max_workers),
            resource_limits=resource_limits or dict(CONFIG.workers),
        )
        self.concurrency = ConcurrencyRegistry(policy)

        self.shared_cache = SharedCache()
        self.temp_vdbs: dict[tuple[str, int, str], VectorDB] = {}

        self.facts_db_path = Path(facts_db_path or CONFIG.facts_db_path).expanduser().resolve()
        self.workflow_fact_repository = SQLiteWorkflowFactRepository(self.facts_db_path)
        self.facts_db = DuckDb(self.facts_db_path)

        self._permanent_vdb_cleared = False

    def _get_metadata(self,tables:list[pd.DataFrame]) -> list[TableMetadata]:
        metadatas = []
        for table in tables:
            metadata = TableMetadata(pos=tables.index(table), table_name=table.name,
                                     row_count=len(table),
                                     sample_rows=table.sample(min(len(table), 5)).to_dict(orient="records"),
                                     column_metadata=[ColumnMetadata(pos=table.columns.get_loc(col), column_name=col, dtype=str(table[col].dtype),
                                                                     unique_count=table[col].nunique(),
                                                                     sample_values=table[col].sample(min(len(table), 5)).tolist())
                                                      for col in table.columns])
            metadatas.append(metadata)
        return metadatas

    def _refresh_facts_db(self) -> None:
        self.facts_db.close()
        self.facts_db = DuckDb(self.facts_db_path)

    def _result_envelope(
        self,
        run_id: str,
        epochs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "epoch_count": len(epochs),
            "epochs": epochs,
        }

    def _single_epoch_result(
        self,
        run_id: str,
        batch_id: str,
        output: dict[str, Any],
        success: bool,
    ) -> dict[str, Any]:
        return self._result_envelope(
            run_id=run_id,
            epochs={
                "1": {
                    "batch_ids": [batch_id],
                    "output": output,
                    "epoch_success": success,
                }
            },
        )

    def _to_dataset(
        self,
        queries: list[str],
        tables: list[list[pd.DataFrame]],
        outputs: Optional[list[pd.DataFrame]] = None,
        metadata: Optional[list[Dict]] = None,
    ) -> list[Data]:
        if metadata is None:
            metadata = [None for _ in queries]

        if len(queries) != len(tables):
            raise ValueError("Length of queries must match length of tables.")
        if outputs is not None and len(outputs) != len(queries):
            raise ValueError("Length of outputs must match length of queries.")
        if len(metadata) != len(queries):
            raise ValueError("Length of metadata must match length of queries.")

        dataset: list[Data] = []
        for index, query in enumerate(queries):
            table_payload = [
                table.to_dict(orient="records")
                for table in tables[index]
            ]
            output_payload = (
                outputs[index].to_dict(orient="records")
                if outputs is not None
                else None
            )

            dataset.append(
                Data(
                    id=str(uuid.uuid4()),
                    query=query,
                    tables=table_payload,
                    output=output_payload,
                    metadata=metadata[index],
                )
            )

        return dataset

    def _slice_batches(
        self,
        dataset: list[Data],
        batch_size: int | str,
    ) -> list[list[Data]]:
        if not dataset:
            return []

        if batch_size == "all":
            return [dataset]

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be 'all' or a positive integer.")

        return [
            dataset[i: i + batch_size]
            for i in range(0, len(dataset), batch_size)
        ]

    def clear_permanent_vdb(self) -> None:
        self.permanent_vdb.delete()
        self._permanent_vdb_cleared = True

    def clear_db(self):
        self.clear_permanent_vdb()

    def initialize(
        self,
        library_path: str | Path | None = None,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        batch_id = "batch-1"

        if not self._permanent_vdb_cleared:
            return self._single_epoch_result(
                run_id=run_id,
                batch_id=batch_id,
                output={
                    "stored_count": 0,
                    "stored_ids": [],
                    "error": "initialize blocked: clear_permanent_vdb must be called before initialize.",
                },
                success=False,
            )

        if library_path is not None:
            self.permanent_library_path = str(Path(library_path).expanduser().resolve())

        workflow = build_initialize_workflow()

        async def _run_initialize() -> dict[str, Any]:
            registry = FunctionRegistry([self.permanent_library_path])
            context = InitializeWorkflowContext(
                embedding_model=self.embedding_model,
                function_registry=registry,
                vdb=self.permanent_vdb,
                run_id=run_id,
                epoch=1,
                workflow_name="initialize",
                process_name="initialize",
                batch_id=batch_id,
                workflow_fact_repository=self.workflow_fact_repository,
            )
            state = InitializeWorkflowState()
            return await workflow.ainvoke(state, context=context)

        try:
            result = asyncio.run(_run_initialize())
            file_metadata = result.get("file_metadata", [])
            output = {
                "stored_count": len(file_metadata),
                "stored_ids": [metadata.id for metadata in file_metadata],
                "vector_db_path": result.get("vector_db_path", str(self.permanent_vdb.path)),
            }
            self._permanent_vdb_cleared = False
            success = True
        except Exception as exc:
            output = {
                "stored_count": 0,
                "stored_ids": [],
                "error": str(exc),
            }
            success = False
        finally:
            self._refresh_facts_db()

        return self._single_epoch_result(
            run_id=run_id,
            batch_id=batch_id,
            output=output,
            success=success,
        )

    def init_db(self, library_path=None):
        return self.initialize(library_path=library_path)

    def inference(
        self,
        data: list[Data],
        result_type: Literal["raw_code", "execution_result", "comparison_result"] = "comparison_result",
        execute: bool = True,
        compare: bool = True,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        batch_id = "batch-1"
        compare = compare and execute

        workflow = build_inference_workflow()

        async def _run_inference() -> dict[str, Any]:
            context = InferenceWorkflowContext(
                transformer_model=self.transformer_model,
                embedding_model=self.embedding_model,
                function_registry=FunctionRegistry([self.permanent_library_path]),
                vdb=self.permanent_vdb,
                run_id=run_id,
                epoch=1,
                workflow_name="inference",
                process_name="inference",
                batch_id=batch_id,
                workflow_fact_repository=self.workflow_fact_repository,
                concurrency=self.concurrency,
            )
            state = InferenceWorkflowState(
                data=data,
                execute=execute,
                compare=compare,
            )
            return await workflow.ainvoke(state, context=context)

        try:
            result = asyncio.run(_run_inference())

            if result_type == "raw_code":
                ordered_results = list(result.get("code_results", []))
            elif result_type == "execution_result":
                ordered_results = list(result.get("execution_results", []))
            else:
                if not compare:
                    ordered_results = [None for _ in data]
                else:
                    comparison_results = list(result.get("comparison_results", []))
                    ordered_results = []
                    for value in comparison_results:
                        if isinstance(value, ComparisonResult):
                            ordered_results.append(
                                {
                                    "matches": value.matches,
                                    "reasons": list(value.reasons),
                                }
                            )
                        elif isinstance(value, dict):
                            ordered_results.append(
                                {
                                    "matches": bool(value.get("matches", False)),
                                    "reasons": list(value.get("reasons", [])),
                                }
                            )
                        else:
                            ordered_results.append(None)

            output = {
                "result_type": result_type,
                "execute": execute,
                "compare": compare,
                "results": ordered_results,
            }
            success = True
        except Exception as exc:
            output = {
                "result_type": result_type,
                "execute": execute,
                "compare": compare,
                "results": [],
                "error": str(exc),
            }
            success = False
        finally:
            self._refresh_facts_db()

        return self._single_epoch_result(
            run_id=run_id,
            batch_id=batch_id,
            output=output,
            success=success,
        )

    def test(
        self,
        function_texts: list[str],
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        batch_id = "batch-1"

        workflow = build_test_workflow()

        async def _run_test() -> dict[str, Any]:
            context = TestWorkflowContext(
                transformer_model=self.transformer_model,
                function_registry=FunctionRegistry([self.permanent_library_path]),
                vdb=self.permanent_vdb,
                run_id=run_id,
                epoch=1,
                workflow_name="test",
                process_name="test",
                batch_id=batch_id,
                workflow_fact_repository=self.workflow_fact_repository,
                concurrency=self.concurrency,
            )
            state = TestWorkflowState(function_texts=function_texts)
            return await workflow.ainvoke(state, context=context)

        try:
            result = asyncio.run(_run_test())
            worker_results = list(result.get("test_results", []))
            ordered_results = []

            for worker_result in worker_results:
                passed = bool(worker_result.get("passed", False))
                error_message = str(worker_result.get("error_message", "") or "")
                reason = error_message

                if not reason:
                    comparison_result = worker_result.get("comparison_result")
                    if isinstance(comparison_result, ComparisonResult):
                        reasons = list(comparison_result.reasons)
                        reason = "" if comparison_result.matches else (reasons[0] if reasons else "failed")
                    elif isinstance(comparison_result, dict):
                        reasons = list(comparison_result.get("reasons", []))
                        matches = bool(comparison_result.get("matches", False))
                        reason = "" if matches else (reasons[0] if reasons else "failed")
                    else:
                        reason = "" if passed else "failed"

                ordered_results.append(
                    {
                        "passed": passed,
                        "reason": reason,
                    }
                )

            output = {
                "results": ordered_results,
            }
            success = all(item["passed"] for item in ordered_results) if ordered_results else True
        except Exception as exc:
            output = {
                "results": [
                    {
                        "passed": False,
                        "reason": str(exc),
                    }
                ],
                "error": str(exc),
            }
            success = False
        finally:
            self._refresh_facts_db()

        return self._single_epoch_result(
            run_id=run_id,
            batch_id=batch_id,
            output=output,
            success=success,
        )

    def train(
        self,
        data: Optional[list[Data]] = None,
        queries: Optional[list[str]] = None,
        tables: Optional[list[list[pd.DataFrame]]] = None,
        outputs: Optional[list[pd.DataFrame]] = None,
        metadata: Optional[list[Dict]] = None,
        batch_size: int | str = CONFIG.batch_size,
        epochs: int = int(CONFIG.epochs),
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())

        if data is None:
            if queries is None or tables is None:
                return self._result_envelope(
                    run_id=run_id,
                    epochs={
                        "1": {
                            "batch_ids": [],
                            "output": {
                                "write_count": 0,
                                "written_ids": [],
                                "error": "Either data or both queries and tables must be provided.",
                            },
                            "epoch_success": False,
                        }
                    },
                )
            data = self._to_dataset(
                queries=queries,
                tables=tables,
                outputs=outputs,
                metadata=metadata,
            )

        batches = self._slice_batches(data, batch_size)
        if not batches:
            return self._result_envelope(
                run_id=run_id,
                epochs={
                    "1": {
                        "batch_ids": [],
                        "output": {
                            "write_count": 0,
                            "written_ids": [],
                            "error": "No data provided for training.",
                        },
                        "epoch_success": False,
                    }
                },
            )

        workflow = build_train_workflow()
        epoch_results: dict[str, dict[str, Any]] = {}

        for epoch_number in range(1, epochs + 1):
            epoch_key = str(epoch_number)
            epoch_batch_ids: list[str] = []
            epoch_written_ids: list[str] = []
            epoch_write_count = 0
            epoch_batches: list[dict[str, Any]] = []
            epoch_success = True

            for batch_index, batch_data in enumerate(batches, start=1):
                batch_id = str(batch_index)
                epoch_batch_ids.append(batch_id)

                self.shared_cache.clear_cache()

                temp_vdb_name = f"vdb-{run_id}-{epoch_number}{batch_id}"
                temp_vdb = VectorDB(name=temp_vdb_name)
                self.temp_vdbs[(run_id, epoch_number, batch_id)] = temp_vdb

                temp_library_path = self.temp_library_root / f"{run_id}:{epoch_number}:{batch_id}"
                temp_library_path.mkdir(parents=True, exist_ok=True)

                async def _run_train_batch() -> dict[str, Any]:
                    context = TrainWorkflowContext(
                        transformer_model=self.transformer_model,
                        embedding_model=self.embedding_model,
                        shared_cache=self.shared_cache,
                        concurrency=self.concurrency,
                        perm_vdb=self.permanent_vdb,
                        temp_vdb=temp_vdb,
                        perm_library_path=self.permanent_library_path,
                        temp_library_path=str(temp_library_path),
                        workflow_fact_repository=self.workflow_fact_repository,
                        run_id=run_id,
                        epoch=str(epoch_number),
                        batch_id=batch_id,
                        workflow_name="train",
                        process_name="train",
                    )
                    state = TrainWorkflowState(dataset=batch_data)
                    return await workflow.ainvoke(state, context=context)

                try:
                    result = asyncio.run(_run_train_batch())
                    merge_result = result.get("merge_write_embed_result", {})
                    write_count = int(
                        merge_result.get(
                            "vector_db_write_count",
                            merge_result.get("write_count", 0),
                        )
                    )
                    written_ids = list(
                        merge_result.get(
                            "vector_db_written_ids",
                            merge_result.get("written_ids", []),
                        )
                    )

                    epoch_write_count += write_count
                    epoch_written_ids.extend(written_ids)
                    epoch_batches.append(
                        {
                            "batch_id": batch_id,
                            "write_count": write_count,
                            "written_ids": written_ids,
                        }
                    )
                except Exception as exc:
                    epoch_success = False
                    epoch_batches.append(
                        {
                            "batch_id": batch_id,
                            "write_count": 0,
                            "written_ids": [],
                            "error": str(exc),
                        }
                    )

            epoch_results[epoch_key] = {
                "batch_ids": epoch_batch_ids,
                "output": {
                    "write_count": epoch_write_count,
                    "written_ids": epoch_written_ids,
                    "batches": epoch_batches,
                },
                "epoch_success": epoch_success,
            }

        self._refresh_facts_db()

        return self._result_envelope(
            run_id=run_id,
            epochs=epoch_results,
        )

    def predict(self, query, tables):
        raise NotImplementedError("Use inference(...) for controlled result selection.")
