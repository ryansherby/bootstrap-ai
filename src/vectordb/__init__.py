from dataclasses import dataclass
import json
from pathlib import Path
import random
import string
from typing import Literal

import faiss
import numpy as np

from configs import CONFIG

from src.types import FileMetadata





class VectorDB:
    def __init__(
        self,
        name: str | None = None,
        map_type: Literal["IDMap", "IVFFlat"] = "IDMap",
        nprobe: int | None = None,
    ):
        if map_type not in {"IDMap", "IVFFlat"}:
            raise ValueError("map_type must be 'IDMap' or 'IVFFlat'")

        if map_type == "IVFFlat":
            if nprobe is None:
                nprobe = CONFIG.vector_db.nprobe
            if nprobe <= 0:
                raise ValueError("nprobe must be greater than 0")

        if name is None:
            rand = "".join(random.choices(string.digits, k=8))
            name = f"vdb-{rand}"

        self.name = name
        self.map_type = map_type

        self.path = Path(CONFIG.vectordb_path) / "dbs" / self.name
        self.index_file = self.path / "index.faiss"
        self.metadata_file = self.path / "metadata.jsonl"
        self.manifest_file = self.path / "manifest.json"

        self.index = None
        self.dim = None
        self.metadata_by_faiss_id: dict[int, dict] = {}
        self.nprobe = nprobe if map_type == "IVFFlat" else None
        self.size = 0
        self.max_size = CONFIG.vector_db.max_size
        self.next_faiss_id = 0

    @classmethod
    def load(cls, path: str | Path) -> "VectorDB":
        path = Path(path)

        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"VectorDB path does not exist: {path}")

        manifest_file = path / "manifest.json"
        metadata_file = path / "metadata.jsonl"
        index_file = path / "index.faiss"

        if not manifest_file.exists():
            raise FileNotFoundError(f"Missing manifest file: {manifest_file}")

        with manifest_file.open("r") as f:
            manifest = json.load(f)

        obj = cls(
            name=manifest["name"],
            map_type=manifest["map_type"],
            nprobe=manifest.get("nprobe"),
        )
        obj.path = path
        obj.index_file = index_file
        obj.metadata_file = metadata_file
        obj.manifest_file = manifest_file
        obj.next_faiss_id = manifest.get("next_faiss_id", 0)

        if metadata_file.exists():
            with metadata_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    faiss_id = int(record["faiss_id"])
                    metadata = record["metadata"]

                    obj.metadata_by_faiss_id[faiss_id] = metadata

        if index_file.exists() and index_file.stat().st_size > 0:
            obj.index = faiss.read_index(str(index_file))
            obj.dim = obj.index.d
            obj.size = obj.index.ntotal
            
        if (
            obj.index is not None
            and obj.map_type == "IVFFlat"
            and obj.nprobe is not None
        ):
            obj.index.nprobe = min(
                obj.nprobe,
                obj.index.nlist,
            )

        return obj

    def _write_manifest(self):
        self.path.mkdir(parents=True, exist_ok=True)
        with self.manifest_file.open("w") as f:
            json.dump(
                {
                    "name": self.name,
                    "map_type": self.map_type,
                    "nprobe": self.nprobe if hasattr(self, "nprobe") else None,
                    "next_faiss_id": self.next_faiss_id if hasattr(self, "next_faiss_id") else 0,
                },
                f,
            )

    def _create_index(self, dim: int, embeddings: np.ndarray | None = None):
        if self.map_type == "IDMap":
            base_index = faiss.IndexFlatIP(dim)
            return faiss.IndexIDMap2(base_index)

        if self.map_type == "IVFFlat":
            if embeddings is None:
                raise ValueError("embeddings are required to build an IVFFlat index")

            nlist = max(1, min(int(np.sqrt(len(embeddings))), len(embeddings)))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            return index

        raise ValueError(f"Unsupported map_type: {self.map_type}")

    def build_index(
        self,
        embeddings,
        metadatas: list[FileMetadata],
    ):
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, D)")

        n, d = embeddings.shape

        if len(metadatas) != n:
            raise ValueError(
                "len(metadatas) must match number of embeddings"
            )

        # FileMetadata.id is a string, so create separate
        # integer IDs for FAISS.
        faiss_ids = np.arange(n, dtype=np.int64)
        self.next_faiss_id = n

        if self.path.exists():
            for file in self.path.iterdir():
                file.unlink()
        else:
            self.path.mkdir(parents=True, exist_ok=True)

        self.metadata_by_faiss_id = {}
        self.dim = d

        self.index = self._create_index(
            d,
            embeddings=embeddings,
        )

        if self.map_type == "IVFFlat":
            self.index.nprobe = min(
                self.nprobe,
                self.index.nlist,
            )

        self.index.add_with_ids(
            embeddings,
            faiss_ids,
        )

        self.size = n

        faiss.write_index(
            self.index,
            str(self.index_file),
        )

        self._write_manifest()

        with self.metadata_file.open("w") as f:
            for faiss_id, metadata in zip(
                faiss_ids,
                metadatas,
            ):
                obj = metadata.dict()

                # Store the FAISS ID alongside the user's
                # string metadata ID.
                record = {
                    "faiss_id": int(faiss_id),
                    "metadata": obj,
                }

                self.metadata_by_faiss_id[
                    int(faiss_id)
                ] = obj

                f.write(json.dumps(record) + "\n")

    def add(self, embeddings, metadatas: list[FileMetadata]):
        """
        Add vectors to an IDMap index.
        If the index is not built yet, build it first.

        IVFFlat does not support add() in this class; use build_index() instead.
        """
        if self.map_type == "IVFFlat":
            raise ValueError("IVFFlat does not support add(); use build_index() to rebuild the index.")

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, D)")

        n, d = embeddings.shape

        if len(metadatas) != n:
            raise ValueError("len(metadatas) must match number of embeddings")

        if self.index is None:
            self.build_index(embeddings, metadatas)
            return

        if d != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {d}")

        faiss_ids = np.arange(
            self.next_faiss_id,
            self.next_faiss_id + n,
            dtype=np.int64,
        )

        self.next_faiss_id += n
        self.index.add_with_ids(embeddings, faiss_ids)
        faiss.write_index(self.index, str(self.index_file))

        self.path.mkdir(parents=True, exist_ok=True)
        self._write_manifest()
        self.metadata_file.touch(exist_ok=True)

        with self.metadata_file.open("a") as f:
            for i, metadata in enumerate(metadatas):
                obj = metadata.dict()
                self.metadata_by_faiss_id[int(faiss_ids[i])] = obj
                f.write(json.dumps({
                    "faiss_id": int(faiss_ids[i]),
                    "metadata": obj,
                }) + "\n")
                
        self.size += len(embeddings)

    def search(
    self,
    query_embeddings,
    top_k: int = 5,
    search_type: str = "basic",
    similarity_threshold: float | None = None,
    ):
        if search_type not in {"basic", "mmr"}:
            raise ValueError(
                "search_type must be either 'basic' or 'mmr'"
            )

        if search_type == "mmr" and self.map_type == "IVFFlat":
            raise ValueError(
                "MMR search is not supported for IVFFlat indices"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if (
            similarity_threshold is not None
            and not 0 <= similarity_threshold <= 1
        ):
            raise ValueError(
                "similarity_threshold must be between 0 and 1"
            )

        query_embeddings = np.asarray(
            query_embeddings,
            dtype=np.float32,
        )

        if query_embeddings.ndim != 2:
            raise ValueError(
                "query_embeddings must have shape (Q, D)"
            )

        if self.index is None:
            if (
                self.index_file.exists()
                and self.index_file.stat().st_size > 0
            ):
                self.index = faiss.read_index(
                    str(self.index_file)
                )

                self.dim = self.index.d
                self.size = self.index.ntotal

                if (
                    self.map_type == "IVFFlat"
                    and self.nprobe is not None
                ):
                    self.index.nprobe = min(
                        self.nprobe,
                        self.index.nlist,
                    )

            else:
                return [
                    []
                    for _ in range(len(query_embeddings))
                ]

        if query_embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected query dim {self.dim}, "
                f"got {query_embeddings.shape[1]}"
            )

        if search_type == "mmr":
            n_candidates = (
                top_k * CONFIG.vector_db.mmr_scaler
            )
        else:
            n_candidates = top_k

        n_candidates = min(
            n_candidates,
            self.index.ntotal,
        )

        if n_candidates == 0:
            return [
                []
                for _ in range(len(query_embeddings))
            ]

        scores, indices = self.index.search(
            query_embeddings,
            n_candidates,
        )

        results = []

        for q_idx in range(len(query_embeddings)):
            # Remove invalid FAISS results.
            valid_mask = indices[q_idx] != -1

            local_indices = (
                indices[q_idx][valid_mask].copy()
            )

            local_scores = (
                scores[q_idx][valid_mask].copy()
            )

            # ---------------------------------
            # Similarity threshold filtering
            # ---------------------------------
            if similarity_threshold is not None:
                # Convert raw cosine similarity [-1, 1]
                # into the public score range [0, 1].
                normalized_scores = (
                    local_scores + 1.0
                ) / 2.0

                threshold_mask = (
                    normalized_scores
                    >= similarity_threshold
                )

                local_indices = (
                    local_indices[threshold_mask]
                )

                local_scores = (
                    local_scores[threshold_mask]
                )

            if len(local_indices) == 0:
                results.append([])
                continue

            # ---------------------------------
            # MMR reranking
            # ---------------------------------
            if search_type == "mmr":
                candidate_embeddings = np.stack(
                    [
                        self.index.reconstruct(
                            int(idx)
                        )
                        for idx in local_indices
                    ]
                )

                _, selected_positions = self._mmr(
                    query_scores=local_scores,
                    doc_embs=candidate_embeddings,
                    k=top_k,
                )

                selected_positions = np.asarray(
                    selected_positions,
                    dtype=np.int64,
                )

                local_indices = (
                    local_indices[selected_positions]
                )

                local_scores = (
                    local_scores[selected_positions]
                )

            # ---------------------------------
            # Convert cosine similarity
            # [-1, 1] -> [0, 1]
            # ---------------------------------
            scores_01 = np.clip(
                (local_scores + 1.0) / 2.0,
                0.0,
                1.0,
            )

            result = []

            for score, faiss_id in zip(
                scores_01,
                local_indices,
            ):
                metadata = self.metadata_by_faiss_id.get(
                    int(faiss_id)
                )

                if metadata is not None:
                    metadata_with_mode = {
                        **metadata,
                        "mode": metadata.get("mode", ""),
                    }
                    result.append(
                        {
                            "metadata": FileMetadata(
                                **metadata_with_mode
                            ),
                            # Original retrieval
                            # similarity, not MMR score.
                            "score": float(score),
                        }
                    )

            results.append(result)

        return results
    
    
    def _mmr(
        self,
        query_scores: np.ndarray,
        doc_embs: np.ndarray,
        k: int,
    ):
        if len(doc_embs) == 0:
            return [], []

        if len(query_scores) != len(doc_embs):
            raise ValueError(
                "query_scores and doc_embs must contain "
                "the same number of candidates"
            )

        mmr_lambda = CONFIG.vector_db.mmr_lambda

        if not 0 <= mmr_lambda <= 1:
            raise ValueError(
                "mmr_lambda must be between 0 and 1"
            )

        # Pairwise document cosine similarities assuming
        # doc_embs have already been L2 normalized.
        doc_sims = np.dot(doc_embs, doc_embs.T)

        selected = []
        mmr_scores = []

        candidate_idxs = list(range(len(doc_embs)))

        # Standard MMR starts with the most query-relevant document.
        first = int(np.argmax(query_scores))

        selected.append(first)
        mmr_scores.append(float(query_scores[first]))
        candidate_idxs.remove(first)

        while (
            len(selected) < min(k, len(doc_embs))
            and candidate_idxs
        ):
            best_idx = None
            best_score = -float("inf")

            for i in candidate_idxs:
                redundancy = max(
                    doc_sims[i, j]
                    for j in selected
                )

                mmr_score = (
                    mmr_lambda * query_scores[i]
                    - (1 - mmr_lambda) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(best_idx)
            candidate_idxs.remove(best_idx)
            mmr_scores.append(float(best_score))

        return mmr_scores, selected
        
        
        

    def delete(self):
        if self.path.exists():
            for file in self.path.iterdir():
                file.unlink()
            self.path.rmdir()

        self.index = None
        self.dim = None
        self.metadata_by_faiss_id = {}
        self.size = 0