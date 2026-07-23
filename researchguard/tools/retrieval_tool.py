# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\retrieval_tool.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from researchguard.pipeline import DEFAULT_CONFIG_PATH, PipelineSettings, load_pipeline_settings
from researchguard.retrieval.filters import MetadataFilter
from researchguard.retrieval.models import RetrievalError
from researchguard.retrieval.retrieval_v1 import RetrievalEngine
from researchguard.tools.contracts import EvidenceRecord, ToolError, ToolResult, ToolSpec


class RetrievalTool:
    name = "retrieve_evidence"
    version = "1.0.0"
    description = "Retrieve and rerank canonical ResearchGuard evidence for a query."

    def __init__(
        self,
        *,
        engine: Any | None = None,
        settings: PipelineSettings | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self._engine = engine
        self._settings = settings
        self._config_path = config_path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={
                "query": "non-empty string",
                "filters": "optional metadata filter mapping",
                "top_k": "optional positive integer",
                "candidate_k": "optional positive integer",
                "read_cache": "boolean",
            },
        )

    def _pipeline_settings(self) -> PipelineSettings:
        if self._settings is None:
            _, self._settings = load_pipeline_settings(self._config_path)
        return self._settings

    def _retrieval_engine(self) -> Any:
        if self._engine is None:
            settings = self._pipeline_settings()
            self._engine = RetrievalEngine.from_config(settings.retrieval_config_path)
        return self._engine

    def invoke(self, **kwargs: Any) -> ToolResult:
        return self.retrieve_evidence(**kwargs)

    def retrieve_evidence(
        self,
        query: str,
        *,
        filters: MetadataFilter | Mapping[str, Any] | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        read_cache: bool = True,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            normalized_query = str(query).strip()
            if not normalized_query:
                raise ValueError("Query must not be empty.")
            if top_k is not None and top_k < 1:
                raise ValueError("top_k must be a positive integer.")
            if candidate_k is not None and candidate_k < 1:
                raise ValueError("candidate_k must be a positive integer.")
            if isinstance(filters, Mapping):
                filters = MetadataFilter.from_mapping(filters)
            elif filters is not None and not isinstance(filters, MetadataFilter):
                raise TypeError("filters must be a MetadataFilter or mapping.")

            settings = self._pipeline_settings()
            response = self._retrieval_engine().retrieve(
                normalized_query,
                mode=settings.retrieval_mode,
                top_k=top_k or settings.retrieval_top_k,
                candidate_k=candidate_k or settings.retrieval_candidate_k,
                filters=filters,
                rerank=settings.reranker_enabled,
                rerank_candidate_k=settings.reranker_candidate_k,
                rerank_read_cache=read_cache,
                rewrite=settings.rewrite_enabled,
                multi_query=settings.multi_query_enabled,
                rewrite_read_cache=read_cache,
            )
            evidence = [EvidenceRecord.from_retrieval_hit(hit) for hit in response.hits]
            response_data = response.to_dict(include_text=False)
            response_data.pop("hits", None)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return ToolResult.create(
                status="success",
                message=f"Retrieved {len(evidence)} evidence records.",
                tool_name=self.name,
                tool_version=self.version,
                latency_ms=latency_ms,
                data={
                    "query": response.query,
                    "evidence": [record.to_dict() for record in evidence],
                    "ranking": [
                        {
                            "rank": record.rank,
                            "chunk_id": record.chunk_id,
                            "doc_id": record.doc_id,
                            "score": record.score,
                        }
                        for record in evidence
                    ],
                    "retrieval": response_data,
                },
            )
        except (ValueError, TypeError) as exc:
            return self._failure(started, exc, "invalid_input", "invalid_retrieval_input", False)
        except TimeoutError as exc:
            return self._failure(started, exc, "timeout", "retrieval_timeout", True)
        except RetrievalError as exc:
            return self._failure(started, exc, "retrieval_failure", "retrieval_failed", True)
        except Exception as exc:
            return self._failure(started, exc, "retrieval_failure", "retrieval_failed", False)

    def _failure(
        self,
        started: float,
        exc: Exception,
        category: str,
        code: str,
        retryable: bool,
    ) -> ToolResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        error = ToolError(
            code=code,
            category=category,
            message=str(exc),
            retryable=retryable,
            details={"exception_type": type(exc).__name__},
        )
        return ToolResult.create(
            status="failed",
            message="Evidence retrieval failed.",
            reason=code,
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=latency_ms,
            error=error,
        )
