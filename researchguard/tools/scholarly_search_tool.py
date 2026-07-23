# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\scholarly_search_tool.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from researchguard.tools.contracts import ToolError, ToolResult, ToolSpec
from researchguard.tools.scholarly import (
    ArxivProvider,
    OpenAlexProvider,
    ScholarPaperRecord,
    ScholarlyProvider,
    ScholarlyProviderError,
    ScholarlyProviderTimeout,
    ScholarlySearchCache,
)
from researchguard.tools.scholarly.base import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIRECTORY = PROJECT_ROOT / "data" / "cache" / "scholarly_search_v1"


class ScholarlySearchTool:
    name = "search_scholarly_sources"
    version = "1.0.0"
    description = "Discover scholarly paper metadata that must not be used as answer evidence."

    def __init__(
        self,
        *,
        providers: Mapping[str, ScholarlyProvider] | None = None,
        cache: ScholarlySearchCache | None = None,
        cache_enabled: bool = True,
        cache_directory: str | Path = DEFAULT_CACHE_DIRECTORY,
        config_version: str = "scholarly_search_v1.0",
    ):
        if providers is None:
            providers = {
                "arxiv": ArxivProvider(),
                "openalex": OpenAlexProvider(),
            }
        self.providers = {str(name).casefold(): provider for name, provider in providers.items()}
        self.cache = cache or ScholarlySearchCache(cache_directory, enabled=cache_enabled)
        self.config_version = config_version

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={
                "query": "non-empty scholarly search query",
                "sources": "optional ordered provider list; defaults to arxiv",
                "limit": "integer between 1 and 50",
                "read_cache": "boolean",
            },
        )

    def invoke(self, **kwargs: Any) -> ToolResult:
        return self.search_scholarly_sources(**kwargs)

    def search_scholarly_sources(
        self,
        query: str,
        *,
        sources: Iterable[str] | None = None,
        limit: int = 10,
        read_cache: bool = True,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            normalized_query = normalize_text(query)
            if not normalized_query:
                raise ValueError("Scholarly search query must not be empty.")
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
                raise ValueError("limit must be an integer between 1 and 50.")
            requested_sources = self._normalize_sources(sources)
        except (TypeError, ValueError) as exc:
            return self._failure(
                started,
                exc,
                category="invalid_input",
                code="invalid_scholarly_search_input",
                retryable=False,
            )

        records: list[ScholarPaperRecord] = []
        completed_sources: list[str] = []
        cache_hits: dict[str, bool] = {}
        provider_errors: list[dict[str, Any]] = []
        for source in requested_sources:
            provider = self.providers[source]
            request = {
                "query": normalized_query,
                "provider": source,
                "provider_version": provider.version,
                "config_version": self.config_version,
                "limit": limit,
            }
            key = self.cache.make_key(
                query=normalized_query,
                provider=source,
                config_version=self.config_version,
                limit=limit,
            )
            cached = self.cache.get(key, request=request) if read_cache else None
            if cached is not None:
                provider_records = cached
                cache_hits[source] = True
                completed_sources.append(source)
            else:
                cache_hits[source] = False
                try:
                    provider_records = provider.search(normalized_query, limit=limit)
                    if not isinstance(provider_records, list) or not all(
                        isinstance(record, ScholarPaperRecord) for record in provider_records
                    ):
                        raise ScholarlyProviderError(
                            f"{source} returned records outside ScholarPaperRecord schema."
                        )
                except ScholarlyProviderError as exc:
                    provider_errors.append(
                        {
                            "provider": source,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "retryable": isinstance(exc, ScholarlyProviderTimeout),
                        }
                    )
                    continue
                self.cache.set(key, request=request, records=provider_records)
                completed_sources.append(source)
            records.extend(provider_records)

        deduplicated = self._deduplicate(records)[:limit]
        latency_ms = (time.perf_counter() - started) * 1000.0
        data = {
            "query": normalized_query,
            "candidate_papers": [record.to_dict() for record in deduplicated],
            "candidate_count": len(deduplicated),
            "providers_requested": requested_sources,
            "providers_completed": completed_sources,
            "provider_errors": provider_errors,
            "cache_hits": cache_hits,
            "config_version": self.config_version,
            "metadata_only": True,
            "evidence_eligible": False,
        }
        if not completed_sources:
            timeout_only = bool(provider_errors) and all(
                error["error_type"] == "ScholarlyProviderTimeout" for error in provider_errors
            )
            error = ToolError(
                code="scholarly_provider_timeout" if timeout_only else "scholarly_provider_failure",
                category="timeout" if timeout_only else "api_failure",
                message="All requested scholarly providers failed.",
                retryable=timeout_only,
                details={"provider_errors": provider_errors},
            )
            return ToolResult.create(
                status="failed",
                message="Scholarly discovery failed.",
                reason=error.code,
                tool_name=self.name,
                tool_version=self.version,
                latency_ms=latency_ms,
                data=data,
                error=error,
            )
        return ToolResult.create(
            status="success",
            message=f"Discovered {len(deduplicated)} candidate papers.",
            reason="partial_provider_failure" if provider_errors else None,
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=latency_ms,
            data=data,
        )

    def _normalize_sources(self, sources: Iterable[str] | None) -> list[str]:
        if sources is None:
            requested = ["arxiv"]
        elif isinstance(sources, str):
            requested = [sources.casefold().strip()]
        else:
            requested = [str(source).casefold().strip() for source in sources]
        requested = [source for source in requested if source]
        if not requested:
            requested = ["arxiv"]
        deduplicated = list(dict.fromkeys(requested))
        unknown = [source for source in deduplicated if source not in self.providers]
        if unknown:
            raise ValueError(f"Unknown scholarly providers: {', '.join(unknown)}")
        return deduplicated

    @staticmethod
    def _deduplicate(records: Iterable[ScholarPaperRecord]) -> list[ScholarPaperRecord]:
        deduplicated: list[ScholarPaperRecord] = []
        seen: set[str] = set()
        for record in records:
            doi_key = normalize_text(record.doi).casefold()
            title_key = normalize_text(record.title).casefold()
            key = f"doi:{doi_key}" if doi_key else f"title:{title_key}|year:{record.year}"
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(record)
        return deduplicated

    def _failure(
        self,
        started: float,
        exc: Exception,
        *,
        category: str,
        code: str,
        retryable: bool,
    ) -> ToolResult:
        return ToolResult.create(
            status="failed",
            message="Scholarly discovery failed.",
            reason=code,
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=ToolError(
                code=code,
                category=category,
                message=str(exc),
                retryable=retryable,
                details={"exception_type": type(exc).__name__},
            ),
        )
