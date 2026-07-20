# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\query_rewrite_pipeline.py
from __future__ import annotations

import time
from typing import Any

from researchguard.retrieval.query_rewriter import (
    FORBIDDEN_IDENTIFIER_RE,
    BackendRewrite,
    OpenAIQueryRewriteBackend,
    QueryAnalysis,
    QueryRewriteBackend,
    QueryRewriteBackendError,
    QueryRewriteResult,
    QueryRewriteSettings,
    analyze_query,
    missing_entities,
    missing_constraints,
    normalize_query_text,
    utc_timestamp,
)
from researchguard.retrieval.rewrite_cache import QueryRewriteCache


class QueryRewritePipeline:
    def __init__(
        self,
        settings: QueryRewriteSettings,
        *,
        backend: QueryRewriteBackend | None = None,
        cache: QueryRewriteCache | None = None,
    ):
        self.settings = settings
        self.backend = backend or OpenAIQueryRewriteBackend(settings)
        self.cache = cache or QueryRewriteCache(settings.cache_directory, enabled=settings.cache_enabled)

    def rewrite(self, query: str, *, read_cache: bool = True) -> QueryRewriteResult:
        started = time.perf_counter()
        analysis = analyze_query(query)
        key = self.cache.make_key(original_query=analysis.normalized_input, settings=self.settings)
        cached = self.cache.get(key) if read_cache else None
        if cached is not None:
            result = self._result_from_cache(cached, analysis)
            if result is not None:
                return QueryRewriteResult(
                    **{
                        **result.to_dict(),
                        "expansion_queries": result.expansion_queries,
                        "preserved_entities": result.preserved_entities,
                        "preserved_constraints": result.preserved_constraints,
                        "dropped_expansion_reasons": result.dropped_expansion_reasons,
                        "cache_hit": True,
                        "api_call_count": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )

        try:
            backend_result = self.backend.rewrite(analysis)
            result = self._validated_result(analysis, backend_result, started=started)
        except QueryRewriteBackendError as exc:
            result = self._fallback_result(
                analysis,
                reason=f"backend_failure:{type(exc.__cause__ or exc).__name__}",
                api_call_count=exc.api_call_count,
                started=started,
            )
        except Exception as exc:
            result = self._fallback_result(
                analysis,
                reason=f"rewrite_failure:{type(exc).__name__}",
                api_call_count=0,
                started=started,
            )
        self.cache.put(key, result.to_dict())
        return result

    def _validated_result(
        self,
        analysis: QueryAnalysis,
        backend_result: BackendRewrite,
        *,
        started: float,
    ) -> QueryRewriteResult:
        normalized = normalize_query_text(backend_result.normalized_query)
        expansions = [normalize_query_text(item) for item in backend_result.expansion_queries]
        expansions = [item for item in expansions if item][: self.settings.max_expansions]
        if not normalized:
            return self._fallback_result(
                analysis,
                reason="empty_rewrite",
                api_call_count=backend_result.api_call_count,
                input_tokens=backend_result.input_tokens,
                output_tokens=backend_result.output_tokens,
                started=started,
            )

        if FORBIDDEN_IDENTIFIER_RE.search(normalized):
            return self._fallback_result(
                analysis,
                reason="forbidden_identifier",
                api_call_count=backend_result.api_call_count,
                input_tokens=backend_result.input_tokens,
                output_tokens=backend_result.output_tokens,
                started=started,
            )
        if missing_entities(normalized, analysis.preserved_entities) or missing_constraints(
            normalized,
            analysis.preserved_constraints,
        ):
            return self._fallback_result(
                analysis,
                reason="entity_preservation_failure",
                api_call_count=backend_result.api_call_count,
                input_tokens=backend_result.input_tokens,
                output_tokens=backend_result.output_tokens,
                started=started,
            )

        deduplicated: list[str] = []
        dropped_reasons: list[str] = []
        seen = {normalized.casefold(), analysis.normalized_input.casefold()}
        for index, expansion in enumerate(expansions, start=1):
            if FORBIDDEN_IDENTIFIER_RE.search(expansion):
                dropped_reasons.append(f"expansion_{index}:forbidden_identifier")
                continue
            missing = missing_entities(expansion, analysis.preserved_entities)
            if missing:
                dropped_reasons.append(f"expansion_{index}:missing_entities:{','.join(missing)}")
                continue
            key = expansion.casefold()
            if key not in seen:
                seen.add(key)
                deduplicated.append(expansion)
            else:
                dropped_reasons.append(f"expansion_{index}:duplicate_query")
        return QueryRewriteResult(
            original_query=analysis.original_query,
            normalized_query=normalized,
            expansion_queries=tuple(deduplicated),
            preserved_entities=analysis.preserved_entities,
            preserved_constraints=analysis.preserved_constraints,
            dropped_expansion_reasons=tuple(dropped_reasons),
            fallback_used=False,
            fallback_reason=None,
            model=self.settings.model,
            prompt_version=self.settings.prompt_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            api_call_count=backend_result.api_call_count,
            input_tokens=backend_result.input_tokens,
            output_tokens=backend_result.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _fallback_result(
        self,
        analysis: QueryAnalysis,
        *,
        reason: str,
        api_call_count: int,
        started: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=analysis.original_query,
            normalized_query=analysis.normalized_input,
            expansion_queries=(),
            preserved_entities=analysis.preserved_entities,
            preserved_constraints=analysis.preserved_constraints,
            dropped_expansion_reasons=(),
            fallback_used=True,
            fallback_reason=reason,
            model=self.settings.model,
            prompt_version=self.settings.prompt_version,
            timestamp=utc_timestamp(),
            cache_hit=False,
            api_call_count=api_call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _result_from_cache(
        self,
        payload: dict[str, Any],
        analysis: QueryAnalysis,
    ) -> QueryRewriteResult | None:
        try:
            result = QueryRewriteResult(
                original_query=str(payload["original_query"]),
                normalized_query=str(payload["normalized_query"]),
                expansion_queries=tuple(str(item) for item in payload.get("expansion_queries", [])),
                preserved_entities=tuple(str(item) for item in payload.get("preserved_entities", [])),
                preserved_constraints=tuple(str(item) for item in payload.get("preserved_constraints", [])),
                dropped_expansion_reasons=tuple(str(item) for item in payload.get("dropped_expansion_reasons", [])),
                fallback_used=bool(payload.get("fallback_used", False)),
                fallback_reason=payload.get("fallback_reason"),
                model=str(payload["model"]),
                prompt_version=str(payload["prompt_version"]),
                timestamp=str(payload["timestamp"]),
                cache_hit=True,
                api_call_count=0,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
            )
        except (KeyError, TypeError, ValueError):
            return None
        variants = [result.normalized_query, *result.expansion_queries]
        if normalize_query_text(result.original_query) != analysis.normalized_input:
            return None
        if result.preserved_entities != analysis.preserved_entities:
            return None
        if result.preserved_constraints != analysis.preserved_constraints:
            return None
        if result.model != self.settings.model or result.prompt_version != self.settings.prompt_version:
            return None
        if len(result.expansion_queries) > self.settings.max_expansions:
            return None
        dedupe_keys = [normalize_query_text(item).casefold() for item in variants]
        if any(not key for key in dedupe_keys) or len(dedupe_keys) != len(set(dedupe_keys)):
            return None
        if not result.normalized_query or any(
            missing_entities(item, analysis.preserved_entities) for item in variants
        ) or missing_constraints(result.normalized_query, analysis.preserved_constraints):
            return None
        if any(FORBIDDEN_IDENTIFIER_RE.search(item) for item in variants):
            return None
        return result
