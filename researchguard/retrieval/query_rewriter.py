# C:\Users\18449\Desktop\researchguard_workspace\researchguard\retrieval\query_rewriter.py
from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from researchguard.indexing.corpus_loader import load_yaml
from researchguard.retrieval.models import RetrievalError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IDENTIFIER_RE = re.compile(r"\b[a-z0-9_]+_chunk_\d+\b", re.IGNORECASE)
REFERENCE_RE = re.compile(
    r"\b(?:table|figure|fig\.?|equation|eq\.?)\s*[A-Za-z]?\d+(?:[.\-]\d+)*\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s*%")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[kKmMbB])?(?![\w.])")
TOKEN_RE = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9]*(?:[-./][A-Za-z0-9]+)*(?!\w)")
NEGATION_RE = re.compile(r"\b(?:not|no|never|without|exclude|excluding|only|neither|nor)\b", re.IGNORECASE)
TEMPORAL_RE = re.compile(
    r"\b(?:before|after|since|until|during|latest|earliest|newer|older|prior|subsequent)\b",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(
    r"\b(?:versus|vs\.?|compare|comparison|differences?|differ|different|distinguish|distinction|contrast|higher|lower|better|worse|more|less|than)\b",
    re.IGNORECASE,
)
TITLECASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "compare",
    "does",
    "find",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "return",
    "retrieve",
    "the",
    "to",
    "what",
    "where",
    "which",
    "with",
}


class QueryRewriteError(RetrievalError):
    pass


class QueryRewriteBackendError(QueryRewriteError):
    def __init__(self, message: str, *, api_call_count: int):
        super().__init__(message)
        self.api_call_count = api_call_count


@dataclass(frozen=True)
class QueryRewriteSettings:
    enabled: bool
    backend: str
    model: str
    temperature: float
    max_rewrites: int
    timeout: float
    max_retries: int
    prompt_version: str
    entity_rules_version: str
    cache_enabled: bool
    cache_directory: Path
    multi_query_rrf_k: float

    @property
    def max_expansions(self) -> int:
        return max(0, self.max_rewrites - 1)

    @property
    def cache_identity(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "temperature": self.temperature,
            "max_rewrites": self.max_rewrites,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "prompt_version": self.prompt_version,
            "entity_rules_version": self.entity_rules_version,
        }


@dataclass(frozen=True)
class QueryAnalysis:
    original_query: str
    normalized_input: str
    preserved_entities: tuple[str, ...]
    preserved_constraints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_input": self.normalized_input,
            "preserved_entities": list(self.preserved_entities),
            "preserved_constraints": list(self.preserved_constraints),
        }


@dataclass(frozen=True)
class BackendRewrite:
    normalized_query: str
    expansion_queries: tuple[str, ...]
    api_call_count: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    normalized_query: str
    expansion_queries: tuple[str, ...]
    preserved_entities: tuple[str, ...]
    preserved_constraints: tuple[str, ...]
    dropped_expansion_reasons: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str | None
    model: str
    prompt_version: str
    timestamp: str
    cache_hit: bool
    api_call_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "expansion_queries": list(self.expansion_queries),
            "preserved_entities": list(self.preserved_entities),
            "preserved_constraints": list(self.preserved_constraints),
            "dropped_expansion_reasons": list(self.dropped_expansion_reasons),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp,
            "cache_hit": self.cache_hit,
            "api_call_count": self.api_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_query_rewrite_settings(path: str | Path) -> tuple[dict[str, Any], QueryRewriteSettings]:
    config = load_yaml(resolve_project_path(path))
    rewrite = config.get("query_rewrite", {}) or {}
    multi_query = config.get("multi_query", {}) or {}
    settings = QueryRewriteSettings(
        enabled=bool(rewrite.get("enabled", False)),
        backend=str(rewrite.get("backend", "openai")),
        model=str(rewrite.get("model", "gpt-4.1-mini")),
        temperature=float(rewrite.get("temperature", 0)),
        max_rewrites=max(1, min(3, int(rewrite.get("max_rewrites", 3)))),
        timeout=max(1.0, float(rewrite.get("timeout", 30))),
        max_retries=max(0, int(rewrite.get("max_retries", 2))),
        prompt_version=str(rewrite.get("prompt_version", "query_rewrite_v1.2")),
        entity_rules_version=str(rewrite.get("entity_rules_version", "entity_preservation_v1.3")),
        cache_enabled=bool(rewrite.get("cache_enabled", True)),
        cache_directory=resolve_project_path(rewrite.get("cache_directory", "data/cache/query_rewrite_v1")),
        multi_query_rrf_k=max(1.0, float(multi_query.get("rrf_k", 60))),
    )
    if settings.backend != "openai":
        raise QueryRewriteError(f"Unsupported query rewrite backend: {settings.backend}")
    if settings.temperature != 0:
        raise QueryRewriteError("Query Rewrite v1 requires temperature=0 for deterministic generation.")
    return config, settings


def normalize_query_text(query: str) -> str:
    text = " ".join(str(query or "").split()).strip()
    return text


def _entity_key(value: str) -> str:
    return " ".join(value.casefold().split())


def extract_preserved_entities(query: str) -> tuple[str, ...]:
    text = normalize_query_text(query)
    values: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []

    def add_match(match: re.Match[str]) -> None:
        start, end = match.span()
        value = " ".join(match.group(0).split()).strip()
        if not value or any(start >= left and end <= right for left, right in occupied):
            return
        values.append((start, value))
        occupied.append((start, end))

    for pattern in (REFERENCE_RE, PERCENT_RE, NUMBER_RE):
        for match in pattern.finditer(text):
            add_match(match)
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        uppercase_count = sum(character.isupper() for character in token)
        if token.casefold() in TITLECASE_STOPWORDS:
            continue
        if uppercase_count >= 2 or any(character.isdigit() for character in token):
            add_match(match)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for _, value in sorted(values, key=lambda item: (item[0], -len(item[1]))):
        key = _entity_key(value)
        if key not in seen:
            seen.add(key)
            deduplicated.append(value)
    return tuple(deduplicated)


def analyze_query(query: str) -> QueryAnalysis:
    original = str(query or "")
    normalized = normalize_query_text(original)
    if not normalized:
        raise QueryRewriteError("Query must not be empty.")
    return QueryAnalysis(
        original_query=original,
        normalized_input=normalized,
        preserved_entities=extract_preserved_entities(normalized),
        preserved_constraints=extract_preserved_constraints(normalized),
    )


def missing_entities(query: str, entities: tuple[str, ...]) -> list[str]:
    haystack = _entity_key(query)
    return [entity for entity in entities if _entity_key(entity) not in haystack]


def extract_preserved_constraints(query: str) -> tuple[str, ...]:
    constraints: list[str] = []
    constraints.extend(f"negation:{match.group(0)}" for match in NEGATION_RE.finditer(query))
    constraints.extend(f"temporal:{match.group(0)}" for match in TEMPORAL_RE.finditer(query))
    if COMPARISON_RE.search(query):
        constraints.append("comparison")
    deduplicated: list[str] = []
    seen: set[str] = set()
    for constraint in constraints:
        key = constraint.casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append(constraint)
    return tuple(deduplicated)


def missing_constraints(query: str, constraints: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for constraint in constraints:
        kind, _, value = constraint.partition(":")
        if kind == "comparison":
            if not COMPARISON_RE.search(query):
                missing.append(constraint)
        elif kind == "negation":
            if not NEGATION_RE.search(query):
                missing.append(constraint)
        elif kind == "temporal" and _entity_key(value) not in _entity_key(query):
            missing.append(constraint)
    return missing


def build_rewrite_model_input(analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        "original_query": analysis.normalized_input,
        "preserved_entities": list(analysis.preserved_entities),
        "preserved_constraints": list(analysis.preserved_constraints),
    }


class QueryRewriteBackend(ABC):
    @abstractmethod
    def rewrite(self, analysis: QueryAnalysis) -> BackendRewrite:
        raise NotImplementedError


class OpenAIQueryRewriteBackend(QueryRewriteBackend):
    def __init__(self, settings: QueryRewriteSettings):
        self.settings = settings
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise QueryRewriteBackendError("OPENAI_API_KEY is missing.", api_call_count=0)
        self._client = OpenAI(api_key=api_key, timeout=self.settings.timeout, max_retries=0)
        return self._client

    def rewrite(self, analysis: QueryAnalysis) -> BackendRewrite:
        schema = {
            "type": "object",
            "properties": {
                "normalized_query": {"type": "string"},
                "expansion_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": self.settings.max_expansions,
                },
            },
            "required": ["normalized_query", "expansion_queries"],
            "additionalProperties": False,
        }
        instructions = (
            "You normalize and expand a scientific-paper retrieval query. Return one concise normalized query and "
            f"at most {self.settings.max_expansions} complementary expansion queries. Preserve every supplied entity "
            "verbatim in every output query, including model names, dataset names, numbers, table/equation references, "
            "and acronyms. The normalized query must preserve every supplied negation and temporal constraint and use an "
            "explicit comparison term when requested. Expansions may focus sub-aspects but must not reverse those "
            "constraints. Do not answer the query. Do not invent paper "
            "names, findings, numeric claims, chunk IDs, labels, or facts not present in the original query. Expansions "
            "may add generic scientific retrieval terminology or split emphasis, but must preserve the original intent."
        )
        payload = json.dumps(build_rewrite_model_input(analysis), ensure_ascii=False)
        last_error: Exception | None = None
        call_count = 0
        for attempt in range(self.settings.max_retries + 1):
            call_count += 1
            try:
                response = self._get_client().responses.create(
                    model=self.settings.model,
                    instructions=instructions,
                    input=payload,
                    temperature=self.settings.temperature,
                    max_output_tokens=400,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "query_rewrite_v1",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    store=False,
                )
                parsed = json.loads(response.output_text)
                normalized_query = str(parsed["normalized_query"])
                expansions = tuple(str(item) for item in parsed["expansion_queries"])
                usage = getattr(response, "usage", None)
                return BackendRewrite(
                    normalized_query=normalized_query,
                    expansion_queries=expansions,
                    api_call_count=call_count,
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(min(2**attempt, 4))
        raise QueryRewriteBackendError(
            f"Rewrite API failed after retries: {type(last_error).__name__}: {last_error}",
            api_call_count=call_count,
        ) from last_error


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
