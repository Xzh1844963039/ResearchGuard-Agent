# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\answer_tool.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from researchguard.pipeline import DEFAULT_CONFIG_PATH, ResearchGuardPipeline
from researchguard.tools.contracts import ToolError, ToolResult, ToolSpec


class GuardedAnswerTool:
    name = "generate_grounded_answer"
    version = "1.0.0"
    description = "Run the frozen ResearchGuard evidence gate, answer generation, and citation audit flow."

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self._pipeline = pipeline
        self._config_path = config_path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            version=self.version,
            description=self.description,
            input_schema={
                "query": "non-empty string",
            },
        )

    def _guarded_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = ResearchGuardPipeline.from_config(self._config_path)
        return self._pipeline

    def invoke(self, **kwargs: Any) -> ToolResult:
        return self.generate_grounded_answer(**kwargs)

    def generate_grounded_answer(self, query: str) -> ToolResult:
        started = time.perf_counter()
        try:
            normalized_query = str(query).strip()
            if not normalized_query:
                raise ValueError("Query must not be empty.")
            result = self._guarded_pipeline().run(normalized_query)
            latency_ms = (time.perf_counter() - started) * 1000.0
            final_status = str(result.get("final_status", "failed"))
            if final_status == "grounded":
                return ToolResult.create(
                    status="success",
                    message="Grounded answer generated and citation-audited.",
                    tool_name=self.name,
                    tool_version=self.version,
                    latency_ms=latency_ms,
                    data={"pipeline_result": result},
                )
            if final_status in {"rejected", "needs_review", "disabled"}:
                return ToolResult.create(
                    status="rejected",
                    message="Grounded answer was not released.",
                    reason=final_status,
                    tool_name=self.name,
                    tool_version=self.version,
                    latency_ms=latency_ms,
                    data={"pipeline_result": result},
                )
            error = ToolError(
                code="guarded_pipeline_failed",
                category="execution_failure",
                message=f"Pipeline finished with status: {final_status}",
                retryable=False,
            )
            return ToolResult.create(
                status="failed",
                message="Grounded answer pipeline failed.",
                reason=final_status,
                tool_name=self.name,
                tool_version=self.version,
                latency_ms=latency_ms,
                data={"pipeline_result": result},
                error=error,
            )
        except ValueError as exc:
            return self._failure(started, exc, "invalid_input", "invalid_answer_input", False)
        except TimeoutError as exc:
            return self._failure(started, exc, "timeout", "answer_pipeline_timeout", True)
        except Exception as exc:
            return self._failure(started, exc, "execution_failure", "answer_pipeline_failed", True)

    def _failure(
        self,
        started: float,
        exc: Exception,
        category: str,
        code: str,
        retryable: bool,
    ) -> ToolResult:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ToolResult.create(
            status="failed",
            message="Grounded answer pipeline failed.",
            reason=code,
            tool_name=self.name,
            tool_version=self.version,
            latency_ms=latency_ms,
            error=ToolError(
                code=code,
                category=category,
                message=str(exc),
                retryable=retryable,
                details={"exception_type": type(exc).__name__},
            ),
        )
