# C:\Users\18449\Desktop\researchguard_workspace\researchguard\tools\registry.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from researchguard.pipeline import DEFAULT_CONFIG_PATH
from researchguard.tools.answer_tool import GuardedAnswerTool
from researchguard.tools.audit_tool import CitationAuditTool
from researchguard.tools.contracts import ToolError, ToolResult, ToolSpec
from researchguard.tools.evidence_tool import EvidenceTool
from researchguard.tools.retrieval_tool import RetrievalTool
from researchguard.tools.scholarly_search_tool import ScholarlySearchTool


class ToolRegistry:
    version = "1.0.0"

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        name = str(getattr(tool, "name", "")).strip()
        if not name:
            raise ValueError("Registered tools must define a non-empty name.")
        if not callable(getattr(tool, "invoke", None)):
            raise TypeError(f"Tool {name!r} must define an invoke method.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolResult.create(
                status="failed",
                message="Tool invocation failed.",
                reason="unknown_tool",
                tool_name=name,
                tool_version=self.version,
                latency_ms=0.0,
                error=ToolError(
                    code="unknown_tool",
                    category="invalid_input",
                    message=str(exc),
                    retryable=False,
                ),
            )
        result = tool.invoke(**kwargs)
        if not isinstance(result, ToolResult):
            raise TypeError(f"Tool {name!r} returned {type(result).__name__}, expected ToolResult.")
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools[name].spec for name in self.names)


def build_default_registry(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(RetrievalTool(config_path=config_path))
    registry.register(EvidenceTool(config_path=config_path))
    registry.register(GuardedAnswerTool(config_path=config_path))
    registry.register(CitationAuditTool(config_path=config_path))
    registry.register(ScholarlySearchTool())
    return registry
