# C:\Users\18449\Desktop\researchguard_workspace\researchguard\agent\policy.py
from __future__ import annotations

from dataclasses import dataclass

from researchguard.agent.state import ResearchAgentState


@dataclass(frozen=True)
class AgentPolicy:
    max_steps: int = 6
    max_tool_calls: int = 10
    max_retry: int = 2
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive.")
        if self.max_retry < 0:
            raise ValueError("max_retry must not be negative.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")

    def validate_plan(self, state: ResearchAgentState) -> str | None:
        if len(state.plan) > self.max_steps:
            return "max_steps_exceeded"
        return None

    def stop_reason(self, state: ResearchAgentState, *, elapsed_seconds: float) -> str | None:
        if elapsed_seconds >= self.timeout:
            return "timeout_exceeded"
        if state.current_step >= self.max_steps and state.current_step < len(state.plan):
            return "max_steps_exceeded"
        if len(state.tool_history) >= self.max_tool_calls:
            return "max_tool_calls_exceeded"
        return None

    def can_retry(self, retry_count: int) -> bool:
        return retry_count < self.max_retry
