# C:\Users\18449\Desktop\researchguard_workspace\tests\agent\test_policy.py
from __future__ import annotations

import unittest

from researchguard.agent.policy import AgentPolicy
from researchguard.agent.state import ResearchAgentState


class AgentPolicyTests(unittest.TestCase):
    def test_default_limits_are_bounded(self) -> None:
        policy = AgentPolicy()

        self.assertEqual(policy.max_steps, 6)
        self.assertEqual(policy.max_tool_calls, 10)
        self.assertEqual(policy.max_retry, 2)
        self.assertEqual(policy.max_plan_revisions, 2)
        self.assertGreater(policy.timeout, 0)

    def test_timeout_and_call_limits_return_stop_reasons(self) -> None:
        policy = AgentPolicy(max_steps=6, max_tool_calls=2, max_retry=1, timeout=5)
        state = ResearchAgentState(
            query="Question",
            plan=[{"tool": "retrieve_evidence"}],
            tool_history=[{}, {}],
            status="running",
        )

        self.assertEqual(
            policy.stop_reason(state, elapsed_seconds=0.1),
            "max_tool_calls_exceeded",
        )
        state.tool_history.clear()
        self.assertEqual(
            policy.stop_reason(state, elapsed_seconds=5),
            "timeout_exceeded",
        )

    def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AgentPolicy(max_steps=0)
        with self.assertRaises(ValueError):
            AgentPolicy(max_tool_calls=0)
        with self.assertRaises(ValueError):
            AgentPolicy(max_retry=-1)
        with self.assertRaises(ValueError):
            AgentPolicy(max_plan_revisions=-1)
        with self.assertRaises(ValueError):
            AgentPolicy(timeout=0)
        with self.assertRaises(ValueError):
            AgentPolicy(max_plan_revisions=3)


if __name__ == "__main__":
    unittest.main()
