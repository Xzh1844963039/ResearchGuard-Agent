# C:\Users\18449\Desktop\researchguard_workspace\researchguard\evaluation\evaluator.py
from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Mapping

from researchguard.evaluation.agent_metrics import (
    efficiency_metrics,
    evidence_metrics,
    intelligence_metrics,
    memory_metrics,
    planning_metrics,
    tool_metrics,
)
from researchguard.evaluation.schemas import (
    AgentEvaluationCase,
    AgentEvaluationReport,
    AgentEvaluationResult,
    MetricValue,
)
from researchguard.tracing import TraceCollector


class AgentEvaluator:
    def __init__(self, registered_tools: Iterable[str]):
        self.registered_tools = tuple(sorted(set(registered_tools)))
        self.trace_collector = TraceCollector()

    def evaluate(
        self,
        state: Any,
        case: AgentEvaluationCase,
        *,
        memory_snapshot: Mapping[str, Any] | None = None,
    ) -> AgentEvaluationResult:
        return self._evaluate(
            state,
            case=case,
            memory_snapshot=memory_snapshot,
        )

    def evaluate_runtime(
        self,
        state: Any,
        *,
        memory_snapshot: Mapping[str, Any] | None = None,
    ) -> AgentEvaluationResult:
        return self._evaluate(state, case=None, memory_snapshot=memory_snapshot)

    def evaluate_many(
        self,
        runs: Iterable[tuple[Any, AgentEvaluationCase, Mapping[str, Any] | None]],
    ) -> AgentEvaluationReport:
        results = tuple(
            self.evaluate(state, case, memory_snapshot=memory_snapshot)
            for state, case, memory_snapshot in runs
        )
        aggregates = self._aggregate(results)
        return AgentEvaluationReport(
            results=results,
            aggregate_metrics=aggregates,
            passed=bool(results) and all(result.passed for result in results),
        )

    def _evaluate(
        self,
        state: Any,
        *,
        case: AgentEvaluationCase | None,
        memory_snapshot: Mapping[str, Any] | None,
    ) -> AgentEvaluationResult:
        tool_calls = list(getattr(state, "tool_history", ()))
        tool_names = [
            str(call.get("tool_name", call.get("tool", ""))) for call in tool_calls
        ]
        metrics: dict[str, MetricValue] = {}
        metrics.update(
            planning_metrics(
                observed_task_type=str(getattr(state, "task_type", "unknown")),
                observed_workflow=getattr(state, "workflow_name", None),
                expected_task_type=case.expected_task_type if case else None,
                expected_workflow=case.expected_workflow if case else None,
                tool_names=tool_names,
                registered_tools=self.registered_tools,
            )
        )
        metrics.update(
            tool_metrics(
                tool_calls,
                expected_tools=case.expected_tools if case else (),
                forbidden_tools=case.forbidden_tools if case else (),
                allow_failed_calls=bool(
                    case
                    and (
                        case.expected_status == "failed"
                        or (
                            case.expected_plan_revisions is not None
                            and case.expected_plan_revisions > 0
                        )
                    )
                ),
            )
        )
        metrics.update(
            evidence_metrics(
                getattr(state, "evidence", ()),
                relevant_evidence_ids=case.relevant_evidence_ids if case else (),
                audit=getattr(state, "audit_result", None),
            )
        )
        metrics.update(efficiency_metrics(state))
        metrics.update(
            intelligence_metrics(
                state,
                expected_plan_revisions=(
                    case.expected_plan_revisions if case else None
                ),
            )
        )
        metrics.update(
            memory_metrics(
                getattr(state, "memory_status", {}),
                memory_snapshot,
                final_status=str(getattr(state, "status", "")),
            )
        )

        expected_status = case.expected_status if case else None
        status_matches = expected_status is None or getattr(state, "status", "") == expected_status
        failed_metrics = sorted(
            name for name, metric in metrics.items() if metric.passed is False
        )
        issues = list(failed_metrics)
        if not status_matches:
            issues.append("final_status")
        passed = status_matches and not failed_metrics
        trace = self.trace_collector.collect(
            state,
            memory_snapshot=memory_snapshot,
        ).to_dict()
        return AgentEvaluationResult(
            case_id=case.case_id if case else str(getattr(state, "run_id", "runtime")),
            passed=passed,
            metrics=metrics,
            issues=tuple(issues),
            expected=case.to_dict() if case else {},
            observed={
                "task_type": getattr(state, "task_type", "unknown"),
                "workflow": getattr(state, "workflow_name", None),
                "tools": tool_names,
                "status": getattr(state, "status", "unknown"),
                "plan_revision_count": len(
                    getattr(state, "plan_revisions", ())
                ),
                "evidence_ids": [
                    item.get("chunk_id")
                    for item in getattr(state, "evidence", ())
                    if isinstance(item, Mapping)
                ],
            },
            trace=trace,
        )

    @staticmethod
    def _aggregate(
        results: Iterable[AgentEvaluationResult],
    ) -> dict[str, float | int | None]:
        result_list = list(results)
        aggregates: dict[str, float | int | None] = {
            "case_count": len(result_list),
            "pass_rate": (
                sum(result.passed for result in result_list) / len(result_list)
                if result_list
                else 0.0
            ),
        }
        names = sorted(
            {name for result in result_list for name in result.metrics}
        )
        for name in names:
            values = [
                (
                    int(result.metrics[name].value)
                    if isinstance(result.metrics[name].value, bool)
                    else result.metrics[name].value
                )
                for result in result_list
                if name in result.metrics
                and isinstance(result.metrics[name].value, (int, float))
            ]
            aggregates[name] = mean(values) if values else None
        return aggregates
