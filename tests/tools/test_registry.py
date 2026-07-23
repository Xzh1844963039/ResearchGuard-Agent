# C:\Users\18449\Desktop\researchguard_workspace\tests\tools\test_registry.py
from __future__ import annotations

import unittest

from researchguard.tools.audit_tool import CitationAuditTool
from researchguard.tools.registry import ToolRegistry, build_default_registry


class RegistryTests(unittest.TestCase):
    def test_default_registry_exposes_only_guarded_interfaces(self) -> None:
        registry = build_default_registry()

        self.assertEqual(
            registry.names,
            (
                "retrieve_evidence",
                "assess_evidence",
                "generate_grounded_answer",
                "audit_answer",
            ),
        )
        self.assertNotIn("generate_answer", registry.names)
        self.assertEqual([spec.name for spec in registry.specs()], list(registry.names))

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ToolRegistry()
        tool = CitationAuditTool()
        registry.register(tool)

        with self.assertRaises(ValueError):
            registry.register(tool)

    def test_unknown_tool_returns_structured_error(self) -> None:
        result = ToolRegistry().invoke("missing_tool")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.category, "invalid_input")
        self.assertEqual(result.reason, "unknown_tool")

    def test_audit_rejects_raw_answer_string_before_backend_call(self) -> None:
        result = CitationAuditTool().audit_answer(
            "A raw answer must not be audited without generation provenance.",
            [
                {
                    "chunk_id": "doc::chunk-1",
                    "doc_id": "doc",
                    "section": "method",
                    "page": 2,
                    "content": "Evidence.",
                    "source": "paper.pdf",
                }
            ],
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.category, "invalid_input")


if __name__ == "__main__":
    unittest.main()
