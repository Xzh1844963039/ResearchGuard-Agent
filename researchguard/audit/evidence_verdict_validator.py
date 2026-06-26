# C:\Users\18449\Desktop\researchguard_workspace\researchguard\audit\evidence_verdict_validator.py
"""
EvidenceClaw Evidence Verdict Validator
========================================

LLM-as-Proposer, Python-as-Verifier 架构的核心硬验证层。

设计原则:
- LLM 可以提出候选 (claim 拆分、query 生成、evidence 匹配、推理桥)
- LLM 的 suggested_label 仅作为建议，不能直接成为最终颜色
- 所有最终 🟢🟡🔴 标签必须通过本模块的确定性 Python 验证

硬规则:
1. 每个 claim 必须能映射回原文 source_span
2. 每个 query 必须保留必需的领域锚点
3. 每个引用的 E-id / R-id 必须真实存在于 evidence_pool
4. green 必须有直接证据支持
5. yellow 必须有部分证据 + 明确的 evidence-linked 推理桥
6. red 分配于: 证据缺失、矛盾、领域不匹配、或 claim 夸大超出证据

这个设计通过让 LLM 生成候选、Python 强制执行证据可用性、文本锚定和标签约束来减少幻觉。
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

STRICT_YELLOW_BRIDGE_REQUIRED = True
"""黄色必须有至少一个合法的推理桥步骤；无桥则降级为 red。"""

ANCHOR_OVERLAP_GREEN_THRESHOLD = 0.15
"""green 的 anchor_overlap coverage 最低阈值。"""

ANCHOR_OVERLAP_TARGET_TEXT_THRESHOLD = 0.05
"""target_text 模式下 green 的 anchor_overlap coverage 最低阈值。
target_text 模式没有 PDF E-ids，仅靠文献 R-ids 的 title/abstract 做交叉验证，
自然语言覆盖天然低于同源 PDF 证据，因此使用更宽松的阈值。"""

# ---------------------------------------------------------------------------
# 领域锚点词表
# ---------------------------------------------------------------------------

GENERAL_AI_ANCHORS: set[str] = {
    "transformer",
    "self-attention",
    "selfattention",
    "attention",
    "multi-head attention",
    "multihead attention",
    "decoder",
    "encoder",
    "sequence modeling",
    "neural network",
    "deep learning",
    "large language model",
    "llm",
    "generative",
    "pretrain",
    "pre-train",
    "fine-tune",
    "finetune",
}

QUANTUM_ANCHORS: set[str] = {
    "quantum",
    "quantum many-body",
    "quantum many body",
    "many-body",
    "many body",
    "wave function",
    "wavefunction",
    "neural quantum state",
    "neural quantum states",
    "nqs",
    "quantum error correction",
    "qec",
    "syndrome",
    "tensor network",
    "tensor networks",
    "matrix product state",
    "matrix product states",
    "mps",
    "variational monte carlo",
    "vmc",
    "hamiltonian",
    "ground state",
    "schrödinger",
    "schrodinger",
    "entanglement",
    "qubit",
    "qubits",
    "superposition",
}

CRISPR_ANCHORS: set[str] = {
    "crispr",
    "cas9",
    "cas12",
    "cas13",
    "genome editing",
    "gene editing",
    "guide rna",
    "tracrrna",
    "crrna",
    "endonuclease",
    "off-target",
    "base editing",
    "prime editing",
}

COT_EVO_ANCHORS: set[str] = {
    "chain-of-thought",
    "chain of thought",
    "cot",
    "evolutionary algorithm",
    "evolutionary optimization",
    "reasoning trace",
    "distillation",
    "fitness",
}

# 归一化映射: 缩写/变体 → 规范形式
ANCHOR_NORMALIZATION: dict[str, str] = {
    "qec": "quantum error correction",
    "nqs": "neural quantum state",
    "mps": "matrix product state",
    "vmc": "variational monte carlo",
    "cot": "chain-of-thought",
    "selfattention": "self-attention",
    "multihead attention": "multi-head attention",
    "wavefunction": "wave function",
    "finetune": "fine-tune",
    "pre-train": "pretrain",
}

ALL_DOMAIN_ANCHORS: set[str] = (
    GENERAL_AI_ANCHORS | QUANTUM_ANCHORS | CRISPR_ANCHORS | COT_EVO_ANCHORS
)


# ---------------------------------------------------------------------------
# 2.1 normalize_id_list
# ---------------------------------------------------------------------------

def normalize_id_list(ids: Any) -> list[str]:
    """统一处理 evidence ids，避免 None、字符串、重复 id 导致后续误判。

    - 输入 None 返回 []
    - 输入字符串返回 [字符串]
    - 输入 list 时去掉 None 和空字符串
    - 保持顺序去重
    """
    if ids is None:
        return []
    if isinstance(ids, str):
        return [ids.strip()] if ids.strip() else []
    if isinstance(ids, (list, tuple)):
        seen: set[str] = set()
        result: list[str] = []
        for item in ids:
            if item is None:
                continue
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result
    return []


# ---------------------------------------------------------------------------
# 2.2 evidence_id_exists
# ---------------------------------------------------------------------------

def _extract_id_fields(item: dict) -> set[str]:
    """从 evidence item 中提取所有可能的 id 字段值。"""
    ids: set[str] = set()
    for field in ("id", "evidence_id", "source_id", "rid", "eid", "ref_id", "query_id"):
        val = item.get(field)
        if val and isinstance(val, str):
            ids.add(val.strip())
    return ids


def evidence_id_exists(evidence_id: str, evidence_pool: dict | list | None) -> bool:
    """检查 LLM 或 audit 结果中引用的 E-id / R-id 是否真实存在于 evidence_pool。

    支持 evidence_pool 是 dict 或 list 两种情况。
    如果 evidence_pool 是 dict，检查 key。
    如果 evidence_pool 是 list，检查每个 item 里的 id / evidence_id / source_id / rid / eid 字段。

    硬规则: LLM 不能凭空生成 E-id / R-id。
    """
    if not evidence_id or not evidence_pool:
        return False

    eid = evidence_id.strip()

    if isinstance(evidence_pool, dict):
        # 先检查 key
        if eid in evidence_pool:
            return True
        # 再检查每个 value 的 id 字段
        for value in evidence_pool.values():
            if isinstance(value, dict):
                if eid in _extract_id_fields(value):
                    return True
        return False

    if isinstance(evidence_pool, list):
        for item in evidence_pool:
            if isinstance(item, dict):
                if eid in _extract_id_fields(item):
                    return True
        return False

    return False


# ---------------------------------------------------------------------------
# 2.3 filter_existing_evidence_ids
# ---------------------------------------------------------------------------

def filter_existing_evidence_ids(
    used_evidence_ids: Any, evidence_pool: dict | list | None
) -> list[str]:
    """给定 used_evidence_ids 和 evidence_pool，只保留真实存在的 evidence ids。"""
    ids = normalize_id_list(used_evidence_ids)
    return [eid for eid in ids if evidence_id_exists(eid, evidence_pool)]


# ---------------------------------------------------------------------------
# 2.4 extract_domain_anchors
# ---------------------------------------------------------------------------

def extract_domain_anchors(text: str) -> list[str]:
    """从 claim 或 target_text 中抽取必须保留的领域锚点。

    使用规则词表匹配，大小写不敏感。
    返回规范化的锚点列表。
    """
    if not text:
        return []

    lowered = text.lower()
    found: list[str] = []

    # 按长度降序排列，优先匹配长词（如 "quantum error correction" 优于 "quantum"）
    sorted_anchors = sorted(ALL_DOMAIN_ANCHORS, key=len, reverse=True)
    matched_positions: set[int] = set()

    for anchor in sorted_anchors:
        idx = lowered.find(anchor)
        while idx >= 0:
            # 检查这个位置是否已被更长的锚点覆盖
            if not any(pos <= idx < pos + len(anchor) for pos in matched_positions):
                # 归一化
                normalized = ANCHOR_NORMALIZATION.get(anchor, anchor)
                if normalized not in found:
                    found.append(normalized)
                matched_positions.add(idx)
            idx = lowered.find(anchor, idx + 1)

    return found


# ---------------------------------------------------------------------------
# 2.5 query_covers_domain_anchors
# ---------------------------------------------------------------------------

def query_covers_domain_anchors(text: str, queries: list[str]) -> tuple[bool, list[str]]:
    """检查生成的 query 是否覆盖原文重要锚点。

    Returns:
        ok: bool — 所有锚点都被至少一条 query 覆盖
        missing_anchors: list[str] — 未被覆盖的锚点
    """
    anchors = extract_domain_anchors(text)
    if not anchors:
        return True, []

    # 将所有 query 合并为一个文本
    queries_text = " ".join(queries).lower() if queries else ""

    missing: list[str] = []
    for anchor in anchors:
        anchor_lower = anchor.lower()
        if anchor_lower not in queries_text:
            # 也检查近义词/缩写形式
            found_variant = False
            variants = [anchor_lower]
            for abbrev, full in ANCHOR_NORMALIZATION.items():
                if full == anchor:
                    variants.append(abbrev)
                if abbrev == anchor:
                    variants.append(full)
            for variant in variants:
                if variant in queries_text:
                    found_variant = True
                    break
            if not found_variant:
                missing.append(anchor)

    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# 2.6 claim_maps_to_source_span
# ---------------------------------------------------------------------------

# claim 中的强词模式
STRONG_CLAIM_PATTERNS: list[str] = [
    "prove",
    "proves",
    "proved",
    "guarantee",
    "guarantees",
    "guaranteed",
    "state-of-the-art",
    "completely",
    "entirely",
    "always",
    "never",
    "solves all",
    "without any limitation",
    "undeniably",
    "unequivocally",
]


def claim_maps_to_source_span(claim: str, source_span: str | None) -> tuple[bool, str]:
    """检查 LLM 拆出来的 claim 是否能回溯到原文 source_span。

    避免 LLM 自己编新 claim。

    Returns:
        ok: bool
        reason: str
    """
    if not source_span:
        return False, "No source_span provided; claim cannot be verified against original text."

    claim_stripped = (claim or "").strip()
    source_stripped = source_span.strip()

    if not claim_stripped:
        return False, "Claim is empty."

    # 完全相同
    if claim_stripped == source_stripped:
        return True, "Exact match with source_span."

    # 计算 token overlap
    claim_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+|\d+", claim_stripped.lower()))
    source_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+|\d+", source_stripped.lower()))

    if not claim_tokens or not source_tokens:
        return False, "Cannot compute token overlap."

    overlap = claim_tokens & source_tokens
    overlap_ratio = len(overlap) / max(1, len(claim_tokens))

    if overlap_ratio < 0.3:
        return False, f"Token overlap too low ({overlap_ratio:.2f}); claim may be fabricated."

    # 检查 claim 是否比 source_span 多出强词
    claim_lower = claim_stripped.lower()
    source_lower = source_stripped.lower()
    extra_strong_terms: list[str] = []
    for term in STRONG_CLAIM_PATTERNS:
        if term in claim_lower and term not in source_lower:
            extra_strong_terms.append(term)

    if extra_strong_terms:
        return (
            False,
            f"Claim introduces strong terms not in source_span: {', '.join(extra_strong_terms)}",
        )

    return True, f"Claim maps to source_span with token overlap {overlap_ratio:.2f}."


# ---------------------------------------------------------------------------
# 2.7 detect_overclaim
# ---------------------------------------------------------------------------

OVERCLAIM_PATTERNS: dict[str, str] = {
    # severe patterns — 极强的绝对化表述
    "completely replace": "severe",
    "completely replaces": "severe",
    "completely replaced": "severe",
    "completely solves": "severe",
    "completely solved": "severe",
    "entirely eliminate": "severe",
    "entirely eliminates": "severe",
    "entirely replaced": "severe",
    "fully resolved": "severe",
    "fully solves": "severe",
    "fully proves": "severe",
    "eliminate all": "severe",
    "solves all": "severe",
    "without any limitation": "severe",
    "no longer needs": "severe",
    "never fails": "severe",
    "never": "severe",
    "always works": "severe",
    "always correct": "severe",
    "guarantees": "severe",
    "guarantee": "severe",
    "guaranteed": "severe",
    "prove that": "severe",
    "proves that": "severe",
    "undeniably": "severe",
    "unequivocally": "severe",
    # mild patterns — 较强但不极端的表述
    "completely": "mild",
    "entirely": "mild",
    "fully": "mild",
    "perfect": "mild",
    "perfectly": "mild",
    "always": "mild",
    "state-of-the-art": "mild",
    "best": "mild",
    "all problems": "mild",
}


def detect_overclaim(text: str) -> dict:
    """检测 claim 是否有明显过强表述。

    Returns:
        {
            "has_overclaim": bool,
            "severity": "none" | "mild" | "severe",
            "matched_terms": list[str]
        }

    注意: 不要一看到 overclaim 就直接红。最终颜色还要看证据强度。
    但强表述没有强证据时，绝对不能绿。
    """
    if not text:
        return {"has_overclaim": False, "severity": "none", "matched_terms": []}

    lowered = text.lower()
    matched: list[tuple[str, str]] = []

    # 按长度降序遍历，优先匹配长词
    for pattern, severity in sorted(OVERCLAIM_PATTERNS.items(), key=lambda x: -len(x[0])):
        if pattern in lowered:
            matched.append((pattern, severity))

    if not matched:
        return {"has_overclaim": False, "severity": "none", "matched_terms": []}

    severities = [s for _, s in matched]
    overall = "severe" if "severe" in severities else "mild"

    return {
        "has_overclaim": True,
        "severity": overall,
        "matched_terms": [p for p, _ in matched],
    }


# ---------------------------------------------------------------------------
# 2.8 evidence_text_for_id
# ---------------------------------------------------------------------------

EVIDENCE_TEXT_FIELDS = ("text", "snippet", "abstract", "content", "title", "summary",
                         "content_summary", "clean_content", "hypothesis_text",
                         "exact_text", "explanation")


def evidence_text_for_id(evidence_id: str, evidence_pool: dict | list | None) -> str:
    """根据 evidence_id 从 evidence_pool 里取出 evidence 文本。

    支持字段: text, snippet, abstract, content, title, summary, content_summary,
              clean_content, hypothesis_text, exact_text, explanation
    """
    if not evidence_id or not evidence_pool:
        return ""

    eid = evidence_id.strip()

    def _extract_text(item: dict) -> str:
        parts: list[str] = []
        for field in EVIDENCE_TEXT_FIELDS:
            val = item.get(field)
            if val and isinstance(val, str):
                parts.append(val)
        return " ".join(parts)

    if isinstance(evidence_pool, dict):
        if eid in evidence_pool and isinstance(evidence_pool[eid], dict):
            return _extract_text(evidence_pool[eid])
        for value in evidence_pool.values():
            if isinstance(value, dict) and eid in _extract_id_fields(value):
                return _extract_text(value)

    if isinstance(evidence_pool, list):
        for item in evidence_pool:
            if isinstance(item, dict) and eid in _extract_id_fields(item):
                return _extract_text(item)

    return ""


# ---------------------------------------------------------------------------
# 2.9 anchor_overlap_between_claim_and_evidence
# ---------------------------------------------------------------------------

def anchor_overlap_between_claim_and_evidence(
    claim: str, evidence_text: str
) -> dict:
    """计算 claim 的关键锚点有多少出现在 evidence 文本中。

    Returns:
        {
            "claim_anchors": list[str],
            "matched_anchors": list[str],
            "missing_anchors": list[str],
            "coverage": float  # 0.0-1.0
        }
    """
    claim_anchors = extract_domain_anchors(claim)
    if not claim_anchors:
        return {
            "claim_anchors": [],
            "matched_anchors": [],
            "missing_anchors": [],
            "coverage": 1.0,  # 没有领域锚点时不算问题
        }

    evidence_lower = (evidence_text or "").lower()
    matched: list[str] = []
    missing: list[str] = []

    for anchor in claim_anchors:
        anchor_lower = anchor.lower()
        if anchor_lower in evidence_lower:
            matched.append(anchor)
        else:
            # 检查变体
            variants = [anchor_lower]
            for abbrev, full in ANCHOR_NORMALIZATION.items():
                if full == anchor:
                    variants.append(abbrev)
                if abbrev == anchor:
                    variants.append(full)
            if any(v in evidence_lower for v in variants):
                matched.append(anchor)
            else:
                missing.append(anchor)

    coverage = len(matched) / len(claim_anchors) if claim_anchors else 1.0

    return {
        "claim_anchors": claim_anchors,
        "matched_anchors": matched,
        "missing_anchors": missing,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# 2.10 validate_final_verdict — 核心函数
# ---------------------------------------------------------------------------

# 物理/量子相关锚点（用于 domain mismatch 检查）
PHYSICS_QUANTUM_ANCHORS: set[str] = {
    "quantum", "quantum many-body", "many-body", "wave function",
    "quantum error correction", "qec", "tensor network", "tensor networks",
    "matrix product state", "matrix product states", "mps",
    "neural quantum state", "neural quantum states", "nqs",
    "variational monte carlo", "vmc", "hamiltonian", "ground state",
    "schrödinger", "schrodinger", "entanglement", "qubit", "qubits",
    "superposition", "quantum mechanics", "quantum physics",
}


def validate_final_verdict(
    claim: str,
    source_span: str | None = None,
    suggested_label: str | None = None,
    used_evidence_ids: Any = None,
    evidence_pool: dict | list | None = None,
    direct_support: bool = False,
    partial_support: bool = False,
    inference_bridge: list | None = None,
    contradiction: bool = False,
    debug: dict | None = None,
    is_target_text_mode: bool = False,
) -> dict:
    """核心硬验证函数。

    LLM suggested_label 仅作为参考。最终 final_label 由此函数决定。

    硬规则:
    规则 1: 如果没有 valid_evidence_ids → red
    规则 2: 如果 contradiction=True → red
    规则 3: severe overclaim + 没有 direct_support → 不能 green
    规则 4: green 必须: valid evidence + direct_support + 无 severe overclaim + anchor_overlap
    规则 5: yellow 必须: valid evidence + partial_support + inference_bridge (if strict)
    规则 6: 其他情况 → red

    Returns:
        {
            "final_label": "green" | "yellow" | "red",
            "reason": str,
            "valid_evidence_ids": list[str],
            "overclaim": dict,
            "anchor_overlap": dict,
            "debug": dict,
        }
    """
    # --- 预处理 ---
    evidence_ids_raw = normalize_id_list(used_evidence_ids)
    valid_evidence_ids = filter_existing_evidence_ids(evidence_ids_raw, evidence_pool)

    # 合并所有 evidence 文本
    all_evidence_text = " ".join(
        evidence_text_for_id(eid, evidence_pool) for eid in valid_evidence_ids
    )

    # overclaim 检测
    overclaim = detect_overclaim(claim)

    # anchor overlap
    anchor_overlap = anchor_overlap_between_claim_and_evidence(claim, all_evidence_text)

    # target_text 模式: 确定有效的 anchor_overlap 阈值
    # - 无 PDF E-ids 时使用更宽松的阈值（仅靠文献 title/abstract 交叉验证）
    # - 有 PDF E-ids 时使用标准阈值
    _has_eids = any(eid.startswith("E") for eid in valid_evidence_ids)
    _effective_anchor_threshold = (
        ANCHOR_OVERLAP_TARGET_TEXT_THRESHOLD
        if is_target_text_mode and not _has_eids
        else ANCHOR_OVERLAP_GREEN_THRESHOLD
    )

    # bridge 规范化
    bridge = inference_bridge or []
    if isinstance(bridge, dict):
        bridge = [bridge]
    bridge = [b for b in bridge if isinstance(b, dict)]

    # 验证 bridge 中的 evidence_id
    valid_bridge_steps: list[dict] = []
    for step in bridge:
        step_eid = step.get("evidence_id", "")
        if step_eid and evidence_id_exists(step_eid, evidence_pool):
            valid_bridge_steps.append(step)

    # claim→source_span 验证
    span_ok, span_reason = claim_maps_to_source_span(claim, source_span)

    # --- 规则 1: 没有真实 evidence → red ---
    if not valid_evidence_ids:
        reason_parts = ["No valid evidence IDs found in evidence_pool."]
        if evidence_ids_raw:
            reason_parts.append(
                f"LLM cited {evidence_ids_raw}, but none exist in the evidence pool."
            )
        if suggested_label and suggested_label != "red":
            reason_parts.append(
                f"LLM suggested '{suggested_label}', but no real evidence exists. Downgraded to red."
            )
        return {
            "final_label": "red",
            "reason": " ".join(reason_parts),
            "valid_evidence_ids": [],
            "overclaim": overclaim,
            "anchor_overlap": anchor_overlap,
            "support_subtype": "red_no_evidence",
            "required_anchor_eval": {},
            "debug": debug or {},
        }

    # --- 规则 2: contradiction → red ---
    if contradiction:
        reason_parts = ["Evidence contradicts the claim."]
        if suggested_label and suggested_label != "red":
            reason_parts.append(
                f"LLM suggested '{suggested_label}', but contradiction detected. Downgraded to red."
            )
        return {
            "final_label": "red",
            "reason": " ".join(reason_parts),
            "valid_evidence_ids": valid_evidence_ids,
            "overclaim": overclaim,
            "anchor_overlap": anchor_overlap,
            "support_subtype": "red_invalid_reference",
            "required_anchor_eval": {},
            "debug": debug or {},
        }

    # --- 辅助: 开放域 required-anchor coverage (Phase 11) ---
    required_anchor_eval = evaluate_required_anchor_coverage(
        claim, all_evidence_text, valid_evidence_ids, evidence_pool
    )
    coverage_status = required_anchor_eval["coverage_status"]
    method_only_match = required_anchor_eval["method_only_match"]
    target_anchors_req = required_anchor_eval["target_anchors"]
    matched_target_req = required_anchor_eval["matched_target_anchors"]
    missing_target_req = required_anchor_eval["missing_target_anchors"]
    spec_flags = required_anchor_eval.get("specificity_flags", [])

    # --- 辅助: domain mismatch (保留 PHYSICS_QUANTUM_ANCHORS 作为 debug) ---
    claim_anchors = extract_domain_anchors(claim)
    has_physics_claim = any(
        a.lower() in PHYSICS_QUANTUM_ANCHORS or
        any(v in a.lower() for v in PHYSICS_QUANTUM_ANCHORS)
        for a in claim_anchors
    )
    evidence_anchors = extract_domain_anchors(all_evidence_text)
    has_physics_evidence = any(
        a.lower() in PHYSICS_QUANTUM_ANCHORS or
        any(v in a.lower() for v in PHYSICS_QUANTUM_ANCHORS)
        for a in evidence_anchors
    )
    domain_mismatch = has_physics_claim and not has_physics_evidence

    # Phase 11: required-anchor mismatch 是主判断
    _is_high_risk_specific = bool(
        spec_flags or
        overclaim.get("severity") == "severe"
    )

    # --- 规则 2.5: required-anchor 硬拦截 ---
    # missing_required_target / method_only → 不可 green, 不可 yellow
    if coverage_status in {"missing_required_target", "method_only"} and target_anchors_req:
        subtype = "red_missing_required_anchor" if coverage_status == "missing_required_target" else "red_method_only_mismatch"
        reason_parts = []
        if coverage_status == "method_only":
            reason_parts.append(
                f"Method-only evidence: evidence covers only generic method anchors "
                f"({', '.join(required_anchor_eval['matched_method_anchors'][:5])}) "
                f"but misses required target anchors "
                f"({', '.join(missing_target_req[:5])})."
            )
        else:
            reason_parts.append(
                f"Missing required target anchors: "
                f"({', '.join(missing_target_req[:5])}) not found in evidence."
            )
        if suggested_label and suggested_label != "red":
            reason_parts.append(
                f"LLM suggested '{suggested_label}', but evidence does not cover "
                f"the claim's required target anchors. Downgraded to red."
            )
        return {
            "final_label": "red",
            "reason": " ".join(reason_parts),
            "valid_evidence_ids": valid_evidence_ids,
            "overclaim": overclaim,
            "anchor_overlap": anchor_overlap,
            "support_subtype": subtype,
            "required_anchor_eval": required_anchor_eval,
            "debug": debug or {},
        }

    # --- 规则 3+4: green 检查 ---
    can_be_green = True
    green_blockers: list[str] = []

    if not direct_support:
        can_be_green = False
        green_blockers.append("direct_support is False")

    if overclaim["severity"] == "severe" and not direct_support:
        can_be_green = False
        green_blockers.append(
            f"Severe overclaim ({', '.join(overclaim['matched_terms'])}) without strong direct evidence"
        )

    if anchor_overlap["coverage"] < _effective_anchor_threshold:
        can_be_green = False
        green_blockers.append(
            f"Anchor overlap coverage {anchor_overlap['coverage']:.2f} below threshold {_effective_anchor_threshold}"
        )

    if domain_mismatch:
        can_be_green = False
        green_blockers.append(
            "Domain mismatch: claim contains physics/quantum anchors but evidence does not"
        )

    if coverage_status == "background_only":
        can_be_green = False
        green_blockers.append(
            "Coverage status is background_only: evidence is topically related "
            "but does not directly prove the claim's relation"
        )

    if overclaim["severity"] == "severe":
        # severe overclaim 需要 evidence 明确强支持
        # target_text 模式使用更宽松的阈值
        _severe_min_coverage = 0.15 if is_target_text_mode else 0.3
        if not direct_support or anchor_overlap["coverage"] < _severe_min_coverage:
            can_be_green = False
            if "Severe overclaim" not in " ".join(green_blockers):
                green_blockers.append(
                    "Severe overclaim requires strong direct evidence with high anchor overlap"
                )

    # 如果 suggested_label 是 green
    if suggested_label == "green":
        if can_be_green:
            return {
                "final_label": "green",
                "reason": "Green: claim is directly supported by valid evidence with sufficient anchor overlap.",
                "valid_evidence_ids": valid_evidence_ids,
                "overclaim": overclaim,
                "anchor_overlap": anchor_overlap,
                "support_subtype": "green_direct",
                "required_anchor_eval": required_anchor_eval,
                "debug": debug or {},
            }
        else:
            # 降级: 检查是否能 yellow
            can_be_yellow, yellow_blockers = _check_yellow_conditions(
                valid_evidence_ids, partial_support or direct_support,
                valid_bridge_steps, domain_mismatch, required_anchor_eval,
            )
            if can_be_yellow:
                _sub = _compute_support_subtype(
                    "yellow", coverage_status, valid_evidence_ids,
                    overclaim.get("severity", "none"), spec_flags,
                )
                return {
                    "final_label": "yellow",
                    "reason": (
                        f"LLM suggested green, but downgraded to yellow. "
                        f"Green blocked: {'; '.join(green_blockers)}. "
                        f"Yellow allowed with partial evidence and bridge."
                    ),
                    "valid_evidence_ids": valid_evidence_ids,
                    "overclaim": overclaim,
                    "anchor_overlap": anchor_overlap,
                    "support_subtype": _sub,
                    "required_anchor_eval": required_anchor_eval,
                    "debug": debug or {},
                }
            else:
                _sub = _compute_support_subtype(
                    "red", coverage_status, valid_evidence_ids,
                    overclaim.get("severity", "none"), spec_flags,
                )
                return {
                    "final_label": "red",
                    "reason": (
                        f"LLM suggested green, but downgraded to red. "
                        f"Green blocked: {'; '.join(green_blockers)}. "
                        f"Yellow blocked: {'; '.join(yellow_blockers)}."
                    ),
                    "valid_evidence_ids": valid_evidence_ids,
                    "overclaim": overclaim,
                    "anchor_overlap": anchor_overlap,
                    "support_subtype": _sub,
                    "required_anchor_eval": required_anchor_eval,
                    "debug": debug or {},
                }

    # --- 规则 5: yellow 检查 ---
    if suggested_label == "yellow":
        can_be_yellow, yellow_blockers = _check_yellow_conditions(
            valid_evidence_ids, partial_support or direct_support,
            valid_bridge_steps, domain_mismatch, required_anchor_eval,
        )
        if can_be_yellow:
            _sub = _compute_support_subtype(
                "yellow", coverage_status, valid_evidence_ids,
                overclaim.get("severity", "none"), spec_flags,
            )
            return {
                "final_label": "yellow",
                "reason": "Yellow: partial evidence with valid inference bridge.",
                "valid_evidence_ids": valid_evidence_ids,
                "overclaim": overclaim,
                "anchor_overlap": anchor_overlap,
                "support_subtype": _sub,
                "required_anchor_eval": required_anchor_eval,
                "debug": debug or {},
            }
        else:
            _sub = _compute_support_subtype(
                "red", coverage_status, valid_evidence_ids,
                overclaim.get("severity", "none"), spec_flags,
            )
            return {
                "final_label": "red",
                "reason": (
                    f"LLM suggested yellow, but downgraded to red. "
                    f"Yellow blocked: {'; '.join(yellow_blockers)}."
                ),
                "valid_evidence_ids": valid_evidence_ids,
                "overclaim": overclaim,
                "anchor_overlap": anchor_overlap,
                "support_subtype": _sub,
                "required_anchor_eval": required_anchor_eval,
                "debug": debug or {},
            }

    # --- 规则 6: suggested_label == "red" 或未知 label ---
    if suggested_label == "red":
        _sub = _compute_support_subtype(
            "red", coverage_status, valid_evidence_ids,
            overclaim.get("severity", "none"), spec_flags,
        )
        return {
            "final_label": "red",
            "reason": "Red: LLM identified issues with this claim.",
            "valid_evidence_ids": valid_evidence_ids,
            "overclaim": overclaim,
            "anchor_overlap": anchor_overlap,
            "support_subtype": _sub,
            "required_anchor_eval": required_anchor_eval,
            "debug": debug or {},
        }

    # 未知 suggested_label — 按最严格原则判红
    # 除非满足 green 条件
    if can_be_green and direct_support:
        return {
            "final_label": "green",
            "reason": "Green: all hard checks passed despite unknown suggested_label.",
            "valid_evidence_ids": valid_evidence_ids,
            "overclaim": overclaim,
            "anchor_overlap": anchor_overlap,
            "support_subtype": "green_direct",
            "required_anchor_eval": required_anchor_eval,
            "debug": debug or {},
        }

    can_be_yellow, yellow_blockers = _check_yellow_conditions(
        valid_evidence_ids, partial_support or direct_support,
        valid_bridge_steps, domain_mismatch, required_anchor_eval,
    )
    if can_be_yellow:
        _sub = _compute_support_subtype(
            "yellow", coverage_status, valid_evidence_ids,
            overclaim.get("severity", "none"), spec_flags,
        )
        return {
            "final_label": "yellow",
            "reason": f"Yellow: partial evidence with valid bridge (unknown suggested_label: {suggested_label}).",
            "valid_evidence_ids": valid_evidence_ids,
            "overclaim": overclaim,
            "anchor_overlap": anchor_overlap,
            "support_subtype": _sub,
            "required_anchor_eval": required_anchor_eval,
            "debug": debug or {},
        }

    _sub = _compute_support_subtype(
        "red", coverage_status, valid_evidence_ids,
        overclaim.get("severity", "none"), spec_flags,
    )
    return {
        "final_label": "red",
        "reason": (
            f"Red: unknown suggested_label '{suggested_label}' and insufficient evidence. "
            f"Green blocked: {'; '.join(green_blockers) if green_blockers else 'N/A'}. "
            f"Yellow blocked: {'; '.join(yellow_blockers) if yellow_blockers else 'N/A'}."
        ),
        "valid_evidence_ids": valid_evidence_ids,
        "overclaim": overclaim,
        "anchor_overlap": anchor_overlap,
        "support_subtype": _sub,
        "required_anchor_eval": required_anchor_eval,
        "debug": debug or {},
    }


def _check_yellow_conditions(
    valid_evidence_ids: list[str],
    has_partial_support: bool,
    valid_bridge_steps: list[dict],
    domain_mismatch: bool,
    required_anchor_eval: dict | None = None,
) -> tuple[bool, list[str]]:
    """检查是否满足 yellow 条件。Phase 11: 加入 required-anchor coverage。

    Returns:
        (can_be_yellow, blockers)
    """
    blockers: list[str] = []

    if not valid_evidence_ids:
        blockers.append("No valid evidence IDs")
        return False, blockers

    if not has_partial_support:
        blockers.append("No partial support")
        return False, blockers

    # Phase 11: required-anchor check
    if required_anchor_eval:
        coverage_status = required_anchor_eval.get("coverage_status", "")
        if coverage_status in {"missing_required_target", "method_only"}:
            blockers.append(
                f"Required anchor coverage status '{coverage_status}': "
                "evidence does not cover claim's required target anchors"
            )
            return False, blockers
        if coverage_status == "background_only":
            target_anchors = required_anchor_eval.get("target_anchors", [])
            matched_target = required_anchor_eval.get("matched_target_anchors", [])
            spec_flags = required_anchor_eval.get("specificity_flags", [])
            if spec_flags or not matched_target:
                blockers.append(
                    "Coverage status is background_only with high-risk specific claim; "
                    "cannot be yellow"
                )
                return False, blockers
        if coverage_status == "partial":
            spec_flags = required_anchor_eval.get("specificity_flags", [])
            target_cov = required_anchor_eval.get("target_coverage", 0)
            if spec_flags:
                blockers.append(
                    "Coverage is partial but claim has high specificity flags "
                    f"({', '.join(spec_flags[:3])}) and target coverage {target_cov:.2f}; "
                    "cannot be yellow"
                )
                return False, blockers

    if STRICT_YELLOW_BRIDGE_REQUIRED and not valid_bridge_steps:
        blockers.append(
            "STRICT_YELLOW_BRIDGE_REQUIRED is True but no valid inference bridge steps "
            "with existing evidence_ids"
        )
        return False, blockers

    if domain_mismatch:
        blockers.append("Domain mismatch: physics/quantum claim without physics/quantum evidence")
        return False, blockers

    return True, []


def _compute_support_subtype(
    final_label: str,
    coverage_status: str,
    valid_evidence_ids: list,
    overclaim_severity: str,
    spec_flags: list,
) -> str:
    """Compute support_subtype for the audit result."""
    if final_label == "green":
        return "green_direct"
    if final_label == "yellow":
        if coverage_status in {"background_only", "method_only"}:
            return "yellow_background_only"
        if coverage_status == "partial":
            return "yellow_partial_direct"
        return "yellow_bridge"
    # red
    if not valid_evidence_ids:
        return "red_no_evidence"
    if coverage_status == "missing_required_target":
        return "red_missing_required_anchor"
    if coverage_status == "method_only":
        return "red_method_only_mismatch"
    if spec_flags:
        return "red_unsupported_specific_claim"
    if overclaim_severity == "severe":
        return "red_overclaim"
    return "red_invalid_reference"


# =============================================================================
# 开放域 required-anchor matching (Phase 11)
# =============================================================================
# 核心原则:
#   不问“claim 属于哪个预定义领域”
#   而问“claim 自己提出了哪些必须被 evidence 覆盖的锚点”
# =============================================================================

# -- 通用 method anchors: 方法/模型/算法类 --
METHOD_TERMS: set[str] = {
    "transformer", "transformers", "self-attention", "selfattention",
    "attention", "multi-head attention", "multihead attention",
    "encoder", "decoder", "encoder-decoder",
    "neural network", "neural networks", "deep learning",
    "large language model", "large language models", "llm", "llms",
    "language model", "language models",
    "cnn", "rnns", "rnn", "convolutional",
    "graph neural network", "graph neural networks", "gnn", "gnns",
    "diffusion model", "diffusion models",
    "reinforcement learning", "rl",
    "generative model", "generative models",
    "variational", "bayesian",
    "machine learning", "ml", "model", "models",
    "algorithm", "algorithms", "method", "approach",
    "architecture", "framework",
    "attention mechanism", "attention mechanisms",
    "sequence model", "sequence modeling",
    "pretrained", "pre-trained", "fine-tuned", "finetuned",
}

# -- 过于泛化的词 (不应作为 target anchor) --
TOO_GENERIC_TARGETS: set[str] = {
    "method", "methods", "approach", "approaches", "model", "models",
    "system", "systems", "result", "results", "paper", "papers",
    "study", "studies", "task", "tasks", "data", "dataset", "datasets",
    "performance", "accuracy", "experiment", "experiments",
    "problem", "problems", "work", "technique", "techniques",
    "application", "applications", "field", "domain",
    "state", "states", "function", "functions",
    "network", "networks", "learning", "training",
    "inference", "optimization", "evaluation",
    "analysis", "design", "framework",
    "baseline", "baselines", "benchmark", "benchmarks",
    "standard", "traditional", "conventional",
    "novel", "new", "improved", "efficient",
    "based", "using", "via", "through",
    "the", "a", "an", "this", "that", "these", "those",
    "can", "may", "might", "could", "would", "should",
    "be", "been", "is", "are", "was", "were",
    "has", "have", "had", "do", "does", "did",
    "not", "no", "nor",
    "and", "or", "but", "for", "with", "from",
    "its", "their", "our", "we", "they",
}

# -- 关系锚点: 描述 method 对 target 做了什么 --
RELATION_TERMS: set[str] = {
    "help", "helps", "improve", "improves", "outperform", "outperforms",
    "decode", "decodes", "decoding",
    "optimize", "optimizes", "optimization", "optimizing",
    "compile", "compiles", "compilation", "compiling",
    "predict", "predicts", "prediction", "predicting",
    "infer", "infers", "inference",
    "reconstruct", "reconstructs", "reconstruction",
    "reduce", "reduces", "reduction", "reducing",
    "accelerate", "accelerates", "acceleration",
    "represent", "represents", "representation",
    "model", "models", "modeling", "modelling",
    "classify", "classifies", "classification",
    "diagnose", "diagnoses", "diagnosis",
    "design", "designs", "designing",
    "generate", "generates", "generation", "generating",
    "detect", "detects", "detection",
    "estimate", "estimates", "estimation",
    "solve", "solves", "solving", "solution",
    "enhance", "enhances", "enhancement",
    "enable", "enables", "allow", "allows",
    "achieve", "achieves",
    "demonstrate", "demonstrates",
    "apply", "applies", "application",
    "can help", "may help", "could help", "might help",
}

# -- specificity 检测模式 --
SPECIFICITY_PATTERNS: dict[str, str] = {
    "nature paper": r"\bnature\b",
    "science paper": r"\bscience\b",
    "arxiv id": r"\barxiv[:\s]*\d{4}\.\d{4,5}",
    "published at": r"\bpublished\s+(?:at|in)\b",
    "state of the art": r"\bstate.of.the.art\b|\bsota\b",
    "already applied": r"\balready\s+(?:applied|deployed|used)\b",
    "causal or strong effect": r"\b(?:proves|prove|proved|guarantees|causes|causal)\b",
    "outperforms all": r"\boutperform(?:s|ed)?\s+all\b",
    "first to": r"\bfirst\s+to\b",
}

GENERIC_TARGET_PHRASES: set[str] = {
    "the most fundamental",
    "most fundamental",
    "what this paper",
    "what this paper can teach",
    "can teach you",
    "you want",
    "dive into",
    "specific direction",
    "specific directions",
    "architectural insights",
    "several key directions",
    "key directions",
    "different energy",
    "this paper",
    "this method",
    "the contribution",
    "main contribution",
    # Chinese discourse / heading phrases
    "几个关键方向",
    "具体建议",
    "短期能做的事情",
    "可能的角度",
    "可能的论文角度",
    "一些具体建议",
    "几个方向",
    "几个方面",
    "主要方向",
    "主要方面",
    "具体方向",
    "短期目标",
    "下一步工作",
    "未来工作",
    "相关工作",
    "主要贡献",
}

TECHNICAL_TARGET_HINTS: set[str] = {
    "quantum", "qubit", "qubits", "syndrome", "hamiltonian", "ground", "state",
    "wave", "function", "many-body", "tensor", "matrix", "product", "surface",
    "code", "decoding", "decoder", "correction", "transduction", "sequence",
    "attention", "transformer", "neural", "rna", "crispr", "cas9", "cas12",
    "cas13", "genome", "gene", "editing", "off-target", "protein", "molecular",
    "catalyst", "selectivity", "reaction", "tomography", "transpilation",
    "clinical", "therapeutic", "algorithm", "network",
}

ANCHOR_TEXT_VARIANTS: dict[str, list[str]] = {
    "sequence modeling": ["sequence transduction", "sequence model", "sequence models"],
    "quantum error correction": ["qec", "quantum error-correction", "syndrome decoding", "surface code"],
    "wave function": ["wavefunction", "wave functions"],
    "neural quantum state": ["neural quantum states", "nqs"],
    "matrix product state": ["matrix product states", "mps"],
    "self-attention": ["self attention", "selfattention"],
    "guide rna": ["guide rnas", "grna", "grnas"],
    "off-target": ["off target", "off-target effects"],
}


def _norm_anchor_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _anchor_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+|\d+", _norm_anchor_text(value))
        if token not in TOO_GENERIC_TARGETS and len(token) >= 3
    }


def _is_generic_target_phrase(anchor: str) -> bool:
    al = _norm_anchor_text(anchor)
    if not al:
        return True
    if al in GENERIC_TARGET_PHRASES:
        return True
    if any(phrase in al for phrase in GENERIC_TARGET_PHRASES):
        return True
    words = al.replace("-", " ").split()
    if not words:
        return True
    if any(word in {"you", "your", "paper", "contribution", "fundamental", "direction", "directions", "insight", "insights", "teach", "want", "dive",
                     # Chinese discourse words
                     "方向", "建议", "角度", "方面", "思路", "展望", "目标", "工作"} for word in words):
        return True
    if all(word in TOO_GENERIC_TARGETS for word in words):
        return True
    return False


def _technicality_score(anchor: str, known_targets: set[str], method_set: set[str]) -> int:
    al = _norm_anchor_text(anchor)
    if not al or _is_generic_target_phrase(al) or al in method_set:
        return -3
    words = al.replace("-", " ").split()
    score = 0
    if al in known_targets:
        score += 4
    if any(al == known or al in known or known in al for known in known_targets):
        score += 2
    if "-" in al:
        score += 2
    if re.fullmatch(r"[a-z0-9]{2,8}(?:/[a-z0-9]{2,8})?", al) and al not in TOO_GENERIC_TARGETS:
        score += 2
    score += sum(1 for word in words if word in TECHNICAL_TARGET_HINTS)
    score -= sum(1 for word in words if word in TOO_GENERIC_TARGETS)
    if len(words) == 1 and score < 3:
        score -= 1
    return score


def _anchor_variants(anchor: str, include_semantic: bool = False) -> list[str]:
    al = _norm_anchor_text(anchor)
    variants = [al]
    normalized = ANCHOR_NORMALIZATION.get(al)
    if normalized:
        variants.append(normalized)
    for abbrev, full in ANCHOR_NORMALIZATION.items():
        if full == al:
            variants.append(abbrev)
        if abbrev == al:
            variants.append(full)
    if include_semantic:
        variants.extend(ANCHOR_TEXT_VARIANTS.get(al, []))
    return list(dict.fromkeys(_norm_anchor_text(v) for v in variants if v))


def _anchor_matches_evidence(anchor: str, evidence_lower: str, allow_loose: bool = False) -> bool:
    for variant in _anchor_variants(anchor, include_semantic=allow_loose):
        if variant and variant in evidence_lower:
            return True
    if not allow_loose:
        return False
    anchor_toks = _anchor_tokens(anchor)
    if not anchor_toks:
        return False
    evidence_toks = _anchor_tokens(evidence_lower)
    if not evidence_toks:
        return False
    overlap = anchor_toks & evidence_toks
    if len(anchor_toks) <= 2:
        return bool(overlap and any(tok in TECHNICAL_TARGET_HINTS for tok in overlap))
    return len(overlap) / len(anchor_toks) >= 0.6


def extract_required_anchors(claim: str) -> dict:
    """从 claim 中动态提取 method/target/relation 锚点和 specificity flags。

    不依赖预定义领域族，而是从 claim 文本自身识别三类锚点。

    Returns:
        {
            "method_anchors": list[str],
            "target_anchors": list[str],
            "relation_anchors": list[str],
            "specificity_flags": list[str],
        }
    """
    if not claim:
        return {
            "method_anchors": [],
            "target_anchors": [],
            "relation_anchors": [],
            "specificity_flags": [],
        }

    lowered = claim.lower()

    # --- method anchors: 匹配已知通用方法词表 ---
    method_anchors: list[str] = []
    sorted_methods = sorted(METHOD_TERMS, key=len, reverse=True)
    matched_positions: set[int] = set()
    for term in sorted_methods:
        idx = lowered.find(term)
        while idx >= 0:
            if not any(pos <= idx < pos + len(term) for pos in matched_positions):
                method_anchors.append(term.lower())
                for i in range(idx, idx + len(term)):
                    matched_positions.add(i)
            idx = lowered.find(term, idx + 1)

    # --- target anchors: 动态提取技术名词短语 ---
    target_anchors = _extract_target_anchors(claim, method_anchors)

    # --- relation anchors: 匹配已知关系词表 ---
    relation_anchors: list[str] = []
    sorted_relations = sorted(RELATION_TERMS, key=len, reverse=True)
    for term in sorted_relations:
        idx = lowered.find(term)
        if idx >= 0:
            rel = term.lower()
            if rel not in relation_anchors:
                relation_anchors.append(rel)

    # --- specificity flags ---
    specificity_flags: list[str] = []
    for flag_name, pattern in SPECIFICITY_PATTERNS.items():
        if re.search(pattern, lowered, re.IGNORECASE):
            specificity_flags.append(flag_name)

    return {
        "method_anchors": method_anchors,
        "target_anchors": target_anchors,
        "relation_anchors": relation_anchors,
        "specificity_flags": specificity_flags,
    }


def _extract_target_anchors(claim: str, method_anchors: list[str]) -> list[str]:
    """动态提取 target anchors: 技术名词短语，不依赖领域枚举。

    提取策略:
    1. 保留连续技术名词短语 (如 "quantum error correction")
    2. 保留大写缩写 (如 "NQS", "MPS", "CRISPR")
    3. 保留含连字符的技术短语 (如 "many-body", "off-target")
    4. 保留 "X of Y", "X for Y", "X-based Y", "Y decoding" 等结构
    5. 过滤在 TOO_GENERIC_TARGETS 或已在 method_anchors 中的词
    """
    if not claim:
        return []

    # 已知领域锚点作为白名单 (保持向后兼容); 只用于提供种子词
    known_targets = {
        "quantum", "quantum error correction", "quantum many-body",
        "many-body", "wave function", "neural quantum state",
        "neural quantum states", "nqs", "qec", "syndrome",
        "tensor network", "tensor networks", "matrix product state",
        "matrix product states", "mps", "variational monte carlo", "vmc",
        "hamiltonian", "ground state", "schrödinger", "schrodinger",
        "entanglement", "qubit", "qubits", "superposition",
        "quantum tomography", "surface code",
        "crispr", "cas9", "cas12", "cas13", "guide rna", "tracrrna",
        "crrna", "genome editing", "gene editing", "endonuclease",
        "off-target", "off-target effects", "base editing", "prime editing",
        "drug discovery", "protein folding", "molecular dynamics",
        "catalyst", "catalyst selectivity", "selectivity",
        "molecular property", "molecular design",
        "chemical reaction", "reaction prediction",
        "circuit transpilation", "transpilation",
        "clinical trial", "therapeutic",
        "superconducting qubit", "superconducting",
        "quantum computing", "quantum chemistry",
        "quantum simulation", "quantum algorithm",
    }

    text = claim.strip()
    lowered = text.lower()

    found: list[str] = []
    method_set = set(m.lower() for m in method_anchors)

    # 1. 已知锚点匹配 (优先长词)
    sorted_known = sorted(known_targets, key=len, reverse=True)
    occupied: set[int] = set()
    for anchor in sorted_known:
        idx = lowered.find(anchor)
        while idx >= 0:
            if not any(pos <= idx < pos + len(anchor) for pos in occupied):
                found.append(anchor)
                for i in range(idx, idx + len(anchor)):
                    occupied.add(i)
            idx = lowered.find(anchor, idx + 1)

    # 2. 提取大写缩写词 (2-8 个大写字母/数字，不在已知锚点中)
    acronym_pattern = re.compile(r"\b([A-Z]{2,8}(?:/[A-Z]{2,8})?)\b")
    for match in acronym_pattern.finditer(text):
        acronym = match.group(1).lower()
        if acronym not in found and acronym not in TOO_GENERIC_TARGETS:
            # 跳过已识别的通用词
            if acronym not in method_set and acronym not in {"the", "and", "for", "with", "from", "this", "that", "all", "new", "our"}:
                found.append(acronym)

    # 3. 提取含连字符的技术短语
    hyphen_phrase = re.compile(
        r"\b([a-z]+-[a-z]+(?:-[a-z]+)?(?:-[a-z]+)?)\b",
        re.IGNORECASE,
    )
    for match in hyphen_phrase.finditer(text):
        phrase = match.group(1).lower()
        if phrase not in found and phrase not in method_set and phrase not in TOO_GENERIC_TARGETS:
            # 检查是否包含技术含义 (不只是 "state-of-the-art")
            if _technicality_score(phrase, known_targets, method_set) >= 1:
                found.append(phrase)

    # 4. 提取多词名词短语: [A-Z][a-z]+ [a-z]+ [A-Z][a-z]+ 等模式
    #   或 [a-z]+ [a-z]+ 的已知/高信息量组合
    # 尝试提取 "X Y" 形式的技术短语 (连续 2-3 个实词)
    noun_phrase = re.compile(
        r"\b([a-z]{3,}(?:\s+[a-z]{3,}){1,2})\b"
    )
    for match in noun_phrase.finditer(lowered):
        phrase = match.group(1).strip()
        words = phrase.split()
        if len(words) < 2:
            continue
        if phrase in found:
            continue
        if phrase in method_set:
            continue
        if all(w in TOO_GENERIC_TARGETS for w in words):
            continue
        if _technicality_score(phrase, known_targets, method_set) >= 2:
            # 验证: 不在已有的 method phrase 中
            is_contained = any(
                phrase in existing or existing in phrase
                for existing in found
            )
            if not is_contained:
                found.append(phrase)

    # 5. 过滤: 去掉太泛的词 + method anchors
    filtered: list[str] = []
    for anchor in found:
        al = anchor.lower()
        # 单字跳过
        if len(al) < 3 and not al.isupper():
            continue
        # 纯 method 词跳过
        if al in method_set:
            continue
        # 太泛跳过
        if al in TOO_GENERIC_TARGETS:
            continue
        if _technicality_score(al, known_targets, method_set) < 1:
            continue
        # 去重 (最长优先保留)
        is_dup = any(
            al != existing and (al in existing)
            for existing in [f.lower() for f in filtered]
        )
        if not is_dup:
            filtered.append(anchor)

    return filtered[:20]


def evaluate_required_anchor_coverage(
    claim: str,
    evidence_text: str,
    evidence_ids: list[str] | None = None,
    evidence_pool: dict | list | None = None,
) -> dict:
    """检查 evidence 是否覆盖 claim 的必要锚点。

    Returns:
        {
            "method_anchors": list[str],
            "target_anchors": list[str],
            "relation_anchors": list[str],
            "matched_method_anchors": list[str],
            "matched_target_anchors": list[str],
            "matched_relation_anchors": list[str],
            "missing_target_anchors": list[str],
            "missing_relation_anchors": list[str],
            "method_only_match": bool,
            "target_coverage": float,
            "relation_coverage": float,
            "coverage_status": str,
        }
    """
    required = extract_required_anchors(claim)
    method_anchors = required["method_anchors"]
    target_anchors = required["target_anchors"]
    relation_anchors = required["relation_anchors"]

    ev_lower = _norm_anchor_text(evidence_text or "")
    evidence_ids = normalize_id_list(evidence_ids)
    allow_loose_target_match = any(eid.startswith("E") for eid in evidence_ids)

    # 匹配检查
    matched_method = [a for a in method_anchors if _anchor_matches_evidence(a, ev_lower, allow_loose=False)]
    matched_target = [a for a in target_anchors if _anchor_matches_evidence(a, ev_lower, allow_loose=allow_loose_target_match)]
    matched_relation = [a for a in relation_anchors if _anchor_matches_evidence(a, ev_lower, allow_loose=False)]

    missing_target = [a for a in target_anchors if a not in matched_target]
    missing_relation = [a for a in relation_anchors if a not in matched_relation]

    target_cov = len(matched_target) / len(target_anchors) if target_anchors else 1.0
    relation_cov = len(matched_relation) / len(relation_anchors) if relation_anchors else 1.0

    method_only = bool(
        matched_method and not matched_target and target_anchors
    )

    # 判定 coverage_status
    if target_anchors and not matched_target:
        # evidence 完全没有覆盖 claim 的目标对象
        if matched_method and not matched_target:
            coverage_status = "method_only"
        else:
            coverage_status = "missing_required_target"
    elif target_anchors and matched_target:
        if matched_relation and target_cov >= 0.5:
            coverage_status = "direct"
        elif matched_relation or target_cov >= 0.3:
            coverage_status = "partial"
        else:
            coverage_status = "background_only"
    elif not target_anchors:
        # claim 没有明显的 target anchors → 退回到 method+relation 检查
        if matched_method and matched_relation:
            coverage_status = "direct"
        elif matched_method or matched_relation:
            coverage_status = "partial"
        else:
            coverage_status = "background_only"
    else:
        coverage_status = "background_only"

    return {
        "method_anchors": method_anchors,
        "target_anchors": target_anchors,
        "relation_anchors": relation_anchors,
        "matched_method_anchors": matched_method,
        "matched_target_anchors": matched_target,
        "matched_relation_anchors": matched_relation,
        "missing_target_anchors": missing_target,
        "missing_relation_anchors": missing_relation,
        "method_only_match": method_only,
        "target_coverage": round(target_cov, 3),
        "relation_coverage": round(relation_cov, 3),
        "coverage_status": coverage_status,
        "specificity_flags": required.get("specificity_flags", []),
        "allow_loose_target_match": allow_loose_target_match,
    }
