# C:\Users\18449\Desktop\researchguard_workspace\scripts\repair_researchguard.py
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(r"C:\Users\18449\Desktop\researchguard_workspace")
PACKAGE_DIR = ROOT / "researchguard"

HEADER_PREFIXES = (
    "# C:\\Users\\18449\\Desktop\\researchguard_workspace\\",
    "#C:\\Users\\18449\\Desktop\\researchguard_workspace\\",
)


def ensure_dirs() -> None:
    dirs = [
        PACKAGE_DIR,
        PACKAGE_DIR / "ingestion",
        PACKAGE_DIR / "indexing",
        PACKAGE_DIR / "retrieval",
        PACKAGE_DIR / "agent",
        PACKAGE_DIR / "evaluation",
        PACKAGE_DIR / "audit",
        PACKAGE_DIR / "memory",
        PACKAGE_DIR / "reporting",
        PACKAGE_DIR / "api",
        PACKAGE_DIR / "parsers",
        ROOT / "configs",
        ROOT / "data",
        ROOT / "data" / "eval",
        ROOT / "data" / "raw_docs",
        ROOT / "data" / "parsed",
        ROOT / "data" / "indexes",
        ROOT / "frontend",
        ROOT / "tests",
        ROOT / "outputs",
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    init_files = [
        PACKAGE_DIR / "__init__.py",
        PACKAGE_DIR / "ingestion" / "__init__.py",
        PACKAGE_DIR / "indexing" / "__init__.py",
        PACKAGE_DIR / "retrieval" / "__init__.py",
        PACKAGE_DIR / "agent" / "__init__.py",
        PACKAGE_DIR / "evaluation" / "__init__.py",
        PACKAGE_DIR / "audit" / "__init__.py",
        PACKAGE_DIR / "memory" / "__init__.py",
        PACKAGE_DIR / "reporting" / "__init__.py",
        PACKAGE_DIR / "api" / "__init__.py",
        PACKAGE_DIR / "parsers" / "__init__.py",
    ]

    for file in init_files:
        file.touch(exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[WARN] missing source file: {src}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPY] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def copy_tree_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        print(f"[WARN] missing source dir: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in src_dir.rglob("*"):
        if src.is_file():
            relative = src.relative_to(src_dir)
            dst = dst_dir / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"[COPY DIR] {src_dir.relative_to(ROOT)} -> {dst_dir.relative_to(ROOT)}")


def restore_from_legacy_projects() -> None:
    rag = ROOT / "rag_agent_harness"
    claw = ROOT / "EvidenceClaw"

    copy_tree_files(rag / "src" / "parse", PACKAGE_DIR / "ingestion")
    copy_file(rag / "src" / "indexing" / "build_index.py", PACKAGE_DIR / "indexing" / "index_builder.py")
    copy_file(rag / "src" / "agentic" / "agentic_rag.py", PACKAGE_DIR / "agent" / "legacy_agentic_rag.py")
    copy_tree_files(rag / "src" / "eval", PACKAGE_DIR / "evaluation")

    if (rag / "data" / "eval").exists():
        copy_tree_files(rag / "data" / "eval", ROOT / "data" / "eval")

    if (rag / "configs").exists():
        copy_tree_files(rag / "configs", ROOT / "configs")

    copy_file(rag / "app.py", ROOT / "frontend" / "streamlit_app.py")

    audit_files = [
        "base_skill.py",
        "paper_claim_extraction_skill.py",
        "paper_error_audit_skill.py",
        "reference_support_check_skill.py",
        "citation_in_paper_check_skill.py",
        "numerical_claim_check_skill.py",
        "internal_consistency_check_skill.py",
        "evidence_verdict_validator.py",
        "source_evidence_extract_skill.py",
        "evidence_table_build_skill.py",
    ]

    for name in audit_files:
        copy_file(claw / "src" / "skills" / name, PACKAGE_DIR / "audit" / name)

    copy_file(claw / "src" / "text_utils_v2.py", PACKAGE_DIR / "text_utils_v2.py")
    copy_file(claw / "src" / "models.py", PACKAGE_DIR / "schemas.py")
    copy_file(claw / "src" / "utils_ids.py", PACKAGE_DIR / "utils_ids.py")
    copy_file(claw / "src" / "parsers" / "reference_parser.py", PACKAGE_DIR / "parsers" / "reference_parser.py")

    copy_tree_files(claw / "src" / "memory", PACKAGE_DIR / "memory")
    copy_tree_files(claw / "src" / "renderers", PACKAGE_DIR / "reporting")


def remove_existing_header(content: str) -> str:
    content = content.lstrip("\ufeff")

    lines = content.splitlines(keepends=True)

    if not lines:
        return ""

    first_line = lines[0].strip()

    if first_line.startswith(HEADER_PREFIXES):
        return "".join(lines[1:])

    return content


def rewrite_imports(content: str) -> str:
    replacements = {
        "from src.skills.": "from researchguard.audit.",
        "from src.models": "from researchguard.schemas",
        "from src.text_utils_v2": "from researchguard.text_utils_v2",
        "from src.utils_ids": "from researchguard.utils_ids",
        "from src.parsers.": "from researchguard.parsers.",
        "from src.renderers.": "from researchguard.reporting.",
        "from src.agentic.agentic_rag": "from researchguard.agent.legacy_agentic_rag",
        "from src.eval.": "from researchguard.evaluation.",
        "from src.memory import MemoryStore": "from researchguard.memory.memory_store import MemoryStore",
        "from src.memory.": "from researchguard.memory.",
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content


def add_header_and_fix_imports() -> None:
    for file in PACKAGE_DIR.rglob("*.py"):
        try:
            content = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file.read_text(encoding="utf-8-sig")

        content = remove_existing_header(content)
        content = rewrite_imports(content)

        header = f"# {file}\n"
        file.write_text(header + content, encoding="utf-8", newline="\n")

    print("[OK] headers and imports fixed.")


def main() -> None:
    ensure_dirs()
    restore_from_legacy_projects()
    add_header_and_fix_imports()
    print("[DONE] ResearchGuard repair finished.")


if __name__ == "__main__":
    main()