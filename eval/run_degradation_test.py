"""
Degradation Test -- Bonus: Resilience Under Reduced Knowledge

Removes ~50 % of the FAISS vector-store chunks (every other chunk)
and reruns the evaluation question set to measure:

  1. Does the agent fall back to web_search?
  2. Does it hallucinate (answer without citations)?
  3. Does it refuse more often?

Usage:
    python eval/run_degradation_test.py
"""

import sys
import json
import shutil
import time
import os
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss

# ── Paths ──
INDEX_DIR   = Path("data/faiss_index")
BACKUP_DIR  = Path("data/faiss_index_backup")
TRACES_DIR  = Path("traces/degradation")

# ── Evaluation questions (same as EVALUATION.md Q1-Q20) ──
QUESTIONS = [
    # Single-tool: search_docs (Q1-Q6)
    ("What is Project Maximus and what are its key goals?", "search_docs"),
    ("What did the Infosys CEO say about generative AI in FY24?", "search_docs"),
    ("What was the main reason Infosys gave for revenue growth in FY24?", "search_docs"),
    ("What strategic priorities did Infosys highlight in FY24?", "search_docs"),
    ("How many new customers did Infosys add in FY24?", "search_docs"),
    ("What is Infosys capital allocation policy?", "search_docs"),
    # Single-tool: query_data (Q7-Q12)
    ("What was Infosys operating margin in FY24?", "query_data"),
    ("What was Infosys revenue growth from FY21 to FY24?", "query_data"),
    ("Which company had the highest operating margin in FY24?", "query_data"),
    ("What was Infosys EPS in FY24?", "query_data"),
    ("How did Infosys headcount change from FY23 to FY24?", "query_data"),
    ("What was Wipro revenue trend over 4 years?", "query_data"),
    # Multi-tool (Q13-Q16)
    ("How did Infosys and TCS operating margins compare and what drove the difference?", "multi"),
    ("What was Infosys headcount in FY24 and what is their workforce strategy?", "multi"),
    ("Compare revenue growth across all 3 companies and explain Infosys strategy", "multi"),
    ("What was Infosys net profit in FY24 and what factors did management cite?", "multi"),
    # Refusal / Edge (Q17-Q20)
    ("Should I invest in Infosys stock right now?", "refuse"),
    ("What was Infosys revenue in FY2015?", "out_of_range"),
    ("What is the airspeed velocity of an unladen swallow?", "irrelevant"),
    ("Tell me everything about Infosys Q1 FY25 results in complete detail including all segment breakdowns.", "web_search"),
]


def _citation_keywords():
    return ["Source:", "Page:", "URL:", "financials.csv", ".pdf", "http"]


def _has_citation(answer: str) -> bool:
    return any(kw in answer for kw in _citation_keywords())


def _classify_answer(answer: str, trace: list, result: dict, expected_tool: str) -> dict:
    """Classify the answer into behavioral categories."""
    tools_used  = [t["tool"] for t in trace]
    answer_l    = (answer or "").lower()
    has_cite    = _has_citation(answer or "")
    refused     = result.get("refused", False)
    steps       = result.get("steps_used", 0)
    used_web    = "web_search" in tools_used
    cap_hit     = result.get("cap_hit", False)

    # Did it hallucinate? (gave a non-empty answer but no citations for Q1-Q16)
    hallucinated = False
    if expected_tool not in ("refuse", "irrelevant"):
        if answer and answer.strip() and not has_cite and not refused:
            hallucinated = True

    # Did it fall back to web search when it shouldn't?
    web_fallback = used_web and expected_tool in ("search_docs", "query_data", "multi")

    # Did it refuse when it shouldn't?
    false_refuse = refused and expected_tool not in ("refuse", "irrelevant")

    # Did it admit lack of data?
    admitted_no_data = any(kw in answer_l for kw in [
        "not available", "no data", "no relevant", "couldn't find",
        "no documents", "unable to find", "no structured data",
        "don't have", "outside", "cannot find",
    ])

    return {
        "tools_used":     tools_used,
        "steps":          steps,
        "has_citation":   has_cite,
        "hallucinated":   hallucinated,
        "web_fallback":   web_fallback,
        "false_refuse":   false_refuse,
        "admitted_no_data": admitted_no_data,
        "refused":        refused,
        "cap_hit":        cap_hit,
    }


# ──────────────────────────────────────────────────────────────────────
# Index manipulation helpers
# ──────────────────────────────────────────────────────────────────────
def backup_index():
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(INDEX_DIR, BACKUP_DIR)
    print(f"[OK] Full index backed up to {BACKUP_DIR}")


def restore_index():
    if BACKUP_DIR.exists():
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        shutil.copytree(BACKUP_DIR, INDEX_DIR)
        shutil.rmtree(BACKUP_DIR)
        print(f"[OK] Full index restored from backup")


def create_degraded_index():
    """Remove every other chunk (≈50 %) from the FAISS index + metadata."""
    meta_path  = INDEX_DIR / "metadata.json"
    index_path = INDEX_DIR / "index.faiss"

    with open(meta_path, "r") as f:
        data = json.load(f)

    chunks   = data["chunks"]
    metadata = data["metadata"]
    total    = len(chunks)

    # Keep only even-indexed chunks → ~50 % removal
    keep_idx       = list(range(0, total, 2))
    new_chunks     = [chunks[i] for i in keep_idx]
    new_metadata   = [metadata[i] for i in keep_idx]
    removed_count  = total - len(keep_idx)

    print(f"[OK] Degradation: {total} -> {len(keep_idx)} chunks "
          f"(removed {removed_count}, {removed_count/total*100:.0f} %)")

    # Rebuild embeddings from the kept chunks
    from sentence_transformers import SentenceTransformer
    model      = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(new_chunks)
    dimension  = embeddings.shape[1]

    new_index = faiss.IndexFlatL2(dimension)
    new_index.add(np.array(embeddings).astype("float32"))

    # Overwrite index on disk
    faiss.write_index(new_index, str(index_path))
    with open(meta_path, "w") as f:
        json.dump({"metadata": new_metadata, "chunks": new_chunks}, f)

    return total, len(keep_idx), removed_count


# ──────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────
def run_single_pass(label: str) -> list:
    """Run all QUESTIONS once and return a list of result dicts."""
    # Force reload of cached index inside search_docs
    import src.tools.search_docs as _sd
    _sd._index_cache = {}

    from src.agent import run_agent, write_trace

    results = []
    out_dir = TRACES_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (q, expected) in enumerate(QUESTIONS):
        q_short = q[:50].ljust(50)
        try:
            res     = run_agent(q, enable_planning=True, enable_reflection=True, verbose=False)
            answer  = res.get("answer", "") or ""
            trace   = res.get("trace", [])
            cls     = _classify_answer(answer, trace, res, expected)
            cls["question"]      = q
            cls["expected_tool"] = expected
            cls["answer_short"]  = answer[:200]
            results.append(cls)

            write_trace(q, res, str(out_dir / f"q{i+1}.txt"))

            status = "CITE" if cls["has_citation"] else ("REFUSE" if cls["refused"] else "NO_CITE")
            halluc = " !HALLUC" if cls["hallucinated"] else ""
            webfb  = " >WEB"    if cls["web_fallback"] else ""
            print(f"  Q{i+1:>2} [{status:>7}{halluc}{webfb}] {q_short}")
        except Exception as e:
            print(f"  Q{i+1:>2} [ ERROR ] {q_short}  — {e}")
            results.append({
                "question": q, "expected_tool": expected,
                "tools_used": [], "steps": 0, "has_citation": False,
                "hallucinated": False, "web_fallback": False,
                "false_refuse": False, "admitted_no_data": False,
                "refused": False, "cap_hit": False,
                "answer_short": f"ERROR: {e}",
                "error": True,
            })

    return results


def compare_results(full: list, degraded: list):
    """Print a side-by-side comparison table and summary."""
    print("\n" + "=" * 120)
    print(f"{'Q#':>3}  {'Expected':>12}  {'Full->Tools':>20}  {'Degr->Tools':>20}  "
          f"{'Full Cite':>9}  {'Degr Cite':>9}  {'Halluc':>6}  {'Web FB':>6}  {'Delta Notes'}")
    print("-" * 120)

    full_cites   = 0; degr_cites   = 0
    full_halluc  = 0; degr_halluc  = 0
    full_webfb   = 0; degr_webfb   = 0
    full_refuse  = 0; degr_refuse  = 0
    degr_nodata  = 0

    notes_list = []

    for i, (f, d) in enumerate(zip(full, degraded)):
        f_tools = ",".join(f["tools_used"]) or "—"
        d_tools = ",".join(d["tools_used"]) or "—"
        f_cite  = "Y" if f["has_citation"] else "N"
        d_cite  = "Y" if d["has_citation"] else "N"
        halluc  = "!" if d["hallucinated"] and not f["hallucinated"] else ("-" if not d["hallucinated"] else "!(both)")
        webfb   = ">" if d["web_fallback"] and not f["web_fallback"] else "-"

        notes = []
        if d["web_fallback"] and not f["web_fallback"]:
            notes.append("fell back to web")
        if d["hallucinated"] and not f["hallucinated"]:
            notes.append("NEW hallucination")
        if d["false_refuse"] and not f["false_refuse"]:
            notes.append("NEW false refuse")
        if d["admitted_no_data"] and not f["admitted_no_data"]:
            notes.append("admitted no data")
        if not d["has_citation"] and f["has_citation"]:
            notes.append("lost citation")

        full_cites  += int(f["has_citation"])
        degr_cites  += int(d["has_citation"])
        full_halluc += int(f["hallucinated"])
        degr_halluc += int(d["hallucinated"])
        full_webfb  += int(f["web_fallback"])
        degr_webfb  += int(d["web_fallback"])
        full_refuse += int(f["refused"])
        degr_refuse += int(d["refused"])
        degr_nodata += int(d["admitted_no_data"])

        note_str = "; ".join(notes) if notes else ""
        if notes:
            notes_list.append((i + 1, note_str))

        print(f"{i+1:>3}  {f['expected_tool']:>12}  {f_tools:>20}  {d_tools:>20}  "
              f"{f_cite:>9}  {d_cite:>9}  {halluc:>6}  {webfb:>6}  {note_str}")

    print("-" * 120)
    total = len(full)
    print(f"\n{'Metric':<35}  {'Full Index':>12}  {'Degraded':>12}  {'Delta':>6}")
    print("-" * 70)
    print(f"{'Questions with citations':<35}  {full_cites:>12}  {degr_cites:>12}  {degr_cites-full_cites:>+6}")
    print(f"{'Hallucinations (no-cite answers)':<35}  {full_halluc:>12}  {degr_halluc:>12}  {degr_halluc-full_halluc:>+6}")
    print(f"{'Web search fallbacks':<35}  {full_webfb:>12}  {degr_webfb:>12}  {degr_webfb-full_webfb:>+6}")
    print(f"{'Refusals':<35}  {full_refuse:>12}  {degr_refuse:>12}  {degr_refuse-full_refuse:>+6}")
    print(f"{'Admitted no data (degraded only)':<35}  {'--':>12}  {degr_nodata:>12}  {'':>6}")

    return {
        "total": total,
        "full_cites": full_cites, "degr_cites": degr_cites,
        "full_halluc": full_halluc, "degr_halluc": degr_halluc,
        "full_webfb": full_webfb, "degr_webfb": degr_webfb,
        "full_refuse": full_refuse, "degr_refuse": degr_refuse,
        "degr_nodata": degr_nodata,
        "notes": notes_list,
    }


def save_report(stats: dict, full_results: list, degr_results: list,
                orig_chunks: int, kept_chunks: int, removed_chunks: int):
    """Write a JSON report alongside the traces."""
    report = {
        "index_degradation": {
            "original_chunks": orig_chunks,
            "kept_chunks": kept_chunks,
            "removed_chunks": removed_chunks,
            "removal_pct": round(removed_chunks / orig_chunks * 100, 1),
        },
        "summary": stats,
        "full_results": full_results,
        "degraded_results": degr_results,
    }
    report_path = TRACES_DIR / "degradation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OK] Detailed report saved to {report_path}")
    return report


# ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DEGRADATION TEST — Agentic RAG System")
    print("  Remove ~50 % of vector-store chunks and compare agent behavior")
    print("=" * 70)

    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Run with full index ──
    print("\n>> Phase 1/3: Running evaluation with FULL index...")
    full_results = run_single_pass("full")

    # ── Phase 2: Create degraded index and run ──
    print("\n>> Phase 2/3: Creating degraded index (~50 % chunks removed)...")
    backup_index()

    try:
        orig, kept, removed = create_degraded_index()

        print("\n>> Running evaluation with DEGRADED index...")
        degr_results = run_single_pass("degraded")
    finally:
        # Always restore
        print("\n>> Phase 3/3: Restoring original index...")
        restore_index()
        # Re-invalidate cache so later runs use the restored index
        import src.tools.search_docs as _sd
        _sd._index_cache = {}

    # ── Compare ──
    stats = compare_results(full_results, degr_results)
    save_report(stats, full_results, degr_results, orig, kept, removed)

    print("\n[DONE] Degradation test complete. Traces in traces/degradation/")
    print("   Add findings to EVALUATION.md for bonus points!")


if __name__ == "__main__":
    main()
