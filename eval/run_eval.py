import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent import run_agent, write_trace

def run_eval():
    """Run 20 questions evaluation."""
    questions = [
        "What did the Infosys CEO say about generative AI in FY24?",
        "What was the main reason Infosys gave for revenue growth FY24?",
        "Tell me everything about Infosys Q1 FY25 results in complete detail including all segment breakdowns."
    ]

    improved_dir = Path("traces") / "improved"
    improved_dir.mkdir(parents=True, exist_ok=True)

    def grade(i, res):
        tools_called = ", ".join([t['tool'] for t in res.get('trace', [])])
        steps = res.get("steps_used", 0)
        answer = res.get("answer", "") or ""

        if i == 16:
            return "PASS" if res.get("refused") and steps == 0 else "FAIL"
        elif i == 17:
            return "PASS" if any(kw in answer.lower() for kw in ["not available", "no data", "no structured data", "don't have", "outside"]) else "FAIL"
        elif i == 18:
            return "PASS" if any(kw in answer.lower() for kw in ["not available", "cannot", "no data", "not related", "outside", "don't have"]) else "FAIL"
        elif i == 19:
            return "PASS" if res.get("cap_hit") or "web_search" in tools_called else "FAIL"
        else:
            has_citation = any(kw in answer for kw in ["Source:", "Page:", "URL:", "financials.csv", ".pdf", "http"])
            return "PASS" if answer.strip() and has_citation else "FAIL"

    print(
        f"{'Q#':^5} | {'Question (first 40 chars)':<40} | "
        f"{'Baseline':^8} | {'Plan':^8} | {'Plan+Refl':^10} | {'Steps':^7} | {'Tools (improved)':<20}"
    )
    print("Note: This run executes baseline + plan-only + plan+reflection per question (slower by design).")
    print("-" * 115)

    baseline_summary = []
    plan_summary = []
    improved_summary = []

    for i, q in enumerate(questions):
        try:
            # Baseline: no planning, no reflection (not written to disk)
            res_baseline = run_agent(q, enable_planning=False, enable_reflection=False, verbose=False)
            pf_baseline = grade(i, res_baseline)
            baseline_summary.append(pf_baseline)

            # Plan-only: planning enabled, reflection disabled (not written to disk)
            res_plan = run_agent(q, enable_planning=True, enable_reflection=False, verbose=False)
            pf_plan = grade(i, res_plan)
            plan_summary.append(pf_plan)

            # Improved: planning + reflection enabled (written to traces/improved)
            print(f"\n=== Improved run Q{i+1} ===")
            res_improved = run_agent(q, enable_planning=True, enable_reflection=True, verbose=True)
            pf_improved = grade(i, res_improved)
            improved_summary.append(pf_improved)

            # Attach eval summary to improved trace (for quick audit)
            res_improved["eval_summary"] = {
                "baseline": pf_baseline,
                "plan_only": pf_plan,
                "plan_plus_reflection": pf_improved,
            }

            trace_path = improved_dir / f"q{i+1}.txt"
            write_trace(q, res_improved, str(trace_path))

            tools_called = ", ".join([t['tool'] for t in res_improved.get('trace', [])])
            steps = res_improved.get("steps_used", 0)
            q_short = q[:40] if len(q) > 40 else q.ljust(40)

            print(
                f"{i+1:^5} | {q_short} | {pf_baseline:^8} | {pf_plan:^8} | {pf_improved:^10} | "
                f"{steps:^7} | {tools_called[:20] if tools_called else 'None':<20}"
            )

        except Exception as e:
            q_short = q[:40] if len(q) > 40 else q.ljust(40)
            print(f"{i+1:^5} | {q_short:<40} | {'ERROR':^8} | {'ERROR':^8} | {'ERROR':^10} | {'-':^7} | {'-':<20}")
            baseline_summary.append("FAIL")
            plan_summary.append("FAIL")
            improved_summary.append("FAIL")

    b_passed = baseline_summary.count("PASS")
    p_passed = plan_summary.count("PASS")
    i_passed = improved_summary.count("PASS")
    total = len(questions)

    print("-" * 115)
    print(f"Baseline accuracy:         {b_passed}/{total}")
    print(f"Planning-only accuracy:    {p_passed}/{total}  (Δ vs baseline: {p_passed - b_passed:+d})")
    print(f"Plan+Reflection accuracy:  {i_passed}/{total}  (Δ vs plan-only: {i_passed - p_passed:+d})")
    print(f"Improved traces written to: {improved_dir}")

if __name__ == "__main__":
    run_eval()
