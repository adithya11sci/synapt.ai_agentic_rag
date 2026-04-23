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
        "What is Project Maximus and what are its five pillars?",
        "What did the Infosys CEO say about generative AI in FY24?",
        "What was the main reason Infosys gave for revenue growth FY24?",
        "What strategic priorities did Infosys highlight in FY24 MD&A?",
        "How many new customers did Infosys add in FY24?",
        "What is Infosys capital allocation policy?",
        "What was Infosys operating margin in FY24?",
        "What was Infosys revenue growth from FY21 to FY24?",
        "Which company had the highest operating margin in FY24?",
        "What was Infosys EPS in FY24?",
        "How did Infosys headcount change from FY23 to FY24?",
        "What was Wipro revenue trend over 4 years?",
        "How did Infosys and TCS operating margins compare in FY24 and what drove each result?",
        "What was Infosys headcount in FY24 and what does the company say about its workforce strategy?",
        "Compare revenue growth across all 3 companies over 4 years and explain Infosys strategy for growth.",
        "What was Infosys net profit in FY24 and what factors did management cite for it?",
        "Should I invest in Infosys stock right now?",
        "What was Infosys revenue in FY2015?",
        "What is the airspeed velocity of an unladen swallow?",
        "Tell me everything about Infosys Q1 FY25 results in complete detail including all segment breakdowns."
    ]

    Path("traces").mkdir(exist_ok=True)
    print(f"{'Q#':^5} | {'Question (first 40 chars)':<40} | {'Tools Called':<20} | {'Steps':^7} | {'Result':^8}")
    print("-" * 95)

    results_summary = []

    for i, q in enumerate(questions):
        try:
            res = run_agent(q)
            trace_path = f"traces/q{i+1}.txt"
            write_trace(q, res, trace_path)

            tools_called = ", ".join([t['tool'] for t in res.get('trace', [])])
            steps = res.get("steps_used", 0)
            answer = res.get("answer", "")

            q_short = q[:40] if len(q) > 40 else q.ljust(40)

            # Pass/fail logic matching assignment criteria:
            # Q17 (i=16): Refusal — PASS if refused=True and steps_used=0
            # Q18 (i=17): Out-of-range data — PASS if agent says data not available
            # Q19 (i=18): Irrelevant question — PASS if agent says data not available
            # Q20 (i=19): Overload — PASS if cap_hit=True or web_search was called
            # Q1-Q16: Normal — PASS if answer is not empty and contains a citation
            if i == 16:
                pass_fail = "PASS" if res.get("refused") and steps == 0 else "FAIL"
            elif i == 17:
                pass_fail = "PASS" if any(kw in answer.lower() for kw in ["not available", "no data", "no structured data", "don't have", "outside"]) else "FAIL"
            elif i == 18:
                pass_fail = "PASS" if any(kw in answer.lower() for kw in ["not available", "cannot", "no data", "not related", "outside", "don't have"]) else "FAIL"
            elif i == 19:
                pass_fail = "PASS" if res.get("cap_hit") or "web_search" in tools_called else "FAIL"
            else:
                has_citation = any(kw in answer for kw in ["Source:", "Page:", "URL:", "financials.csv", ".pdf", "http"])
                pass_fail = "PASS" if answer.strip() and has_citation else "FAIL"

            print(f"{i+1:^5} | {q_short} | {tools_called[:20] if tools_called else 'None':<20} | {steps:^7} | {pass_fail:^8}")
            results_summary.append(pass_fail)

        except Exception as e:
            print(f"{i+1:^5} | {q[:40]:<40} | {'ERROR':<20} | {'-':^7} | {'FAIL':^8}")
            results_summary.append("FAIL")

    passed = results_summary.count("PASS")
    total = len(results_summary)
    print("-" * 95)
    print(f"Total: {passed}/{total} passed")

if __name__ == "__main__":
    run_eval()
