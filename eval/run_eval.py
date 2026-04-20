from src.agent import run_agent, write_trace
from pathlib import Path

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
    print(f"{'Q number':^10} | {'Question (first 40 chars)':^40} | {'Tools Called':^15} | {'Steps':^10} | {'Pass/Fail':^10}")
    print("-" * 95)
    
    for i, q in enumerate(questions):
        try:
            res = run_agent(q)
            trace_path = f"traces/q{i+1}.txt"
            write_trace(q, res, trace_path)
            
            tools_called = ", ".join([t['tool'] for t in res.get('trace', [])])
            steps = res.get("steps_used", 0)
            
            q_short = q[:40] if len(q) > 40 else q.ljust(40)
            
            # Simple pass/fail logic
            pass_fail = "PASS" if not res.get("refused") or i in [16, 17, 18, 19] else "FAIL"
            
            print(f"{i+1:^10} | {q_short} | {tools_called[:15] if tools_called else 'None':^15} | {steps:^10} | {pass_fail:^10}")
        except Exception as e:
            print(f"{i+1:^10} | {q[:40]} | {'ERROR':^15} | {'-':^10} | {'FAIL':^10}")

if __name__ == "__main__":
    run_eval()
