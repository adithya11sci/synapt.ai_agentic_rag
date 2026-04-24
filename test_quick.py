import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent import run_agent, write_trace

queries = [
    "How did Infosys' and TCS' operating margins compare in FY24, and what drove each?",
    "how to make pizza dough",
    "what is the name of the monkey in the movie 'the monkey king'?"
]

Path("traces").mkdir(exist_ok=True)

print("=== STARTING AGENT TESTS ===")
for i, q in enumerate(queries):
    print(f"\n--- Question {i+1}: {q} ---")
    res = run_agent(q)
    tools_called = [t['tool'] for t in res.get('trace', [])]
    print(f"TOOLS CALLED: {tools_called}")
    print(f"ANSWER: {res['answer']}".encode('utf-8', 'ignore').decode('utf-8'))
    print(f"STEPS: {res['steps_used']} / 8")

    # Save trace
    write_trace(q, res, f"traces/quick_test_q{i+1}.txt")

print("\n=== TESTS COMPLETE ===")