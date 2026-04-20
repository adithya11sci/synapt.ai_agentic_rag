import json
from src.agent import run_agent

queries = [
    "What was Infosys operating margin in FY24?",
    "What is Project Maximus and what are its five pillars?",
    "Infosys stock price today",
    "Should I invest in Infosys stock right now?",
    "How did Infosys and TCS operating margins compare in FY24 and what drove each result?"
]

print("=== STARTING AGENT TESTS ===")
for q in queries:
    with open("final_test_output.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- Question: {q} ---\n")
        print(f"\n--- Question: {q} ---")
        res = run_agent(q)
        tools_called = [t['tool'] for t in res.get('trace', [])]
        f.write(f"TOOLS CALLED: {tools_called}\n")
        f.write(f"ANSWER: {res['answer']}\n")
        f.write(f"STEPS: {res['steps_used']}\n")
        print(f"TOOLS CALLED: {tools_called}")
        print(f"ANSWER: {res['answer']}".encode('utf-8', 'ignore').decode('utf-8'))
        print(f"STEPS: {res['steps_used']}")
print("=== TESTS COMPLETE ===")