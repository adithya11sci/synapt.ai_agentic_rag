import sys
import os
from pathlib import Path

# Suppress TensorFlow logging to avoid cluttering the terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent import run_agent, write_trace

queries = [
    "What strategic priorities did Infosys highlight in FY24 MD&A?",
    "How many new customers did Infosys add in FY24?",
    "What is Infosys capital allocation policy?",
    "what is the name of the monkey in the movie 'the monkey king'?"
]

Path("traces").mkdir(exist_ok=True)

print("=== STARTING AGENT TESTS ===")
for i, q in enumerate(queries):
    print(f"\n--- Question {i+1}: {q} ---")
    res = run_agent(q, enable_planning=True, enable_reflection=True, verbose=True)
    tools_called = [t['tool'] for t in res.get('trace', [])]
    print(f"TOOLS CALLED: {tools_called}")
    print(f"ANSWER: {res['answer']}".encode('utf-8', 'ignore').decode('utf-8'))
    print(f"STEPS: {res['steps_used']} / 8")

    # Save trace
    write_trace(q, res, f"traces/quick_test_q{i+1}.txt")

print("\n=== TESTS COMPLETE ===")