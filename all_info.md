# Comprehensive Overview: Agentic RAG Financial Research Agent

## 1. Project Overview
The **Agentic RAG Financial Research Agent** is a highly specialized, autonomous AI system designed to answer complex, multi-faceted questions regarding three major IT companies: Infosys, TCS, and Wipro. 

Unlike traditional Chatbots that rely solely on pre-trained knowledge or simple RAG that only does vector lookups, this system is **"Agentic"**. It acts as an autonomous researcher that can interact with three distinct data sources (Unstructured PDFs, Structured CSVs, and the Live Web), reason about what data it needs, and combine multiple tools to formulate precise, cited answers.

---

## 2. System Architecture & The "Agent Loop" (How it Works)

The architecture deliberately avoids opaque orchestrator frameworks like LangChain or LlamaIndex. Instead, it utilizes a fully transparent **Python `while` loop** to drive the LLM's reasoning engine.

### The Step-by-Step Flow:
1. **The Bouncer (Safety Guardrail):** Before the LLM even sees the prompt, a strict `check_refusal()` function scans the user's input. If the user asks for investment advice (e.g., "Should I buy TCS stock?"), the system immediately halts and outputs a refusal. This costs zero tokens and guarantees compliance.
2. **The Briefing:** The user's question is sent to the LLM (Groq's `llama-3.1-8b-instant`) alongside a strict `SYSTEM_PROMPT` containing the "instruction manual" for three specific tools.
3. **The Agent Loop (The "Thought" Process):** A `while steps_used < 8` loop begins.
   - **Reasoning:** The LLM analyzes the question and determines if it needs a tool.
   - **Tool Invocation:** It outputs a JSON command to execute a tool (e.g., "Use `query_data` for FY24 margin").
   - **Execution & Feedback:** The Python code runs the corresponding script, gets the data, and appends it to the chat history.
   - **Multi-Tool Synthesis:** The loop repeats. The LLM might say, "Now that I have the numbers from the CSV, let me use `search_docs` to find the CEO's explanation in the PDF."
   - **Final Delivery:** Once the LLM is satisfied with the gathered context, it generates the final human-readable answer with formatted citations, and the loop safely breaks.
4. **Infinite Loop Protection:** A hard cap is enforced. If the agent hits 8 tool calls without finding the answer, it is forced to abort the loop and summarize whatever data it managed to collect.

---

## 3. The Tool Arsenal

The agent makes intelligent routing decisions across three distinct modalities:

| Tool Name | Technology | Purpose | Data Allowed / Scope |
| :--- | :--- | :--- | :--- |
| **`search_docs`** | FAISS + `all-mpnet-base` embeddings | **Unstructured Data / Qualitative:** Extracts context from ~1,800 chunks of parsed FY24 Annual Report PDFs. | CEO commentary, corporate strategy, MD&A, risk factors, ESG goals. (Do *not* use for exact numbers). |
| **`query_data`** | Pandas DataFrame | **Structured Data / Quantitative:** Executes keyword matching on an internal `financials.csv` database. | Exact historical numbers (FY21 - FY24) like revenue, operating margin, EPS, headcount. |
| **`web_search`** | Tavily Search API | **Live Data / Real-time:** Searches the public internet for current events not present in local files. | Real-time stock prices, analyst ratings, post-April 2024 news, FY25 results. |

---

## 4. What Kind of Questions Can It Answer?

Based on its evaluation suite (`run_eval.py`), the system handles four distinct categories of questions:

1. **Single-Tool Quantitative (CSV):** *"What was Infosys' operating margin in FY24?"*
2. **Single-Tool Qualitative (PDF):** *"What strategic priorities did Infosys highlight in their FY24 MD&A?"*
3. **Multi-Tool Complex Queries (CSV + PDF):** *"How did Infosys and TCS operating margins compare in FY24, and what drove each result?"* (The agent pulls the margins from the CSV, then searches the PDFs for the driving factors, combining both into one cited answer).
4. **Edge Cases & Out-of-Bounds:** *"What was Infosys revenue in FY20?"* (The agent accurately realizes its CSV only covers FY21-FY24 and politely replies that the data is not available, completely avoiding hallucinations).

---

## 5. Performance & Evaluation

The system is rigorously benchmarked via an automated grading script (`eval/run_eval.py`) containing 20 test questions. 
- **Citations:** The system is strictly prompted to append source citations (e.g., `[Source: financials.csv]` or `[Source: Infosys_AR_FY24.pdf, Page 12]`) to every claim.
- **Traceability:** Every single execution writes a detailed log file to the `traces/` repository, exposing the exact chain of thought, the tool arguments used, and the latency.
- **Known Limitations (Honesty in Design):** The `DESIGN.md` explicitly recognizes that substring matching in CSVs can be brittle (e.g., asking for "profit margin" instead of "operating margin" requires exact keyword hits), and PDF chunking heuristics occasionally mislabel dense financial tables.

---

## 6. Project Structure

```text
├── src/
│   ├── agent.py          # Core agent loop (Groq LLM, tool dispatch, infinite loop caps)
│   ├── refusal.py        # Investment advice pre-token refusal guard
│   └── tools/
│       ├── search_docs.py  # FAISS vector search logic
│       ├── query_data.py   # Pandas CSV query logic
│       └── web_search.py   # Tavily API web search logic
├── scripts/
│   ├── create_csv.py     # Generates financials.csv
│   ├── index_docs.py     # Parses PDFs, chunks text, builds FAISS index
│   └── verify_tools.py   # Tool verification test script
├── eval/
│   └── run_eval.py       # 20-question evaluation benchmark
├── data/
│   ├── pdfs/             # Annual report PDFs (source material)
│   ├── faiss_index/      # FAISS vector database
│   └── financials.csv    # Structured financial data
├── traces/               # Auto-generated agent execution logs
├── DESIGN.md             # Theoretical design documentation
├── EVALUATION.md         # Empirical benchmark results
└── test_quick.py         # 5-question quick smoke test
```

---

## 7. Tech Stack

*   **LLM Engine:** Groq API (`llama-3.1-8b-instant`) — Chosen for incredibly fast inference speeds which is critical for loops running multiple tool calls.
*   **Vector Database:** FAISS (`IndexFlatL2`)
*   **Embeddings:** HuggingFace `all-mpnet-base-v2`
*   **Data Processing:** Pandas (CSV handling), pypdf (PDF parsing)
*   **Web Retrieval:** Tavily API 

---

## 8. Uniqueness & Standout Features

1. **Zero Orchestrator Bloat:** By entirely avoiding heavy frameworks (LangChain/LlamaIndex), the codebase remains incredibly lightweight, debuggable, and easy to trace. 
2. **Deterministic Guardrails:** The refusal mechanism is positioned *before* the LLM call. It guarantees absolutely no investment advice is given, bypassing the risk of prompt-injection tricks against the LLM itself.
3. **Dynamic Multi-Tool Synthesis:** The true power of the project lies in its ability to realize when it doesn't have enough data. If tool A only answers half the query, it actively decides to invoke tool B to finish the job before returning text to the user.
