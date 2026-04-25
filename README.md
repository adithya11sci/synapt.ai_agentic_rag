# 🚀 Agentic RAG — Indian IT Company Financials

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange.svg)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-green.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-blueviolet.svg)

A fully autonomous LLM reasoning agent that answers complex questions about company financials by integrating three distinct data sources natively:

1.  **Unstructured Documents** (Annual Report PDFs for Infosys, TCS, Wipro — FY24)
2.  **Structured Financials** (Pandas CSV database with metrics: revenue, operating margins, headcount, EPS)
3.  **Live Web Search** (Tavily API for real-time stock prices, recent news, and analyst ratings)

The agent implements a complete reasoning loop from scratch: **query rewrite / gate check → planning → tool selection → execution → sufficiency check → answer composition** (max 8 steps limit).

**Built completely from scratch using a raw Python `while` loop — Zero LangChain, Zero LlamaIndex, Zero Agent Framework bloat.**

## ✨ Core Features

- **🧠 Autonomous Tool Routing:** The LLM independently chooses the correct tool based on context.
- **🔗 Multi-Tool Synthesis:** Handles complex requests by chaining tools (e.g., fetching a hard number from the CSV, then searching the PDFs for the CEO's explanation).
- **🛡️ Strict Safety Guardrails:** Pre-token interceptors politely block out-of-domain queries (general trivia) and attempts to elicit investment advice.
- **♾️ Infinite Loop Protection:** A hard cap of 8 tool calls prevents runaway API usage and forces summarization.
- **🔍 Full Traceability (Bonus)**: Every tool invocation, latency metric (in seconds), and LLM reasoning step is logged locally to `traces/` for developer auditing.

## ✅ Implemented: Planning + Reflection (Bonuses)

- **Bonus A — Planning step**: The agent emits a short, single-line `Plan:` *before* any tool call. This plan is shown in the terminal output and written into trace files.
- **Bonus C — Reflection step**: After drafting an answer, the agent emits `Reflection: OK` (or `Reflection: Needs retrieval`). If needed, it performs **one additional retrieval** and revises the answer.

### ⏱️ Why runs can take longer now

- Enabling planning/reflection adds extra LLM calls.
- `search_docs` is slow on the *first* call because it loads the embedding model + FAISS index.
- The evaluation script may run multiple variants per question (baseline vs plan-only vs plan+reflection), which multiplies runtime.

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/adithya11sci/synapt.ai_agentic_rag.git
cd synapt.ai_agentic_rag
python -m venv .venv
# Activate the virtual environment:
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```
Fill in your `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
TAVILY_API_KEY=your_tavily_api_key_here
```
*Get Free API Keys: [Groq](https://console.groq.com/) (free tier: 30 RPM) | [Tavily](https://tavily.com/) (free tier: 1000 calls/month)*

### 3. Build Data Indexes (First Time Only)
```bash
# Generate the structured financial CSV data (FY21-FY24)
python scripts/create_csv.py

# Embed PDF annual reports and build the FAISS vector index (~10 mins)
python scripts/index_docs.py
```
*This creates `data/financials.csv` and the `data/faiss_index/` vector database.*

## 🚀 Usage

### Verify Components
Run the internal health checks to ensure all your tools and APIs are authenticating correctly:
```bash
python scripts/verify_tools.py
```

### Run the Agent (Smoke Tests)
The agent automatically selects tools based on the question constraints:
```bash
python test_quick.py
```
This runs 6 diverse questions (including numeric comparisons, qualitative lookups, live web searches, and out-of-domain refusals). Traces are automatically generated and saved to the `traces/` folder.

### Run the Full Benchmark
To evaluate the system against the 20-question assignment rubric:
```bash
python eval/run_eval.py
```

## ⚙️ Architecture & Data Flow

### The Reasoning Loop
```text
User Question
    ↓
① Gate Check (trivial / out-of-domain / refuse / proceed)
    ↓
② Planning (decide tools & order based on system prompt)
    ↓
③ Tool Selection & Execution (max 8 iterations)
    ├─ search_docs (unstructured textual PDFs)
    ├─ query_data (structured numeric CSV)
    └─ web_search (live, real-time web data)
    ↓
④ Sufficiency Check (have enough context?)
    ↓
⑤ Answer Composition (with precise citations)
    ↓
Final Answer + Execution Trace (.txt output)
```

### Available Data Source Mapping

| Source | Tool | Companies | Data Type | Purpose |
|--------|------|-----------|-----------|---------|
| **PDF Reports** | `search_docs` | Infosys, TCS, Wipro | Textual / MD&A | Strategy, CEO commentary, reasons, ESG, risks |
| **Pandas CSV** | `query_data` | Infosys, TCS, Wipro | Numeric (FY21-FY24) | Revenue, margin, headcount, EPS, comparisons |
| **Web API** | `web_search` | Any Company | Real-time Live | Stock prices, current news, analyst ratings |

### Routing Rules (Agent Logic)

The agent strictly follows these rules mapped in the `SYSTEM_PROMPT`:

| Question Type | Tool Selected | Reason |
|---------------|---------------|--------|
| *"What was Infosys FY24 revenue?"* | `query_data` | Numeric historical data residing in CSV |
| *"Why did TCS margins improve?"* | `search_docs` | Qualitative explanation from Annual Reports |
| *"Compare financials, then explain why"* | `query_data` + `search_docs` | Requires numbers first, then textual context |
| *"Current stock price"* | `web_search` | Real-time live data, not historical |
| *"Should I buy Infosys?"* | **Refuse** | Pre-token trigger blocks investment advice |
| *"How to make pizza?"* | **Refuse** | Out-of-domain general trivia block |

## 📂 Project Structure

```text
synapt.ai_agentic_rag/
├── src/
│   ├── agent.py          # Core agent loop (Groq LLM, tool dispatch, 8-step max, trace logger)
│   ├── refusal.py        # Investment advice / Out-of-domain pre-token refusal guard
│   └── tools/
│       ├── search_docs.py  # FAISS vector search implementation over PDFs
│       ├── query_data.py   # Pandas CSV structured query logic
│       └── web_search.py   # Live Tavily API integration
├── scripts/
│   ├── create_csv.py     # Sandbox data generation
│   ├── index_docs.py     # PDF parsing, sentence chunking, and FAISS indexing
│   └── verify_tools.py   # CI/CD style health checks
├── eval/
│   └── run_eval.py       # 20-question comprehensive grading evaluation
├── data/
│   ├── pdfs/             # Raw Annual Report PDFs 
│   ├── faiss_index/      # Processed FAISS DB and JSON metadata
│   └── financials.csv    # Tabular numeric memory
├── traces/               # Auto-generated agent execution logs
├── DESIGN.md             # Theoretical architecture design
├── EVALUATION.md         # Empirical benchmark and failure analysis
├── all_info.md           # Master compiled architecture doc
└── test_quick.py         # Developer smoke test script
```

## 🛠️ Tech Stack & Dependencies

- **Large Language Model**: Groq API via `llama-3.1-8b-instant` (Selected for ultra-low latency function calling and rate-limit backoff handling)
- **Vector Database**: FAISS (Facebook AI Similarity Search) using `IndexFlatL2`
- **Embeddings**: HuggingFace `all-mpnet-base-v2` via `SentenceTransformers` (768-dim)
- **Data Engineering**: `pypdf`, `pandas`, `numpy`
- **Web Retrieval**: `Tavily` Search API
- **Agent Orchestrator**: 100% Vanilla Python `while` loop (No frameworks)

## 🤖 AI Disclosure

GitHub Copilot and Claude were used during development for code generation, markdown formatting, and debugging. All design decisions, architectural mapping, and the core agent loop logic were strictly reviewed and driven by the author.