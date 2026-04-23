# Agentic RAG — Financial Research Agent

An LLM agent that answers questions about Infosys, TCS, and Wipro using three tools: document search (FAISS + annual report PDFs), structured data query (CSV via pandas), and live web search (Tavily API). The agent picks the right tool for each question, combines tools when needed, cites sources, and refuses investment advice.

Built as a plain Python while loop — no LangChain, no agent frameworks.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your real API keys:
#   GROQ_API_KEY=your_key
#   TAVILY_API_KEY=your_key

# 3. Build the data
python scripts/create_csv.py        # Creates data/financials.csv (12 rows)
python scripts/index_docs.py        # Indexes PDFs into FAISS (~10 min, 1800+ chunks)

# 4. Verify tools work independently
python scripts/verify_tools.py      # Should show 4/4 passed

# 5. Run the agent
python -X utf8 test_quick.py        # 5 representative questions
python eval/run_eval.py             # Full 20-question evaluation
```

## How It Works

1. **User asks a question** (e.g., "How did Infosys and TCS margins compare in FY24?")
2. **Refusal check**: If the question asks for investment advice, return a refusal immediately (no API call)
3. **Agent loop**: The LLM decides which tool to call based on the question type:
   - `search_docs` — qualitative info from annual report PDFs (strategy, MD&A, CEO commentary)
   - `query_data` — exact numbers from the financial CSV (revenue, margin, EPS, headcount)
   - `web_search` — live data (stock prices, recent news, FY25 results)
4. **Multi-tool**: For complex questions, the LLM calls multiple tools and composes a combined answer
5. **Hard cap**: Maximum 8 tool calls per question, enforced in code
6. **Trace logging**: Every run saves a trace file to `traces/`

## Project Structure

```
├── src/
│   ├── agent.py          # Core agent loop (Groq LLM, tool dispatch, trace logging)
│   ├── refusal.py        # Investment advice refusal guard
│   └── tools/
│       ├── search_docs.py  # FAISS vector search over PDF chunks
│       ├── query_data.py   # Pandas CSV query with keyword matching
│       └── web_search.py   # Tavily API with 10s timeout
├── scripts/
│   ├── create_csv.py     # Generates financials.csv with hardcoded FY21-FY24 data
│   ├── index_docs.py     # Parses PDFs, chunks text, builds FAISS index
│   └── verify_tools.py   # 4 independent tool tests
├── eval/
│   └── run_eval.py       # 20-question evaluation sweep
├── data/
│   ├── pdfs/             # Annual report PDFs (place here before indexing)
│   ├── faiss_index/      # Generated FAISS index + metadata
│   └── financials.csv    # Generated financial data
├── traces/               # Agent trace logs (auto-generated)
├── DESIGN.md             # Agent loop design document
├── EVALUATION.md         # Evaluation results and failure analysis
└── test_quick.py         # 5-question smoke test
```

## Tech Stack

- **LLM**: Groq API with `llama-3.1-8b-instant` (fast inference, function calling)
- **Vector store**: FAISS (IndexFlatL2) with `all-mpnet-base-v2` embeddings
- **PDF parsing**: pypdf
- **Structured data**: pandas with CSV
- **Web search**: Tavily API
- **No frameworks**: Agent loop is a plain Python while loop

## Known Failure Modes

1. **Section detection is imperfect**: PDF chunking uses an uppercase-line heuristic for section headers. Real annual reports have inconsistent formatting, so some chunks carry incorrect section labels. This affects citation quality but not retrieval relevance.

2. **query_data keyword matching is brittle**: The tool detects metrics by substring matching ("operating margin", "revenue", etc.). Unusual phrasings like "profit margin" or "top-line growth" won't match the expected keywords, causing the tool to return "no structured data found" even when the data exists.

3. **Small model may pick the wrong tool**: `llama-3.1-8b-instant` occasionally routes a quantitative question to `search_docs` instead of `query_data`, or calls `web_search` unnecessarily. Upgrading to a 70B model improves tool selection accuracy but increases latency.

## AI Disclosure

GitHub Copilot and Claude were used during development for code generation and debugging. All design decisions and the agent loop logic were reviewed and understood by the author.