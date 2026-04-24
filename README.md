# 🚀 Agentic RAG — Financial Research Agent

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange.svg)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-green.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-blueviolet.svg)

An autonomous LLM-powered financial research agent capable of answering complex, multi-layered questions about **Infosys, TCS, and Wipro**. 

Unlike traditional Chatbots that rely on pre-trained knowledge or basic RAG, this system acts as a true **Agent**. It dynamically routes queries, uses three distinct data tools (Unstructured PDFs, Structured CSVs, and Live Web Search), and synthesizes information across multiple sources to provide precise, cited answers. 

**Built completely from scratch using a raw Python `while` loop — Zero LangChain, Zero LlamaIndex, Zero Agent Framework bloat.**

## ✨ Core Features

- **🧠 Autonomous Tool Routing:** The LLM independently chooses the correct tool for the job.
- **🔗 Multi-Tool Synthesis:** Handles complex requests by chaining tools (e.g., fetching a hard number from the CSV, then searching the PDFs for the CEO's explanation).
- **🛡️ Strict Safety Guardrails:** Pre-token interceptors block any attempts to elicit investment advice.
- **♾️ Infinite Loop Protection:** A hard cap of 8 tool calls prevents runaway API usage and forces summarization.
- **🔍 Full Traceability:** Every tool invocation, latency metric, and LLM reasoning step is logged locally for developer auditing.

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your real API keys:
#   GROQ_API_KEY=your_key
#   TAVILY_API_KEY=your_key

# 3. Build the data sources
python scripts/create_csv.py        # Generates data/financials.csv (FY21-FY24)
python scripts/index_docs.py        # Embeds PDFs into FAISS (~10 min, ~1800 chunks)

# 4. Verify components are working
python scripts/verify_tools.py      # Runs discrete tool checks; should show 4/4 passing

# 5. Run the Agent!
python -X utf8 test_quick.py        # Quick 5-question smoke test
python eval/run_eval.py             # Full 20-question rigorous benchmark
```

## ⚙️ How The Agent Loop Works

1. **User Prompt:** The system receives a natural language query (e.g., *"How did Infosys and TCS margins compare in FY24?"*)
2. **Refusal Check:** If the question is asking for investment advice, the loop aborts and returns a refusal immediately (Zero API calls used).
3. **The Agent Loop:** The LLM decides which tool to call based on its strict instruction manual:
   - 📄 `search_docs`: FAISS vector search inside Annual Report PDFs for qualitative context (strategy, MD&A, CEO quotes).
   - 📊 `query_data`: Pandas lookups inside the financial CSV for exact quantitative numbers (revenue, margin, EPS).
   - 🌐 `web_search`: Tavily API calls for live, real-time data (current stock prices, news post-April 2024).
4. **Execution & Feedback:** The Python code runs the tool, appends the data to the chat history, and the LLM re-evaluates.
5. **Synthesis:** Once the LLM is satisfied with the gathered context, it generates a final answer with explicit citations.

## 📂 Project Structure

```text
├── src/
│   ├── agent.py          # Core agent loop (Groq LLM, tool dispatch, infinite loop caps)
│   ├── refusal.py        # Investment advice pre-token refusal guard
│   └── tools/
│       ├── search_docs.py  # FAISS vector search implementation 
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
│   └── financials.csv    # Sandbox tabular memory
├── traces/               # Auto-generated agent execution logs
├── DESIGN.md             # Theoretical architecture design
├── EVALUATION.md         # Empirical benchmark and failure analysis
└── test_quick.py         # Developer smoke test script
```

## 🛠️ Tech Stack

- **Large Language Model**: Groq API via `llama-3.1-8b-instant` (Selected for ultra-low latency function calling)
- **Vector Database**: FAISS (Facebook AI Similarity Search) using `IndexFlatL2`
- **Embeddings**: HuggingFace `all-mpnet-base-v2` via `SentenceTransformers`
- **Data Engineering**: `pypdf`, `pandas`, `numpy`
- **Web Retrieval**: `Tavily` Search API
- **Agent Orchestrator**: 100% Vanilla Python `while` loop

## 🤖 AI Disclosure

GitHub Copilot and Claude were used during development for code generation and debugging. All design decisions and the agent loop logic were reviewed and understood by the author.