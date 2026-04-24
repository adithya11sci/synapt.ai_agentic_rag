# 🏗️ Design Document — Agentic RAG System

## 1. 🌟 Overview

This system is a specialized financial research agent designed to answer complex queries about Infosys, TCS, and Wipro. It dynamically synthesizes information across three distinct data modalities:
* **Unstructured Data:** Annual report PDFs (via Vector DB)
* **Structured Data:** Financial metrics CSV (via Pandas logic)
* **Real-time Data:** Live web search (via Tavily API)

The agent autonomously selects which tool to invoke, calls multiple tools sequentially when required, and composes a highly accurate, cited final answer.

## 2. 🔄 Agent Loop — Step by Step

The core architecture avoids heavy orchestration frameworks in favor of a highly deterministic, fully transparent Python `while` loop housed in `src/agent.py` (`run_agent()` function).

```text
User Question
     │
     ▼
[1] check_refusal() ── triggers? ──→ Return refusal (0 API calls/tokens)
     │ no
     ▼
[2] Build messages: Strict System Prompt + User Question
     │
     ▼
┌────────────────────────────────────────────────────────┐
│ while steps_used < 8:                                  │
│                                                        │
│  [3] Call Groq LLM with chat history + tool schemas    │
│       │                                                │
│       ├─ No tool calls? → Return final text answer     │
│       │                                                │
│       └─ Has tool_calls:                               │
│           [4] For each tool_call:                      │
│               steps_used += 1                          │
│               │                                        │
│               ├─ steps >= 8? → Abort: Hit Hard Cap     │
│               │   Return partial summarized answer     │
│               │                                        │
│               └─ Execute Python Tool Function          │
│                   Append output to message history     │
│                   Log telemetry to trace files         │
│                                                        │
│  [5] Loop back to step 3 (LLM reviews new tool data)   │
└────────────────────────────────────────────────────────┘
```

### 🔑 Key Design Decisions

1. **Pre-LLM Refusal Check**: Investment advice triggers are caught explicitly by substring matching in `check_refusal()` *before* the LLM is ever called. This guarantees absolute compliance and zero token usage for rejected queries.
2. **Autonomous LLM Routing**: The system prompt provides explicit instructions on tool boundaries. The agent loop does not hardcode `if/else` routing logic; the LLM handles it dynamically.
3. **Strict Iteration Caps**: The `steps_used` counter forcefully terminates infinite loops at 8 executions, saving exponential API cost explosions.
4. **Resilient Error Isolation**: Each tool call is strictly wrapped in `try/except` blocks. If a tool crashes (e.g., Network Timeout), the exception is converted to a string and fed *back* to the LLM, allowing the AI to naturally apologize or pick a fallback tool.

## 3. 🛠️ Tool Schemas

### 📄 search_docs

| Field | Description / Value |
|-------|-------|
| **Name** | `search_docs` |
| **Purpose** | Semantic search over FY24 annual report PDFs |
| **Input** | `query` (string) — natural language search query |
| **Output** | Top-3 text chunks with source filename, page number, and section |
| **When to use** | Qualitative questions: corporate strategy, MD&A commentary, CEO statements, risk factors, ESG goals |
| **When NOT to use**| Specific financial numbers (use `query_data`), live/recent data (use `web_search`) |

* **Implementation Details**: Uses `SentenceTransformers` (`all-mpnet-base-v2`) to encode queries, searching a persistent local `FAISS` index containing 1,803 pre-embedded text chunks (chunked at 450 words with a 50-word sliding overlap).

### 📊 query_data

| Field | Description / Value |
|-------|-------|
| **Name** | `query_data` |
| **Purpose** | Query structured, tabular financial CSV database |
| **Input** | `question` (string) — financial data query |
| **Output** | Exact scalar values or Markdown tables from `financials.csv` with absolute source citations |
| **When to use** | Revenue, margin, net profit, EPS, headcount metrics, year-over-year comparisons |
| **When NOT to use**| Qualitative explanations (use `search_docs`), real-time stock news (use `web_search`) |

* **Implementation Details**: Employs deterministic exact keyword matching to filter/sort a Pandas DataFrame. Extracts company names, years, and specific metric combinations safely.

### 🌐 web_search

| Field | Description / Value |
|-------|-------|
| **Name** | `web_search` |
| **Purpose** | Live web search for recent, post-dataset information |
| **Input** | `query` (string) — concise SEO search query (< 10 words) |
| **Output** | Top-3 live web results with URL, date, and relevant text snippet |
| **When to use** | Current stock prices, post-April 2024 news, analyst ratings, FY25 current quarter results |
| **When NOT to use**| Historical FY21-FY24 data available natively in the CSV or PDFs |

* **Implementation Details**: Calls the Trivily Search API with strict 10-second timeout threads to prevent locking the main agent loop.

## 4. 🛑 Infinite Loop Prevention

Three concurrent mechanisms strictly prevent runaway loops:

1. **Hard Execution Cap**: `steps_used` evaluates on *every* tool call. At `>= 8`, execution breaks and a system prompt commands the LLM: *"Hard cap reached. Summarize what you currently have."*
2. **Tool Feedback Loops**: If a tool raises a Python exception, it doesn't crash the script. The `Exception` returns as a payload to the LLM, signaling it shouldn't try that exact path again.
3. **Natural Termination Signal**: The Groq API signals the loop exit automatically when it returns standard text content with an empty `tool_calls` array.

## 5. ⚠️ Known Failure Modes (Design Limitations)

1. **Heuristic Section Detection**: The PDF chunker tags pages using a simplistic UPPERCASE regex rule to detect headers. Because Annual Reports have chaotic formatting, some text blocks receive incorrect section labels (affecting citation precision, though not search recall).
2. **Brittle Keyword Mapping**: The `query_data` tool searches using literal substring matching. Valid human questions (e.g., *"What was the bottom-line profit?"*) miss the expected column name (`"net profit"`), triggering a false-negative "No structured data found" response.
3. **Small-Model Routing Quirks**: Running on `llama-3.1-8b-instant` favors latency over extreme comprehension. The agent might occasionally mismatch ambiguous financial queries to `search_docs` rather than `query_data`. Scaling the requested model to a 70B variant inside `.env` immediately resolves this at the cost of slower response times.
