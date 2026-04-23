# Design Document — Agentic RAG System

## 1. Overview

This system is a financial research agent that answers questions about Infosys, TCS, and Wipro using three data sources: annual report PDFs (unstructured), a financial metrics CSV (structured), and live web search. The agent autonomously selects which tool to call, calls multiple tools when needed, and composes a cited answer.

## 2. Agent Loop — Step by Step

The core is a plain Python `while` loop in `src/agent.py` (`run_agent()` function, under 100 lines).

```
User Question
     │
     ▼
[1] check_refusal() ── triggers? ──→ Return refusal (0 API calls)
     │ no
     ▼
[2] Build messages: system prompt + user question
     │
     ▼
┌──────────────────────────────────────────┐
│ while steps_used < 8:                    │
│                                          │
│  [3] Call Groq LLM with messages + tools │
│       │                                  │
│       ├─ No tool calls? → Return answer  │
│       │                                  │
│       └─ Has tool_calls:                 │
│           [4] For each tool_call:        │
│               steps_used += 1            │
│               │                          │
│               ├─ steps >= 8? → Hard cap  │
│               │   Return partial answer  │
│               │                          │
│               └─ Execute tool function   │
│                   Append result to msgs  │
│                   Log to trace           │
│                                          │
│  [5] Loop back to step 3                 │
└──────────────────────────────────────────┘
```

### Key Design Decisions

1. **Refusal check before any API call**: Investment advice triggers are caught by substring matching in `check_refusal()` before the LLM is ever called. This costs zero tokens and guarantees no tool is invoked.

2. **Tool choice is delegated to the LLM**: The system prompt and tool descriptions tell the LLM when to use each tool. The agent loop does not hardcode routing logic.

3. **Hard cap at 8 tool calls**: The `steps_used` counter increments per tool call (not per LLM call). When the cap is hit, we inject a "hard cap reached" message and make one final LLM call to compose a summary of whatever data was collected.

4. **Error isolation**: Each tool call is wrapped in try/except. Tool failures produce an error string that goes back to the LLM, which can then decide to try a different tool or report the issue.

## 3. Tool Schemas

### search_docs

| Field | Value |
|-------|-------|
| **Name** | `search_docs` |
| **Purpose** | Semantic search over FY24 annual report PDFs |
| **Input** | `query` (string) — natural language search query |
| **Output** | Top-3 text chunks with source filename, page number, and section |
| **When to use** | Qualitative questions: strategy, MD&A commentary, CEO statements, risk factors, ESG, large deals |
| **When NOT to use** | Specific financial numbers (use `query_data`), live/recent data (use `web_search`) |

**Implementation**: Encodes the query with `all-mpnet-base-v2`, searches a FAISS `IndexFlatL2` index of 1,803 chunks (450 words each, 50-word overlap), and returns the top-3 results with metadata.

### query_data

| Field | Value |
|-------|-------|
| **Name** | `query_data` |
| **Purpose** | Query structured financial CSV |
| **Input** | `question` (string) — financial data query |
| **Output** | Scalar value or table from `financials.csv` with source citation |
| **When to use** | Revenue, margin, profit, EPS, headcount — specific numbers, comparisons, rankings |
| **When NOT to use** | Qualitative explanations (use `search_docs`), recent news (use `web_search`) |

**Implementation**: Detects company names and years via keyword matching, identifies the metric from the question, filters/sorts the pandas DataFrame, and returns the result.

### web_search

| Field | Value |
|-------|-------|
| **Name** | `web_search` |
| **Purpose** | Live web search for recent information |
| **Input** | `query` (string) — short search query under 10 words |
| **Output** | Top-3 results with URL, date, and snippet |
| **When to use** | Current stock prices, post-April 2024 news, analyst ratings, FY25 results |
| **When NOT to use** | Historical data in the reports (FY21-FY24), annual report content |

**Implementation**: Calls the Tavily API with a 10-second timeout via threading. Returns formatted snippets or a clear error message on failure.

## 4. Infinite Loop Prevention

Three mechanisms prevent runaway loops:

1. **Hard cap counter**: `steps_used` is incremented for every tool call. At `steps_used >= 8`, no more tools are executed and a structured response is forced.

2. **Tool error handling**: If a tool raises an exception, the error is returned as a string to the LLM. The LLM can then choose to answer with available data or report the failure — rather than retrying indefinitely.

3. **LLM stop signal**: When the Groq API returns a response with no `tool_calls`, the loop exits immediately with the final answer.

## 5. Known Failure Modes

1. **Section detection is imperfect**: The chunking heuristic detects uppercase lines as section headers, but real PDFs have inconsistent formatting. Some chunks may have incorrect section labels.

2. **query_data keyword matching is brittle**: The tool uses substring matching to detect metrics. A question like "What was the profit margin?" won't match "operating margin" unless both keywords appear.

3. **Small model limitations**: `llama-3.1-8b-instant` occasionally fails to cite sources properly or may call the wrong tool for ambiguous questions. Larger models (70B) improve accuracy but increase latency.
