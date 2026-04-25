# Evaluation Report — Agentic RAG System

## Evaluation Setup

- **LLM**: Groq API, `llama-3.1-8b-instant`
- **Vector store**: FAISS with 1,803 chunks from 3 annual report PDFs
- **Structured data**: financials.csv (12 rows, 7 columns, FY21-FY24)
- **20 questions** across 4 categories

## Results Summary

Run the full evaluation with:
```bash
python eval/run_eval.py
```

The evaluator computes a before/after comparison across three modes:
- **baseline** (no planning, no reflection)
- **plan-only** (planning enabled, reflection disabled)
- **plan+reflection** (planning + reflection enabled)

Only the **improved** traces are written to disk under `traces/improved/q1.txt` ...

Note: This comparison is intentionally slower because it runs multiple variants per question and reflection may trigger one additional retrieval.

### Category Breakdown

| Category | Questions | Description |
|----------|:---------:|-------------|
| Single-tool: search_docs | Q1–Q6 | Qualitative questions answerable from PDF content |
| Single-tool: query_data | Q7–Q12 | Quantitative questions answerable from CSV |
| Multi-tool | Q13–Q16 | Questions requiring both CSV numbers and PDF context |
| Refusal / Edge cases | Q17–Q20 | Investment advice, out-of-range, irrelevant, overload |

### Pass/Fail Criteria

| Category | Pass Condition |
|----------|---------------|
| Q1–Q16 | Answer is non-empty AND contains a citation (Source:, Page:, URL:, .pdf, financials.csv) |
| Q17 | `refused=True` and `steps_used=0` |
| Q18 | Agent states data is not available (FY2015 is outside our FY21-FY24 range) |
| Q19 | Agent states it cannot answer (irrelevant question) |
| Q20 | `cap_hit=True` or `web_search` was called (FY25 Q1 data requires web) |

## Failure Analysis

### Failure Mode 1: search_docs Returns Irrelevant Chunks

**What happens**: For some qualitative questions, the top-3 FAISS results come from financial tables or statutory boilerplate pages rather than the actual management commentary sections that contain the answer.

**Why it happens**: The chunking strategy (450 words, 50-word overlap) doesn't distinguish between narrative text and tabular content within PDFs. Financial statements and statutory reports contain many pages of dense text that get chunked and embedded alongside the useful MD&A sections.

**Proposed fix**: Add a post-retrieval relevance filter that scores chunks based on keyword overlap with the query, or use a re-ranker model (e.g., cross-encoder) to re-score the top-K results before returning them to the agent.

### Failure Mode 2: query_data Misses Valid Questions

**What happens**: Questions like "What was the profit margin?" or "Show me the top-line numbers" return "No structured data found" even though the data exists in the CSV.

**Why it happens**: The tool uses exact substring matching (`"operating margin" in q`, `"revenue" in q`). It doesn't handle synonyms or paraphrased metric names.

**Proposed fix**: Either expand the keyword list to include common synonyms, or let the LLM translate the question into a specific metric name before calling the tool (by adding a `metric` parameter to the tool schema).

## Questions and Expected Behavior

| Q# | Question (first 40 chars) | Expected Tool(s) | Expected Behavior |
|----|--------------------------|-------------------|-------------------|
| 1 | What is Project Maximus and what are... | search_docs | Return PDF excerpt with citation |
| 2 | What did the Infosys CEO say about g... | search_docs | CEO commentary with page number |
| 3 | What was the main reason Infosys gav... | search_docs | Revenue growth explanation |
| 4 | What strategic priorities did Infosys... | search_docs | MD&A priorities with citation |
| 5 | How many new customers did Infosys a... | search_docs | Customer count from PDF |
| 6 | What is Infosys capital allocation po... | search_docs | Policy description with citation |
| 7 | What was Infosys operating margin in... | query_data | 20.7% from financials.csv |
| 8 | What was Infosys revenue growth from... | query_data | FY21-FY24 revenue trend |
| 9 | Which company had the highest operat... | query_data | TCS at 24.7% |
| 10 | What was Infosys EPS in FY24? | query_data | 63.39 from financials.csv |
| 11 | How did Infosys headcount change fro... | query_data | FY23: 343,234 → FY24: 317,240 |
| 12 | What was Wipro revenue trend over 4... | query_data | 4-year revenue table |
| 13 | How did Infosys and TCS operating ma... | query_data + search_docs | Numbers + drivers |
| 14 | What was Infosys headcount in FY24 a... | query_data + search_docs | Count + workforce strategy |
| 15 | Compare revenue growth across all 3... | query_data + search_docs | Table + Infosys strategy |
| 16 | What was Infosys net profit in FY24... | query_data + search_docs | Profit + management factors |
| 17 | Should I invest in Infosys stock rig... | None (refuse) | Refuse, 0 tools called |
| 18 | What was Infosys revenue in FY2015? | query_data | "No data" — FY2015 outside range |
| 19 | What is the airspeed velocity of an... | None | Politely decline — irrelevant |
| 20 | Tell me everything about Infosys Q1... | web_search | FY25 data requires web search |

*Note: Actual agent outputs are recorded in `traces/q1.txt` through `traces/q20.txt` after running `python eval/run_eval.py`.*
