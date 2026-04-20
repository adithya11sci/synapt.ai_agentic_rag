Build me a complete Agentic RAG project for an internship assignment.
Read every section carefully before writing a single line of code.

=======================================================================
CONTEXT
=======================================================================
I am building an LLM agent that answers questions about 3 Indian IT
companies — Infosys, TCS, and Wipro — using their FY24 annual report
PDFs, a structured CSV of 4-year financials, and live web search.

The agent must:
- Pick the right tool for each question
- Combine multiple tools when needed
- Cite every claim with exact source, page, or URL
- Refuse investment advice without calling any tool
- Hard-stop after 8 tool calls and return a structured refusal
- Log a full trace for every run

=======================================================================
STACK  (do not deviate from this)
=======================================================================
- Language: Python 3.11
- LLM: Anthropic Claude via the anthropic SDK
  · Dev model:  claude-haiku-4-5-20251001
  · Final eval: claude-sonnet-4-6
- Vector store: FAISS (faiss-cpu) + sentence-transformers
- Embeddings model: all-mpnet-base-v2
- PDF parsing: pypdf
- Structured data: pandas, CSV file
- Web search: Tavily Python SDK (from tavily import TavilyClient)
- No LangChain. No LangGraph. No agent frameworks of any kind.
  The agent loop must be a plain Python while loop I can read line by line.

=======================================================================
FOLDER STRUCTURE  (generate exactly this)
=======================================================================
agentic-rag/
├── .env.example
├── .gitignore
├── requirements.txt
├── data/
│   ├── pdfs/               ← I will place PDFs here manually
│   ├── faiss_index/        ← created by index_docs.py
│   └── financials.csv      ← created by create_csv.py
├── scripts/
│   ├── create_csv.py
│   ├── index_docs.py
│   └── verify_tools.py
├── src/
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_docs.py
│   │   ├── query_data.py
│   │   └── web_search.py
│   ├── agent.py
│   └── refusal.py
├── traces/
└── eval/
    └── run_eval.py

=======================================================================
FILE 1 — .env.example
=======================================================================
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
CLAUDE_MODEL=claude-haiku-4-5-20251001

=======================================================================
FILE 2 — .gitignore
=======================================================================
Must include: .env, __pycache__, *.pyc, data/faiss_index/,
traces/, *.egg-info, .DS_Store, venv/

=======================================================================
FILE 3 — requirements.txt
=======================================================================
anthropic
tavily-python
faiss-cpu
sentence-transformers
pypdf
pandas
python-dotenv
numpy

=======================================================================
FILE 4 — scripts/create_csv.py
=======================================================================
Creates data/financials.csv with these EXACT hardcoded values.
Do not invent or round any number.

Columns: company, year, revenue_crore, operating_margin_pct,
         net_profit_crore, eps_inr, headcount

INFOSYS rows:
  FY21: 100472, 24.5, 19351, 46.71, 259619
  FY22: 121641, 23.0, 22110, 52.52, 314015
  FY23: 146767, 21.0, 24095, 57.63, 343234
  FY24: 153670, 20.7, 26233, 63.39, 317240

TCS rows:
  FY21: 164177, 26.6, 33388,  90.19, 488649
  FY22: 191754, 25.0, 38327, 103.60, 556986
  FY23: 225458, 24.1, 42147, 114.67, 614795
  FY24: 240893, 24.7, 46099, 129.00, 601546

WIPRO rows:
  FY21: 61943, 19.2, 10796, 1.96, 197712
  FY22: 79312, 17.8, 12229, 2.22, 236023
  FY23: 90488, 16.3, 11496, 2.09, 258096
  FY24: 89763, 16.8, 11149, 2.13, 234054

=======================================================================
FILE 5 — src/tools/search_docs.py
=======================================================================
Build two functions:

FUNCTION 1: chunk_and_index(pdf_paths, index_path)
- Parse each PDF with pypdf
- Chunk: 450 tokens per chunk, 50 token overlap, measured in words
- Always prepend the nearest section heading to each chunk text
- Store metadata per chunk: source (filename), page (int), section (str)
- Embed all chunks with SentenceTransformer("all-mpnet-base-v2")
- Build a FAISS IndexFlatL2 index
- Save: index_path/index.faiss and index_path/metadata.json

FUNCTION 2: search_docs(query, top_k=3) → str
- Load index.faiss and metadata.json from data/faiss_index/
- Embed query and retrieve top_k chunks
- Return this exact format:
    [1] Source: infosys_fy24.pdf | Page: 71 | Section: MD&A
        <first 300 chars of chunk text>
    [2] ...
- If index files not found, return a clear error string. Never crash.

TOOL DESCRIPTION for the LLM (use this exact wording in agent.py):
"Use this tool to search the Infosys, TCS, and Wipro FY24 annual
report PDFs for qualitative information. Use it when the question
asks for explanations, management commentary, strategic priorities,
reasons behind financial results, risk factors, ESG initiatives,
CEO or CFO statements, large deal narratives, or any narrative text
from the reports. Do NOT use for specific financial numbers or for
current market data — use query_data or web_search for those.
Input: a natural language query string."

=======================================================================
FILE 6 — src/tools/query_data.py
=======================================================================
Build one function: query_data(question) → str

- Load data/financials.csv into pandas on every call
- Detect intent with keyword matching on the lowercased question:
    "operating margin"  → filter company + year, return margin value
    "revenue"           → filter or return multi-year trend table
    "net profit"        → filter and return
    "eps"               → filter and return
    "headcount"         → filter and return
    "compare" or "all"  → return full multi-row table
    "highest" / "best"  → sort and return top row with company name
    "lowest" / "worst"  → sort and return bottom row
- Detect company name: look for "infosys", "tcs", "wipro" in question
- Detect year: look for FY21, FY22, FY23, FY24 (or 2021..2024)
- Always append: "Source: financials.csv"
- If nothing matches, return:
  "No structured data found for: <question>.
   Try rephrasing or use search_docs for qualitative content."

TOOL DESCRIPTION for the LLM:
"Use this tool to query the structured financial CSV containing 4
years of key metrics (FY21 to FY24) for Infosys, TCS, and Wipro:
revenue in crore rupees, operating margin percent, net profit in
crore rupees, EPS in rupees, and headcount. Use it when the question
asks for a specific number, a year-over-year comparison, a ranking,
or any calculation on financial metrics. Do NOT use for qualitative
explanations or recent news — use search_docs or web_search instead."

=======================================================================
FILE 7 — src/tools/web_search.py
=======================================================================
Build one function: web_search(query) → str

- Import: from tavily import TavilyClient
- Read api_key from os.environ["TAVILY_API_KEY"]
- Call: client.search(query=query, search_depth="basic",
         max_results=3, include_answer=False,
         include_raw_content=False)
- Return this exact format:
    [1] URL: https://...
        Date: 2025-04-10
        Snippet: <content[:300]>
    [2] ...
- Wrap the entire call in try/except. On any error return:
  "Web search failed: <error message>. Use search_docs or query_data."
- Set a 10 second timeout using a threading timer. On timeout return:
  "Web search timed out. Use search_docs or query_data instead."

TOOL DESCRIPTION for the LLM:
"Use this tool ONLY for information that cannot exist in a FY24
annual report: current stock prices, analyst ratings from the past
few weeks, news published after April 2024, recent executive
appointments, or FY25 quarterly results. Do NOT use for historical
financial data (FY21 to FY24) or for content from the annual reports.
Input: a short specific search query under 10 words."

=======================================================================
FILE 8 — src/refusal.py
=======================================================================
REFUSAL_TRIGGERS list must include:
"should i invest", "buy or sell", "which stock should",
"recommend a stock", "best investment", "should i buy",
"will the stock go up", "price prediction", "stock tips",
"should i sell", "is it a good time to buy"

Function: check_refusal(question) → str | None
- Lowercase and strip the question
- If any trigger is found, return this message:
  "I am a financial research agent and cannot provide investment
   advice. I can share financial data and management commentary from
   the FY24 annual reports, but buy or sell decisions require a
   qualified financial advisor. No tools were called."
- Otherwise return None

=======================================================================
FILE 9 — src/agent.py  (THE CORE — read this section twice)
=======================================================================
Build the agent loop as a plain Python while loop. Under 100 lines
for the run_agent function. No framework. No initialize_agent.

SYSTEM PROMPT to use (exact wording):
"You are a financial research agent that answers questions about
Infosys, TCS, and Wipro using three tools: search_docs, query_data,
and web_search.

TOOL SELECTION RULES:
- search_docs  → qualitative: reasons, strategy, MD&A, commentary,
                  CEO statements, ESG, risk factors, narratives.
                  Do NOT use for numbers or live data.
- query_data   → quantitative: revenue, margin, profit, EPS,
                  headcount, comparisons, rankings.
                  Do NOT use for explanations or recent news.
- web_search   → live only: current price, recent news post-April
                  2024, analyst ratings, FY25 results.
                  Do NOT use for historical data in the reports.

ANSWER RULES:
1. Cite every claim: document name + page, CSV row, or URL + date.
2. Never guess. If no tool returned relevant data, say so.
3. For questions needing a number AND explanation: call query_data
   first, then search_docs.
4. Hard cap: 8 tool calls maximum per question.
5. For trivial questions (math, basic definitions): answer directly.

OUTPUT FORMAT:
Answer: [your answer with inline citations]
Citations: [tool used → source → page or row or url]
Steps used: [N] / 8"

AGENT LOOP LOGIC:
1. Call check_refusal(question). If it returns a string, return
   immediately with steps_used=0 and refused=True.
2. Build messages list with the user question.
3. While steps_used < MAX_STEPS (8):
   a. Call claude with the system prompt, tools list, and messages.
   b. If stop_reason == "end_turn": extract text, return answer.
   c. Extract tool_use blocks. If none, break.
   d. Append assistant response to messages.
   e. For each tool_use block:
      - Increment steps_used
      - If steps_used >= MAX_STEPS: add a hard cap message as the
        tool result, make one final LLM call to compose a refusal,
        return with cap_hit=True.
      - Call the matching tool function.
      - Append result to tool_results list.
   f. Append tool_results to messages as a user turn.
4. Return function must return a dict with keys:
   answer (str), steps_used (int), trace (list), refused (bool),
   cap_hit (bool, optional)

TRACE: append a dict for every tool call with keys:
  step, tool, input, result (first 500 chars)

Also build write_trace(question, result, filepath) that writes a
human-readable text file with all steps, inputs, results, and
a final summary line showing steps used out of 8.

Also build a __main__ block that:
- Reads the question from sys.argv
- Calls run_agent()
- Saves trace to traces/
- Prints the answer and steps used

=======================================================================
FILE 10 — scripts/index_docs.py
=======================================================================
- Loop over every .pdf file in data/pdfs/
- Call chunk_and_index() from src/tools/search_docs.py
- Print per-file progress: "Indexing tcs_fy24.pdf... 912 chunks"
- Print total time and total chunk count when done

=======================================================================
FILE 11 — scripts/verify_tools.py
=======================================================================
Run these 4 tests and print PASS or FAIL for each:

Test 1 — search_docs:
  Call search_docs("Project Maximus five pillars margin expansion")
  PASS if result contains "[1]" and "Page:"

Test 2 — query_data:
  Call query_data("What was Infosys operating margin in FY24?")
  PASS if "20.7" is in the result string

Test 3 — web_search:
  Call web_search("Infosys stock price today")
  PASS if "URL:" is in the result string

Test 4 — refusal:
  Call check_refusal("Should I buy Infosys stock?")
  PASS if return value is not None

Print final: "N/4 tests passed."
If any fail, print which test failed and why.

=======================================================================
FILE 12 — eval/run_eval.py  (20 QUESTIONS — use exactly as written)
=======================================================================
Run all 20 questions through run_agent(). For each, save the trace
to traces/. Print a results table at the end.

SINGLE-TOOL search_docs (6 questions):
Q1:  "What is Project Maximus and what are its five pillars?"
Q2:  "What did the Infosys CEO say about generative AI in FY24?"
Q3:  "What was the main reason Infosys gave for revenue growth FY24?"
Q4:  "What strategic priorities did Infosys highlight in FY24 MD&A?"
Q5:  "How many new customers did Infosys add in FY24?"
Q6:  "What is Infosys capital allocation policy?"

SINGLE-TOOL query_data (6 questions):
Q7:  "What was Infosys operating margin in FY24?"
Q8:  "What was Infosys revenue growth from FY21 to FY24?"
Q9:  "Which company had the highest operating margin in FY24?"
Q10: "What was Infosys EPS in FY24?"
Q11: "How did Infosys headcount change from FY23 to FY24?"
Q12: "What was Wipro revenue trend over 4 years?"

MULTI-TOOL (4 questions):
Q13: "How did Infosys and TCS operating margins compare in FY24
      and what drove each result?"
Q14: "What was Infosys headcount in FY24 and what does the company
      say about its workforce strategy?"
Q15: "Compare revenue growth across all 3 companies over 4 years
      and explain Infosys strategy for growth."
Q16: "What was Infosys net profit in FY24 and what factors did
      management cite for it?"

REFUSAL questions (4 questions):
Q17: "Should I invest in Infosys stock right now?"
Q18: "What was Infosys revenue in FY2015?"
Q19: "What is the airspeed velocity of an unladen swallow?"
Q20: "Tell me everything about Infosys Q1 FY25 results in
      complete detail including all segment breakdowns."

For the results table print these columns per question:
  Q number | first 40 chars of question | tools called | steps | pass/fail

Mark PASS if:
- Refusal questions: refused=True and steps_used=0 (Q17) or
  agent said data not available (Q18, Q19)
- Q20: cap_hit=True or web_search was called
- All others: answer is not empty and contains a citation

=======================================================================
QUALITY RULES  (apply to every file you generate)
=======================================================================
1. Every file must run without modification after:
      cp .env.example .env   (user adds real keys)
      pip install -r requirements.txt
      python scripts/create_csv.py
      python scripts/index_docs.py
      python scripts/verify_tools.py

2. Load environment variables with python-dotenv at the top of
   any file that reads os.environ.

3. No API key hardcoded anywhere. All keys from .env only.

4. All file paths relative to the project root using pathlib.

5. Every function has a one-line docstring.

6. Wrap every external API call (Anthropic, Tavily) in try/except.
   Return a clear error string. Never let an exception crash the agent.

7. The agent loop function must be under 100 lines. If it is longer,
   refactor — complexity belongs in the tool functions, not the loop.

8. Generate all files in one response. Do not ask clarifying questions.
   Make reasonable decisions and note them briefly at the end.

=======================================================================
START
=======================================================================
Generate all 12 files now, in order, with full working code.
After the last file, add a section called KNOWN LIMITATIONS that
lists 3 honest failure modes you expect this agent to have.