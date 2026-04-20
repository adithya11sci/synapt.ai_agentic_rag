# Agentic RAG System Test Report & Feedback

I have executed manual and automated tests through the RAG Agent loop to ensure that it correctly navigates between the three tools (`query_data`, `search_docs`, `web_search`) and properly handles refusal guardrails. I also updated the internal LLM endpoint inside `.env` and `agent.py` to use `llama-3.3-70b-versatile` because Groq recently decommissioned the older `llama3-70b-8192` model that was originally hardcoded.

Here is the step-by-step documentation of the test results, explaining the user question, the tool chosen by the system, and the corresponding result obtained.

---

### Test 1: Quantitative Data Check
**User Question:** *"What was Infosys operating margin in FY24?"*
**Tool Chosen:** `query_data`
**Why the system chose it:** The System Prompt explicitly directs the LLM to use `query_data` for exact margins, revenue, EPS, or headcount comparisons. 
**Result Obtained:** The agent extracted the value `20.7` directly from the `data/financials.csv` through Pandas formatting, citing it correctly without any hallucinations. 
**Status:** **[PASS]**

### Test 2: Qualitative Statement Check
**User Question:** *"What is Project Maximus and what are its five pillars?"*
**Tool Chosen:** `search_docs`
**Why the system chose it:** The question asks for strategic narrative context (Project Maximus pillars), which triggers the text-based qualitative search rules inside the prompt.
**Result Obtained:** The system converted the query into embeddings via SentenceTransformers and queried the `data/faiss_index/`. It retrieved the specific chunk from `infosys_fy24.pdf` containing the definitions of the Project Maximus pillars and cited the exact Page and Document name.
**Status:** **[PASS]**

### Test 3: Live Real-world Info Check
**User Question:** *"What is the current stock price of Infosys?"*
**Tool Chosen:** `web_search`
**Why the system chose it:** Pricing information is constantly changing and thus does not live inside an FY24 annual report or the CSV. Following the prompt rules, any real-world live facts invoke the Web Search layer.
**Result Obtained:** A live query was dispatched to the Tavily API, which returned recent headlines and the live numerical price along with the source URL. 
**Status:** **[PASS]**

### Test 4: System Guardrails Check
**User Question:** *"Should I invest in Infosys stock right now?"*
**Tool Chosen:** `None` (Refusal Logic)
**Why the system chose it:** The user query triggered the hardcoded refusal logic loop located in `src/refusal.py` ("should i invest").
**Result Obtained:** The process halted cleanly at Step 0, returning: *"I am a financial research agent and cannot provide investment advice..."* No AI tokens or Tool calls were consumed.
**Status:** **[PASS]**

### Test 5: Multi-Tool Complex Verification
**User Question:** *"How did Infosys and TCS operating margins compare in FY24 and what drove each result?"*
**Tools Chosen:** `query_data` **followed by** `search_docs`.
**Why the system chose it:** The prompt combines a need for a numerical comparison ("operating margins compare in FY24") and qualitative context ("what drove each result?"). Groq correctly invokes Function Calling dynamically, executing the data search first and then firing supplementary questions into the PDF database before finalizing the context.
**Result Obtained:** The system outputted an aggregated analysis that compared the 20.7% margin to the 24.7% margin natively from Pandas, whilst synthesizing textual context directly from the PDF extractions. 
**Status:** **[PASS]**

---

## Feedback & Optimization

The integration with Groq is **extremely fast**, allowing the loop to run and execute tool function loops rapidly compared to standard Claude API implementations due to LPU hardware speeds. 

**Notes**:
1. **Model Upgrade**: As noticed during compilation, `llama3-70b-8192` is deprecated. I have proactively modified `.env`, `.env.example`, and the `agent.py` defaults to utilize the more capable `llama-3.3-70b-versatile` endpoint for stable execution. Everything is currently mapped properly to your Groq keys.
2. **Speed & Scalability**: The Python `while` loop cleanly wraps the context windows. Logging the traces locally to `.txt` works reliably, proving exactly what the agent "thought" and which Tool it fired. You are good to go!