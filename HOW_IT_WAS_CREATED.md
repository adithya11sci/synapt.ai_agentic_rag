# Agentic RAG Project Documentation

## How It Was Created

This RAG (Retrieval-Augmented Generation) agent was developed exactly according to the constraints and requirements outlined in the prompt assignment. Here is how the system is set up and how it functions under the hood.

### 1. The Virtual Environment & Dependencies
We created a virtual environment setup and implemented a simple `requirements.txt` containing all explicitly requested core libraries:
* `groq` (to use Groq's high-speed API serving LLMs like Llama 3)
* `tavily-python` (for real-world live web lookups)
* `faiss-cpu`, `sentence-transformers`, `numpy`, and `pypdf` for parsing, chunking, and finding textual relevance inside the PDFs.
* `pandas` for processing the rigid tabular data requirements inside the `financials.csv`.

*To install everything, one simply enters:*
```bash
pip install -r requirements.txt
```
*(Note for Windows users: installing some versions of `torch` via PIP alongside `sentence-transformers` can result in silent OpenMP thread conflicts. To remedy this, it is recommended to install a matching PyTorch wheel directly from the official PyTorch channels).*

### 2. Tabular Data Preparation
We wrote `scripts/create_csv.py` to hardcode the exact values for Infosys, TCS, and Wipro across FY21—FY24 into a `pandas` DataFrame, yielding the structured `data/financials.csv` dataset.

### 3. Local Document Indexing Tool
Because real 300-page Annual Reports are densely packed PDFs, we implemented a custom chunking architecture (450 words max, 50-word overlaps) in `src/tools/search_docs.py`. We explicitly capture the upper-case Headers to attach **metadata context** alongside the embedded numerical vectors inside our `FAISS` L2 index object. 
To build the indexes:
```bash
python scripts/index_docs.py
```

### 4. Structuring the Tools
We provided exactly 3 Python functions acting as precise tools for the agent:
* **search_docs:** Given a topic, encode the text using `all-mpnet-base-v2` and fetch the top `K` most relevant snippets from the `.faiss` index mappings.
* **query_data:** Using pandas keyword matching rules to map exact quantitative answers (Revenue, Headcount, Comparisons, Rankings) directly rather than letting the LLM hallucinate or guess numeric figures. 
* **web_search:** Using `TavilyClient`, this fetches up-to-date FY25 information for things that aren't natively stored locally, handling limits and strictly restricting searches via its prompt rules.

### 5. Orchestrating the Reasoning Loop (Agent.py)
Our `src/agent.py` loops sequentially **up to a strict count of 8 steps limit** processing messages locally without overhead from LangChain. 
* The **Groq API** processes the instructions at lightning speed.
* Every call strictly monitors `tool_calls`.
* We injected our `check_refusal` guardrail directly in at step 1. Before making a single API call, checking the hardcoded trigger filters like "Should I buy" or "investing". 
* Results and tool parameters are logged in a clean text pipeline under `/traces/`.

### Testing Pipeline
Four distinct unit tests in `scripts/verify_tools.py` can be executed to validate that the tools operate as expected (parsing text natively, hitting the CSV values securely, and validating the API constraints).

```bash
# To verify the integrity of the setup locally:
python scripts/verify_tools.py
# To sweep 20 evaluation queries through the agent:
python eval/run_eval.py
```
