# Agentic RAG System for Financial Data Analysis

## Overview
This project implements an intelligent **Agentic Retrieval-Augmented Generation (RAG) system** utilizing Large Language Models (LLMs) via the Groq API (`llama-3.1-8b-instant`). The system acts as a financial data analyst capable of robustly answering queries related to major IT companies (Infosys, TCS, Wipro). It autonomously selects the appropriate tools for text extraction, tabular data querying, and live web search based on the user's intent.

## System Architecture & Workflow

The core of this system relies on a **ReAct (Reasoning + Acting)** style agent loop. Instead of manually specifying where to look for data, the AI dynamically decides based on the user prompt.

1. **User Query:** The user submits a financial question (e.g., "What was the operating margin of Infosys?" or "What is TCS's stock price today?").
2. **Agent Router (Groq LLM):** The LLM analyzes the query, consults its list of capabilities, and calls the appropriate system tool:
   - `search_docs`: For qualitative questions (e.g., "What are the core pillars of Project Maximus?"). This queries the local vector database.
   - `query_data`: For quantitative/tabular questions (e.g., "What was the operating margin for XYZ company in FY24?"). This queries structured CSVs.
   - `web_search`: For real-time, dynamic market data (e.g., "What is the current stock price of TCS?"). This queries the live internet.
3. **Tool Execution:** The Python backend intercepts the LLM's tool-call request, executes the corresponding local Python function, and passes the raw result back to the LLM context.
4. **Final Synthesis:** The LLM reads the tool's raw output, extracts exactly the data needed, and synthesizes it into a natural, comprehensive human-readable response.
5. **Strict Guardrails:** If a user asks for direct investment advice (e.g., "Should I buy Infosys stock right now?"), the system's strict system prompt engages and immediately refuses to provide financial advice, returning a safety warning instead.

## Tech Stack (What it is using)

- **LLM Engine:** [Groq API](https://groq.com/) using the `llama-3.1-8b-instant` model for lightning-fast inference and dynamic function calling.
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) for high-performance localized similarity search of document chunks.
- **Embeddings:** [SentenceTransformers](https://www.sbert.net/) (via HuggingFace) to convert raw text from PDFs into dense vector representations for similarity matching.
- **Tabular Engine:** [Pandas](https://pandas.pydata.org/) for structuring and querying exact numerical financial data directly from CSV datasets.
- **Web Search Integration:** [Tavily API](https://tavily.com/) optimized for AI agents to retrieve live market data, current stock prices, and recent news.

## Repository Structure

```text
📦 agentic_rag
├── 📂 data/
│   ├── 📂 faiss_index/    # Compiled FAISS localized vector database
│   ├── 📂 pdfs/           # Generated/Source qualitative financial reports
│   └── 📜 financial_data.csv # Structured quantitative financial datasets
├── 📂 scripts/
│   ├── 📜 create_csv.py   # Generates the tabular financial datasets
│   ├── 📜 index_docs.py   # Parses PDFs, chunks text, embeddings, and builds the FAISS index
│   └── 📜 verify_tools.py # Validation script to test individual tool integrations
├── 📂 src/
│   ├── 📂 tools/
│   │   ├── 📜 query_data.py # Logic for querying CSV tabular data via Pandas
│   │   ├── 📜 search_docs.py# FAISS vector search logic for querying PDF text
│   │   └── 📜 web_search.py # Tavily API integration for live web scraping
│   └── 📜 agent.py        # Core LLM prompt, routing logic, execution loop, and tool registry
├── 📜 test_quick.py       # Main entry point to interact with the finished Agentic RAG
├── 📜 requirements.txt    # Project dependencies
└── 📜 .env                # Local secrets (API keys) - Safely ignored by Git
```

## Setup & Installation

1. **Install dependencies:**
   Make sure you are in the project folder and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file in the root directory and populate your API keys. *Note: `.env` is ignored by `.gitignore` to prevent leaking secrets to GitHub.*
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```

## How to Run

1. **Initialize the Data & Vector Store:**
   Generate the PDF/CSV data and build the FAISS embedding index locally by running the setup scripts:
   ```bash
   python scripts/create_csv.py
   python scripts/index_docs.py
   ```

2. **Test the Application Workflow:**
   Run the quick test script to see the agent dynamically route questions through the Document RAG, Tabular RAG, and Web Search pipelines:
   ```bash
   python -X utf8 test_quick.py
   ```

## Disclaimer
*This project is created for demonstration and educational purposes as an internship assignment evaluation. The AI model output does not constitute legitimate professional financial advice.*