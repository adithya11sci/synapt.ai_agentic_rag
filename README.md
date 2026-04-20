# Agentic RAG System for Financial Data Analysis

## Overview
This project implements an intelligent Agentic Retrieval-Augmented Generation (RAG) system utilizing Large Language Models (LLMs) via the Groq API (`llama-3.1-8b-instant`). The system acts as a financial data analyst capable of robustly answering queries related to major IT companies (Infosys, TCS, Wipro) by autonomously selecting the appropriate tools for text extraction, tabular data querying, and live web search.

## Features
- **Document RAG (Text Data):** Utilizes `SentenceTransformers` and `FAISS` to parse, index, and retrieve relevant qualitative financial information straight from PDF corporate reports.
- **Tabular RAG (CSV Data):** Uses the pandas library to query and extract precise quantitative numbers (such as specific operating margins, revenues, and key financial metrics).
- **Web Search (Live Data):** Integrates the Tavily API to fetch real-time data, handling dynamic queries such as fetching current stock prices.
- **Agentic Routing & Reasoning:** The LLM autonomously determines the intent of the user's prompt and selects the best tool or combination of tools to formulate a highly accurate response.
- **Strict Guardrails:** The agent is explicitly programmed through rigid system prompts to refuse to provide personal investment or financial advice.

## Environment Setup
In order to run the workflow locally, you must create a `.env` file in the root directory (Note: `.env` is intentionally ignored by Git to prevent leaking secrets):

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## How to Run
1. **Install Dependencies:**
   Ensure you have installed all required dependencies (`faiss-cpu`, `sentence-transformers`, `pandas`, `reportlab`, `python-dotenv`). You can often do this via:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Data:** 
   Run the setup scripts to generate the initial PDFs, construct the CSVs, and build the FAISS vector database indices:
   ```bash
   python scripts/create_csv.py
   python scripts/index_docs.py
   ```

3. **Execute Queries:**
   Use the provided testing script to run the complete agent workflow and evaluate the responses:
   ```bash
   python -X utf8 test_quick.py
   ```

## Repository Structure
- `data/`: Contains the raw generated PDFs, structured CSV data, and the FAISS vector index files.
- `scripts/`: Initialization and setup scripts for building the document indices and validating the tools.
- `src/`: Core logic including the `agent.py` router and `tools/` directory containing the modular toolsets (`search_docs`, `query_data`, `web_search`).
- `test_quick.py`: The main entry point script to validate the prompts and interact with the RAG agent in action.

## Disclaimer
_This project is created for demonstration and educational purposes as an internship assignment evaluation. The AI model output does not constitute legitimate professional financial advice._