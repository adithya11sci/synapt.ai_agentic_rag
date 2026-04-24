import sys
import json
import os
import re
import time
import logging
from groq import Groq
from dotenv import load_dotenv

from .tools.search_docs import search_docs
from .tools.query_data import query_data
from .tools.web_search import web_search
from .refusal import check_refusal

load_dotenv()

logger = logging.getLogger("agentic_rag.agent")


def _validate_env():
    """Validate required environment variables are set."""
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set. Please add it to your .env file.")
    if not os.environ.get("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set — web_search tool will be unavailable.")


def _parse_failed_generation(failed_gen: str):
    """
    Parse the malformed <function=name {...}></function> format that
    LLaMA-3.3-70B sometimes emits, returning (func_name, args_dict) or None.
    """
    # Pattern: <function=TOOL_NAME {JSON}>\n or <function=TOOL_NAME {JSON}</function>
    pattern = r'<function=([\w_]+)\s*({.*?})'
    match = re.search(pattern, failed_gen, re.DOTALL)
    if match:
        func_name = match.group(1)
        try:
            args = json.loads(match.group(2))
            return func_name, args
        except json.JSONDecodeError:
            pass
    return None


def _make_synthetic_tool_call_response(func_name: str, args: dict):
    """
    Build a minimal object that mimics a Groq ChatCompletion response
    containing a single tool call, so the agent loop can dispatch it normally.
    """
    import uuid

    class _FunctionCall:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = json.dumps(arguments)

    class _ToolCall:
        def __init__(self, name, args):
            self.id = f"call_{uuid.uuid4().hex[:8]}"
            self.type = "function"
            self.function = _FunctionCall(name, args)

    class _Message:
        def __init__(self, tool_call):
            self.content = None
            self.tool_calls = [tool_call]

    class _Choice:
        def __init__(self, message):
            self.message = message
            self.finish_reason = "tool_calls"

    class _Response:
        def __init__(self, choice):
            self.choices = [choice]

    tc = _ToolCall(func_name, args)
    return _Response(_Choice(_Message(tc)))


def _call_llm(client, model, messages, tools=None, max_retries=3):
    """Call Groq API with exponential backoff retry.
    
    Handles the tool_use_failed 400 error that LLaMA-3.3-70B emits when it
    generates <function=name {...}> syntax instead of proper JSON tool calls.
    """
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, messages=messages, temperature=0.0)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            # ── LLaMA tool_use_failed: parse the malformed generation ──
            if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
                failed_gen = ""
                try:
                    # Groq wraps the raw error body; extract failed_generation
                    body = e.response.json() if hasattr(e, "response") else {}
                    failed_gen = body.get("error", {}).get("failed_generation", "")
                except Exception:
                    pass
                # Also try extracting from the string representation
                if not failed_gen:
                    fg_match = re.search(r"'failed_generation':\s*'(.*?)'", err_str, re.DOTALL)
                    if fg_match:
                        failed_gen = fg_match.group(1)
                if failed_gen:
                    parsed = _parse_failed_generation(failed_gen)
                    if parsed:
                        func_name, args = parsed
                        logger.warning(
                            f"tool_use_failed recovered: synthesising tool call "
                            f"{func_name}({list(args.keys())})"
                        )
                        return _make_synthetic_tool_call_response(func_name, args)
                logger.error(f"tool_use_failed and could not parse failed_generation: {failed_gen[:200]}")
            # ── Generic retry with backoff ──
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Groq API call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Groq API call failed after {max_retries} attempts: {e}")
                raise


def run_agent(question):
    """Orchestrates tools via an LLM agent loop. Hard cap at 8 tool calls."""
    _validate_env()

    refusal_msg = check_refusal(question)
    if refusal_msg:
        logger.info(f"Refusal triggered for: {question[:60]}...")
        return {"answer": refusal_msg, "steps_used": 0, "trace": [], "refused": True}

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    logger.info(f"Agent starting | model={model} | question={question[:80]}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": (
                    "Use this tool to search the Infosys, TCS, and Wipro FY24 annual "
                    "report PDFs for qualitative information. Use it when the question "
                    "asks for explanations, management commentary, strategic priorities, "
                    "reasons behind financial results, risk factors, ESG initiatives, "
                    "CEO or CFO statements, large deal narratives, or any narrative text "
                    "from the reports. Do NOT use for specific financial numbers or for "
                    "current market data — use query_data or web_search for those. "
                    "Input: a natural language query string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A natural language search query."}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_data",
                "description": (
                    "Use this tool to query the structured financial CSV containing 4 "
                    "years of key metrics (FY21 to FY24) for Infosys, TCS, and Wipro: "
                    "revenue in crore rupees, operating margin percent, net profit in "
                    "crore rupees, EPS in rupees, and headcount. Use it when the question "
                    "asks for a specific number, a year-over-year comparison, a ranking, "
                    "or any calculation on financial metrics. Do NOT use for qualitative "
                    "explanations or recent news — use search_docs or web_search instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "A financial data query string."}
                    },
                    "required": ["question"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Use this tool ONLY for information that cannot exist in a FY24 "
                    "annual report: current stock prices, analyst ratings from the past "
                    "few weeks, news published after April 2024, recent executive "
                    "appointments, or FY25 quarterly results. Do NOT use for historical "
                    "financial data (FY21 to FY24) or for content from the annual reports. "
                    "Input: a short specific search query under 10 words."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "A short web search query."}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    SYSTEM_PROMPT = (
        "You are a financial research agent that answers questions about "
        "Infosys, TCS, and Wipro using three tools: search_docs, query_data, "
        "and web_search.\n\n"
        "TOOL SELECTION RULES:\n"
        "- search_docs  → qualitative: reasons, strategy, MD&A, commentary, "
        "CEO statements, ESG, risk factors, narratives. "
        "Do NOT use for numbers or live data.\n"
        "- query_data   → quantitative: revenue, margin, profit, EPS, "
        "headcount, comparisons, rankings. "
        "Do NOT use for explanations or recent news.\n"
        "- web_search   → live only: current price, recent news post-April "
        "2024, analyst ratings, FY25 results. "
        "Do NOT use for historical data in the reports.\n\n"
        "ANSWER RULES:\n"
        "1. Cite every claim: document name + page, CSV row, or URL + date.\n"
        "2. Never guess. If no tool returned relevant data, say so.\n"
        "3. For questions needing a number AND explanation: call query_data "
        "first, then search_docs.\n"
        "4. Hard cap: 8 tool calls maximum per question.\n"
        "5. If the user asks an out-of-domain question (e.g., general trivia, recipes, non-financial topics), politely refuse to answer and state that you are a financial research agent focused on Infosys, TCS, and Wipro.\n\n"
        "OUTPUT FORMAT:\n"
        "Answer: [your answer with inline citations]\n"
        "Citations: [tool used → source → page or row or url]\n"
        "Steps used: [N] / 8\n\n"
        "CRITICAL: You MUST use the native tool-calling mechanism provided. "
        "NEVER write <function=name {...}> style tags. "
        "NEVER write function calls in plain text. "
        "Use ONLY the structured tool_calls format supplied by the API."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    MAX_STEPS = 8
    steps_used = 0
    trace = []

    available_functions = {
        "search_docs": search_docs,
        "query_data": query_data,
        "web_search": web_search,
    }

    while steps_used < MAX_STEPS:
        response = _call_llm(client, model, messages, tools=tools)
        msg = response.choices[0].message

        if msg.tool_calls:
            # We must serialize the message correctly for groq
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            for tool_call in msg.tool_calls:
                steps_used += 1
                func_name = tool_call.function.name

                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in tool arguments for {func_name}")
                    func_args = {}

                input_arg = func_args.get("query") or func_args.get("question") or str(func_args)
                logger.info(f"Step {steps_used}: {func_name}({input_arg[:60]})")

                if steps_used >= MAX_STEPS:
                    tool_result = "Hard cap of 8 tool calls reached. Summarize what you have and tell the user."
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "name": func_name, "content": tool_result
                    })
                    trace.append({"step": steps_used, "tool": func_name, "input": input_arg, "result": tool_result})
                    final = _call_llm(client, model, messages)
                    return {"answer": final.choices[0].message.content, "steps_used": steps_used,
                            "trace": trace, "refused": False, "cap_hit": True}

                # Safe tool dispatch
                function_to_call = available_functions.get(func_name)
                if not function_to_call:
                    logger.warning(f"LLM requested unknown tool: {func_name}")
                    function_response = f"Error: Unknown tool '{func_name}'. Available: {list(available_functions.keys())}"
                else:
                    try:
                        function_response = function_to_call(**func_args)
                    except Exception as e:
                        logger.error(f"Tool {func_name} raised an error: {e}")
                        function_response = f"Error executing {func_name}: {str(e)}"

                messages.append({
                    "role": "tool", "tool_call_id": tool_call.id,
                    "name": func_name, "content": function_response
                })
                trace.append({"step": steps_used, "tool": func_name, "input": input_arg, "result": function_response[:500]})
        else:
            return {"answer": msg.content, "steps_used": steps_used, "trace": trace, "refused": False}

    return {"answer": "Failed to complete within step limit.", "steps_used": steps_used,
            "trace": trace, "refused": False, "cap_hit": True}


def write_trace(question, result, filepath):
    """Write a human-readable trace file for a single agent run."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {result['answer']}\n")
        f.write(f"Steps used: {result['steps_used']} / 8\n")
        if result.get("refused"):
            f.write("Status: REFUSED (no tools called)\n")
        if result.get("cap_hit"):
            f.write("Status: CAP HIT (8 tool calls reached)\n")
        f.write("\n--- Trace ---\n")
        for t in result['trace']:
            f.write(f"Step {t['step']}: tool={t['tool']}  input='{t['input']}'\n")
            f.write(f"             result={t['result']}\n")


if __name__ == "__main__":
    q = sys.argv[1]
    res = run_agent(q)
    print(res["answer"])
    print(f"Steps used: {res['steps_used']} / 8")
    write_trace(q, res, "traces/trace.txt")
