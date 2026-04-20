import sys
import json
import os
from groq import Groq
from dotenv import load_dotenv

from .tools.search_docs import search_docs
from .tools.query_data import query_data
from .tools.web_search import web_search
from .refusal import check_refusal

load_dotenv()

def run_agent(question):
    """Orchestrates tools via an LLM. Limits to 8 calls."""
    refusal_msg = check_refusal(question)
    if refusal_msg:
        return {"answer": refusal_msg, "steps_used": 0, "trace": [], "refused": True}
        
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Searches the FY24 annual reports for qualitative information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "a search query."}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_data",
                "description": "Queries structured financial CSV containing FY21 to FY24 metrics: revenue, margin, profit, EPS, headcount.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "query string."}
                    },
                    "required": ["question"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Searches the web for live stock prices and news.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "a search query."}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    SYSTEM_PROMPT = """You are a financial research assistant answering questions about Infosys, Wipro, and TCS. 
CRITICAL RULES:
1. ALWAYS extract the exact actual data from the tool responses (numbers, texts, facts) and include them in your final Answer. Do NOT refer to "unspecified row" or just output the citation.
2. If the user asks for a margin, or a project, you MUST output the exact margin or project description that you found from the tool! This is mandatory.
3. Cite your sources inline in parenthesis.
4. Output your detailed natural language answer to the user."""

    
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
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )
        
        msg = response.choices[0].message
        
        if msg.tool_calls:
            messages.append(msg)
            
            for tool_call in msg.tool_calls:
                steps_used += 1
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                # Check for arguments mapping
                input_arg = func_args.get("query") or func_args.get("question")
                
                if steps_used >= MAX_STEPS:
                    tool_result = "Hard cap reached. Compose a refusal."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })
                    trace.append({"step": steps_used, "tool": func_name, "input": input_arg, "result": tool_result[:500]})
                    final_response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                    return {"answer": final_response.choices[0].message.content, "steps_used": steps_used, "trace": trace, "refused": False, "cap_hit": True}
                
                function_to_call = available_functions[func_name]
                function_response = function_to_call(**func_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": function_response
                })
                
                trace.append({"step": steps_used, "tool": func_name, "input": input_arg, "result": function_response[:500]})
        else:
            return {"answer": msg.content, "steps_used": steps_used, "trace": trace, "refused": False}
            
    return {"answer": "Failed to complete.", "steps_used": steps_used, "trace": trace, "refused": False, "cap_hit": True}

def write_trace(question, result, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {result['answer']}\n")
        f.write(f"Steps used: {result['steps_used']} / 8\n")
        for t in result['trace']:
            f.write(f"Step {t['step']}: {t['tool']}({t['input']}) -> {t['result']}\n")

if __name__ == "__main__":
    q = sys.argv[1]
    res = run_agent(q)
    print(res["answer"])
    print(res["steps_used"])
    write_trace(q, res, "traces/trace.txt")
