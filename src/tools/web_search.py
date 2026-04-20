import os
import threading
from tavily import TavilyClient

def web_search(query):
    """Retrieve live facts like stock prices, recent news, or Q1 FY25 results."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Web search failed: Tavily API key missing. Use search_docs or query_data."
        
    client = TavilyClient(api_key=api_key)
    result_holder = {}
    
    def search():
        try:
            res = client.search(query=query, search_depth="basic", max_results=3,
                                include_answer=False, include_raw_content=False)
            result_holder["res"] = res
        except Exception as e:
            result_holder["err"] = str(e)
            
    thread = threading.Thread(target=search)
    thread.start()
    thread.join(timeout=10)
    
    if thread.is_alive():
         return "Web search timed out. Use search_docs or query_data instead."
         
    if "err" in result_holder:
        return f"Web search failed: {result_holder['err']}. Use search_docs or query_data."
        
    res = result_holder.get("res", {})
    results = res.get("results", [])
    
    if not results:
        return "No web search results found."
        
    formatted = []
    for i, hit in enumerate(results):
        url = hit.get('url', 'Unknown URL')
        date = hit.get('published_date', 'Unknown Date')
        content = hit.get('content', '')[:300]
        formatted.append(f"[{i+1}] URL: {url}\n    Date: {date}\n    Snippet: {content}")
        
    return "\n".join(formatted)
