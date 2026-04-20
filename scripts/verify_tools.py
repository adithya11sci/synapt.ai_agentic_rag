from src.tools.search_docs import search_docs
from src.tools.query_data import query_data
from src.tools.web_search import web_search
from src.refusal import check_refusal

def main():
    """Verify tool behavior."""
    passed = 0

    print("Test 1 — search_docs:")
    res1 = search_docs("Project Maximus five pillars margin expansion")
    if "[1]" in str(res1) and "Page:" in str(res1):
        print("PASS")
        passed += 1
    else:
        print("FAIL: Expected '[1]' and 'Page:' in result")

    print("\nTest 2 — query_data:")
    res2 = query_data("What was Infosys operating margin in FY24?")
    if "20.7" in str(res2):
        print("PASS")
        passed += 1
    else:
        print("FAIL: Expected '20.7' in result")

    print("\nTest 3 — web_search:")
    res3 = web_search("Infosys stock price today")
    if "URL:" in str(res3):
        print("PASS")
        passed += 1
    else:
        print("FAIL: Expected 'URL:' in result")

    print("\nTest 4 — refusal:")
    res4 = check_refusal("Should I buy Infosys stock?")
    if res4 is not None:
        print("PASS")
        passed += 1
    else:
        print("FAIL: Expected non-None return value")

    print(f"\n{passed}/4 tests passed.")

if __name__ == "__main__":
    main()
