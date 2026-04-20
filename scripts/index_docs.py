from pathlib import Path
import time
from src.tools.search_docs import chunk_and_index

def index_docs():
    """Indexes all PDFs in data/pdfs/ into a FAISS index."""
    pdf_dir = Path("data/pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print("No PDFs found in data/pdfs. Exiting.")
        return
        
    start = time.time()
    chunk_and_index(pdf_paths, "data/faiss_index")
    end = time.time()
    
    print(f"Indexing completed in {end - start:.2f} seconds.")

if __name__ == "__main__":
    index_docs()
