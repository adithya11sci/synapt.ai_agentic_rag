import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
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
    
    print(f"Found {len(pdf_paths)} PDF(s) to index:")
    for p in pdf_paths:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  - {p.name} ({size_mb:.1f} MB)")
    
    start = time.time()
    chunk_and_index(pdf_paths, "data/faiss_index")
    end = time.time()
    
    # Report chunk count
    meta_file = Path("data/faiss_index/metadata.json")
    if meta_file.exists():
        with open(meta_file, "r") as f:
            data = json.load(f)
        total_chunks = len(data.get("chunks", []))
        print(f"\nTotal chunks indexed: {total_chunks}")
    
    print(f"Indexing completed in {end - start:.2f} seconds.")

if __name__ == "__main__":
    index_docs()
