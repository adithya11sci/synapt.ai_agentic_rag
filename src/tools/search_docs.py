import re
import json
import faiss
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

def chunk_and_index(pdf_paths, index_path):
    """Parses PDFs, chunks text with nearest sections, embeds them, and saves a FAISS index."""
    chunks = []
    metadata = []
    
    # Load model
    model = SentenceTransformer("all-mpnet-base-v2")
    
    for path in pdf_paths:
        path = Path(path)
        if not path.exists():
            continue
            
        reader = PdfReader(path)
        current_section = "Unknown Section"
        
        # Simple section header heuristic (all caps lines)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
                
            words = text.split()
            current_chunk = []
            
            for word in words:
                # Update section heuristics if word seems like a heading part
                if word.isupper() and len(word) > 2:
                    potential_section = word
                
                current_chunk.append(word)
                
                if len(current_chunk) >= 450:
                    chunk_text = " ".join(current_chunk)
                    full_text = f"Section: {current_section}\n{chunk_text}"
                    
                    chunks.append(full_text)
                    metadata.append({
                        "source": path.name,
                        "page": page_num + 1,
                        "section": current_section
                    })
                    
                    # 50 word overlap
                    current_chunk = current_chunk[-50:]
            
            # Add final chunk of the page if substantial
            if len(current_chunk) > 50:
                chunk_text = " ".join(current_chunk)
                full_text = f"Section: {current_section}\n{chunk_text}"
                chunks.append(full_text)
                metadata.append({
                    "source": path.name,
                    "page": page_num + 1,
                    "section": current_section
                })
                
    if not chunks:
        print("No chunks to index.")
        return

    embeddings = model.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    
    out_dir = Path(index_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"metadata": metadata, "chunks": chunks}, f)

def search_docs(query, top_k=3):
    """Retrieve qualitative info from annual report PDFs."""
    index_dir = Path("data/faiss_index")
    index_file = index_dir / "index.faiss"
    meta_file = index_dir / "metadata.json"
    
    if not index_file.exists() or not meta_file.exists():
        return "Error: Index files not found. Please run indexing script."
        
    try:
        index = faiss.read_index(str(index_file))
        with open(meta_file, "r") as f:
            data = json.load(f)
            
        metadata = data["metadata"]
        chunks = data["chunks"]
        
        model = SentenceTransformer("all-mpnet-base-v2")
        query_emb = model.encode([query]).astype("float32")
        
        distances, indices = index.search(query_emb, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            meta = metadata[idx]
            chunk_text = chunks[idx]
            clean_text = re.sub(r'\s+', ' ', chunk_text)[:300]
            
            res = f"[{i+1}] Source: {meta['source']} | Page: {meta['page']} | Section: {meta['section']}\n"
            res += f"    {clean_text}..."
            results.append(res)
            
        return "\n".join(results) if results else "No relevant documents found."
    except Exception as e:
        return f"Error during document search: {str(e)}"
