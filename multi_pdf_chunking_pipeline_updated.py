# src/multi_pdf_chunking_pipeline.py

# ==========================
# 📘 Multi-PDF Chunking Pipeline (PyMuPDF + Tiktoken)
# ==========================
# Usage: python src/multi_pdf_chunking_pipeline.py

import os
import re
import json
import shutil
from pathlib import Path
import datetime
import tiktoken
import fitz  # PyMuPDF

# ------------------------- CONFIG -------------------------
# 512 is often better for retrieval accuracy than 1024
CHUNK_SIZE = 512  
CHUNK_OVERLAP = 64
MODEL_NAME = "gpt-4"

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Input: Where you drop your PDFs
RAW_DIR = os.path.join(BASE_DIR, "data", "raw") 
# Output: Where JSONs go for the ingestion script
PROCESSED_ROOT = os.path.join(BASE_DIR, "data", "processed")

# ------------------------- HELPERS -------------------------
encoder = tiktoken.encoding_for_model(MODEL_NAME)

def now_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def slugify(s: str) -> str:
    """Creates a filesystem-safe filename from a string."""
    s = str(s).strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

def tokenize(text: str):
    return encoder.encode(text)

def detokenize(tokens):
    return encoder.decode(tokens)

def chunk_text_tokens(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks based on token count."""
    tokens = tokenize(text)
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    
    step = chunk_size - overlap
    chunks = []
    
    # If text is smaller than one chunk, return it as is
    if len(tokens) <= chunk_size:
        return [detokenize(tokens)]

    for i in range(0, len(tokens), step):
        segment = tokens[i:i + chunk_size]
        # Only add chunk if it has content
        if segment:
            chunks.append(detokenize(segment))
            
    return chunks

def extract_text_smart(pdf_path: str):
    """
    Extracts text while trying to preserve reading order (good for columns).
    Returns a list of (page_num, text) tuples.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        # "blocks" sorts text by vertical/horizontal position (better for columns)
        text = page.get_text("text", sort=True) 
        # Clean up excessive whitespace/hyphenation if needed
        text = text.replace(" - ", "-") 
        if text.strip():
            pages.append((i + 1, text.strip()))
    doc.close()
    return pages

# ------------------------- MAIN EXECUTION -------------------------
def main():
    print(f"📂 Scanning for PDFs in: {RAW_DIR}")
    
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
        print(f"⚠️ Created missing directory: {RAW_DIR}")
        print("👉 Please place your .pdf files there and run this script again.")
        return

    # User input for Namespace
    org_input = input("🏢 Enter organization name (e.g., Amazon, Nvidia): ").strip()
    if not org_input:
        print("❌ Organization name is required.")
        return
        
    org_id = slugify(org_input)
    
    # Create a specific output folder for this organization
    # e.g., src/data/processed/amazon/
    org_output_dir = os.path.join(PROCESSED_ROOT, org_id)
    os.makedirs(org_output_dir, exist_ok=True)

    # Find PDFs
    uploaded_files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith('.pdf')])

    if not uploaded_files:
        print("⚠️ No PDF files found in raw directory.")
        return

    # Process each PDF
    for pdf_index, filename in enumerate(uploaded_files, start=1):
        pdf_path = os.path.join(RAW_DIR, filename)
        pdf_stem = Path(filename).stem
        pdf_slug = slugify(pdf_stem)
        
        # Unique IDs
        pdf_id = f"{org_id}_{pdf_slug}"
        output_filename = f"{pdf_id}.json"
        output_path = os.path.join(org_output_dir, output_filename)

        print(f"\n📄 Processing [{pdf_index}/{len(uploaded_files)}]: {filename}")
        
        # 1. Extract Text
        pages = extract_text_smart(pdf_path)

        timestamp = now_timestamp()
        all_chunks = []
        total_chunks = 0

        # 2. Chunking Logic
        for page_num, page_text in pages:
            page_chunks = chunk_text_tokens(page_text)
            
            for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
                # Unique Chunk ID (Critical for Vector DBs)
                chunk_id = f"{pdf_id}_p{page_num}_c{chunk_idx}"
                
                metadata = {
                    "chunk_id": chunk_id,
                    "organization_id": org_id,
                    "namespace": org_id, # This matches what ingestion script expects
                    "pdf_name": filename,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(tokenize(chunk_text)),
                    "source_path": str(Path(pdf_path).name) # Store relative name, not absolute path
                }
                
                entry = {
                    "page_content": chunk_text,
                    "metadata": metadata,
                    "ingestion_timestamp": timestamp,
                    "language": "en"
                }
                all_chunks.append(entry)
                total_chunks += 1

        # 3. Save JSON
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_chunks, fh, indent=2, ensure_ascii=False)

        print(f"   ✅ Saved {total_chunks} chunks to {output_path}")

    print(f"\n🎉 Done! Processed files are ready in: {org_output_dir}")
    print("👉 Now run 'python src/genai/retrivel/ingest.py' to upload to Pinecone.")

# ------------------------- ENTRY POINT -------------------------
if __name__ == "__main__":
    main()