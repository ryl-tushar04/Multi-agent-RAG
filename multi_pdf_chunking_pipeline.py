# ==========================
# 📘 Multi-PDF Chunking Pipeline (Markitdown + PyMuPDF)
# ==========================


# pip install markitdown[pdf] tiktoken PyMuPDF
# Place your PDFs in the same directory as the script.


import os
import re
import json
import shutil
from pathlib import Path
import datetime
from markitdown import MarkItDown
import tiktoken
import fitz  # PyMuPDF

# ------------------------- CONFIG -------------------------
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 64
MODEL_NAME = "gpt-4"
OUTPUT_DIR = "Processed_JSON"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------- HELPERS -------------------------
md = MarkItDown()
encoder = tiktoken.encoding_for_model(MODEL_NAME)

def now_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

def tokenize(text: str):
    return encoder.encode(text)

def detokenize(tokens):
    return encoder.decode(tokens)

def chunk_text_tokens(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = tokenize(text)
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(tokens), step):
        segment = tokens[i:i + chunk_size]
        if not segment:
            break
        chunks.append(detokenize(segment))
    return chunks

def extract_pages_pymupdf(pdf_path: str):
    """Return list of (page_number, page_text) for pages that contain text."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        pages.append((i + 1, text.strip()))
    doc.close()
    return pages

# ------------------------- MAIN EXECUTION -------------------------
def main():
    print("📤 Place your PDF files in the same directory as this script (or specify paths).")
    print("After placing them, the script will process all .pdf files in the folder.")

    org_input = input("🏢 Enter organization name (e.g., Cognizant): ").strip() or "org_manual"
    org_id = slugify(org_input)

    # Find all PDF files in current directory
    uploaded_files = sorted([f for f in os.listdir('.') if f.lower().endswith('.pdf')])

    if not uploaded_files:
        print("⚠️ No PDF files found in current directory.")
        return

    # Process each uploaded PDF
    for pdf_index, filename in enumerate(uploaded_files, start=1):
        pdf_path = os.path.join(os.getcwd(), filename)
        pdf_stem = Path(filename).stem
        pdf_slug = slugify(pdf_stem)
        pdf_id = f"{org_id}_pdf{pdf_index:03d}"
        pdf_name_combined = f"{org_id}_{pdf_slug}.pdf"
        output_filename = f"{org_id}_{pdf_slug}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n📄 Processing [{pdf_index}/{len(uploaded_files)}]: {filename}")
        pages = extract_pages_pymupdf(pdf_path)

        timestamp = now_timestamp()
        all_chunks = []
        total_chunks = 0

        for page_num, page_text in pages:
            if not page_text:
                continue
            page_chunks = chunk_text_tokens(page_text)
            for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = f"{org_id}_pdf{pdf_index:03d}_page{page_num:02d}_chunk{chunk_idx:03d}"
                metadata = {
                    "chunk_id": chunk_id,
                    "organization_id": org_id,
                    "namespace": f"{org_id}_namespace",
                    "pdf_id": pdf_id,
                    "organization_name": org_input,
                    "pdf_name": pdf_name_combined,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(tokenize(chunk_text)),
                    "pdf_path": str(Path(pdf_path).absolute())
                }
                entry = {
                    "page_content": chunk_text,
                    "metadata": metadata,
                    "ingestion_timestamp": timestamp,
                    "language": "en",
                    "chunk_source": "text-layer"
                }
                all_chunks.append(entry)
                total_chunks += 1

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_chunks, fh, indent=2, ensure_ascii=False)

        print(f"  ✅ Saved {total_chunks} chunks → {output_path}")

    # ------------------------- ZIP & SAVE -------------------------
    zip_base = "processed_jsons_archive"
    shutil.make_archive(zip_base, 'zip', OUTPUT_DIR)
    zip_path = zip_base + ".zip"
    print("\n📦 Created ZIP:", zip_path)

    print("\n✅ Done. JSON outputs are in:", OUTPUT_DIR)

# ------------------------- ENTRY POINT -------------------------
if __name__ == "__main__":
    main()
