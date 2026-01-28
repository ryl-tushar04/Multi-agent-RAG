# src/genai/retrivel/ingest.py

import os
import json
import hashlib
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ------------------ CONFIGURATION ------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Error: PINECONE_API_KEY or PINECONE_INDEX_NAME missing in .env")

EMBED_DIM = 384
BATCH_SIZE = 50

# Registry to track files is kept in the same folder as this script
REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "ingestion_registry.json")

# ------------------ INITIALIZATION ------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)

# Ensure Index Exists
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)

# ------------------ HELPER FUNCTIONS ------------------
def get_file_hash(filepath):
    """Generates MD5 hash to detect file changes."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_registry(registry):
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)

# ------------------ MAIN INGESTION LOOP ------------------
def ingest_processed_data(processed_root):
    """
    Reads JSON files from src/data/processed/{namespace_folder}/
    and ingests them into Pinecone if they are new.
    """
    registry = load_registry()
    registry_updated = False
    
    print(f"🔍 Scanning Processed Data at: {processed_root}")
    
    if not os.path.exists(processed_root):
        print(f"❌ Error: processed folder not found at {processed_root}")
        print("Please run your PDF chunking pipeline first to generate JSONs.")
        return

    # Iterate over subfolders (namespaces) in 'processed'
    # Example: src/data/processed/amazon/
    subfolders = [f for f in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, f))]
    
    if not subfolders:
        print("⚠️ No namespace folders found in 'processed'.")
        print("Expected structure: src/data/processed/amazon/file.json")
        return

    for folder_name in subfolders:
        folder_path = os.path.join(processed_root, folder_name)
        
        # Namespace logic: Use the folder name (e.g., 'amazon' -> 'amazon')
        namespace = folder_name.lower().replace(" ", "_")
        print(f"\n📂 Checking Namespace: '{namespace}'")

        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        files_to_process = []

        # --- Filter: Identify ONLY new or changed files ---
        for jf in json_files:
            file_path = os.path.join(folder_path, jf)
            current_hash = get_file_hash(file_path)
            
            # Unique ID for the registry: namespace + filename
            registry_key = f"{namespace}::{jf}"

            # Check if hash matches what we have on record
            if registry.get(registry_key) == current_hash:
                continue 
            
            files_to_process.append((jf, file_path, registry_key, current_hash))

        if not files_to_process:
            print(f"   ✅ All files up to date.")
            continue
        
        print(f"   ⚡ Ingesting {len(files_to_process)} new/modified files...")

        # --- Process the files ---
        total_vectors = 0
        
        for jf, file_path, registry_key, current_hash in tqdm(files_to_process, desc="Upserting"):
            vectors = []
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"   ❌ Failed to read {jf}: {e}")
                continue
            
            if not isinstance(data, list): 
                print(f"   ⚠️ format error in {jf} (must be a list of chunks)")
                continue

            for i, chunk in enumerate(data):
                # Handle dictionary format (standard) vs string
                if isinstance(chunk, dict):
                    text = chunk.get("page_content", "") or chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                elif isinstance(chunk, str):
                    text = chunk
                    meta = {}
                else:
                    continue

                if not text.strip(): continue

                # Clean and Prepare Metadata
                clean_meta = {k: str(v) for k, v in meta.items() if v is not None}
                clean_meta.update({
                    "source_file": jf,
                    "namespace": namespace,
                    "page_content": text
                })

                # Create Vector ID
                chunk_id = f"{namespace}_{jf}_{i}"
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedder.encode(text).tolist(),
                    "metadata": clean_meta
                })

                # Batch Upsert
                if len(vectors) >= BATCH_SIZE:
                    index.upsert(vectors=vectors, namespace=namespace)
                    total_vectors += len(vectors)
                    vectors = []

            # Final batch for this file
            if vectors:
                index.upsert(vectors=vectors, namespace=namespace)
                total_vectors += len(vectors)

            # Mark file as done in registry
            registry[registry_key] = current_hash
            registry_updated = True

    # Save Registry State
    if registry_updated:
        save_registry(registry)
        print("\n💾 Ingestion complete. Registry updated.")
    else:
        print("\n✅ System is fully synchronized.")

if __name__ == "__main__":
    # --- PATH SETUP ---
    # Current File: src/genai/retrivel/ingest.py
    # Target:       src/data/processed
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up 2 levels (to 'src'), then into 'data/processed'
    PROCESSED_DIR = os.path.abspath(os.path.join(base_dir, "../../data/processed"))

    # Create it if it doesn't exist (just in case)
    if not os.path.exists(PROCESSED_DIR):
        print(f"Creating missing directory: {PROCESSED_DIR}")
        os.makedirs(PROCESSED_DIR)

    ingest_processed_data(PROCESSED_DIR)