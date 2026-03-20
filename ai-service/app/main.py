from app.embeddings.embedder import embedder
from app.vector_store.faiss_store import FAISSStore
from app.generation.generator import generate_answer
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text

import os
import pickle
import hashlib
import numpy as np

store = FAISSStore(dim=384)

pdf_path = "/home/ritesh/Documents/Ritesh_Yadav_senior_software_developer.pdf"

CACHE_DIR = "cache"
MAX_CACHE_FILES = 5

os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_name(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return os.path.join(CACHE_DIR, f"{file_hash}.pkl")

def cleanup_cache():
    files = sorted(
        [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR)],
        key=os.path.getmtime
    )

    while len(files) > MAX_CACHE_FILES:
        os.remove(files[0])
        files.pop(0)

CACHE_FILE = get_cache_name(pdf_path)

# ✅ SINGLE cache block
if os.path.exists(CACHE_FILE):
    print("✅ Loading from cache...")
    with open(CACHE_FILE, "rb") as f:
        chunks, embeddings = pickle.load(f)
else:
    print("⚡ Processing PDF...")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((chunks, embeddings), f)

    cleanup_cache()

# store in FAISS
store.add(embeddings, chunks)

# query
query = "who is Ritesh yadav?"
query_embedding = embedder.encode(query).astype("float32")

# hybrid search
results = store.search(query_embedding, query, k=5)
retrieved_chunks = [r[0] for r in results]

# ✅ optimized reranking
retrieved_set = set(retrieved_chunks)

scored = []
for chunk, emb in zip(chunks, embeddings):
    if chunk in retrieved_set:
        score = np.dot(query_embedding, emb)
        scored.append((chunk, score))

scored.sort(key=lambda x: x[1], reverse=True)
retrieved_chunks = [s[0] for s in scored[:3]]

# LLM
answer = generate_answer(query, retrieved_chunks)

print("Query:", query)
print("Answer:", answer)