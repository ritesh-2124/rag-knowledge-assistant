from app.embeddings.embedder import embedder
from app.generation.generator import generate_answer
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.vector_store.chroma_store import ChromaStore

store = ChromaStore()

pdf_path = "/home/ritesh/Documents/doc/upprpb.in_#_pscexamservice_candidate-View-Application_id=1b5353ac-5641-4412-a4b9-9256f61bb247&version=V1.pdf"

# ✅ Reset and add new document
print("⚡ Resetting DB and adding new document...")
store.client.delete_collection("documents")
store.collection = store.client.get_or_create_collection("documents")

text = load_pdf(pdf_path)
chunks = chunk_text(text)
embeddings = embedder.encode(chunks)

store.add(embeddings, chunks)

# ✅ Query
query = "What is the Address of the Ritesh in this document?"
query_embedding = embedder.encode(query).astype("float32")

# ✅ Hybrid retrieval: vector search + keyword fallback
retrieved_chunks = store.search(query_embedding, k=8)

# ✅ Keyword fallback — find chunks containing key terms related to the query
query_words = ["name", "applicant", "full name", "dob", "date of birth", "personal"]
for chunk in chunks:
    chunk_lower = chunk.lower()
    if any(word in chunk_lower for word in query_words) and chunk not in retrieved_chunks:
        retrieved_chunks.append(chunk)

# Deduplicate and limit
seen = set()
unique_chunks = []
for c in retrieved_chunks:
    if c not in seen:
        seen.add(c)
        unique_chunks.append(c)
retrieved_chunks = unique_chunks[:12]

print(f"\n--- {len(retrieved_chunks)} chunks sent to LLM ---")
# for i, c in enumerate(retrieved_chunks):
#     print(f"\n[Chunk {i+1}]")
#     print(c[:200], "...")

# ✅ Generate answer
answer = generate_answer(query, retrieved_chunks)

print("\n" + "="*60)
print("Query:", query)
print("="*60)
print("Answer:", answer)