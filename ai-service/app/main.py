from app.embeddings.embedder import embedder
from app.vector_store.faiss_store import FAISSStore

# Initialize store
store = FAISSStore(dim=384)

# Sample documents
documents = [
    "Healthy food includes fruits and vegetables",
    "I go to gym daily",
    "Python is a great programming language",
    "Artificial Intelligence is the future",
    "I love eating pizza and burgers"
]

# Convert docs → embeddings
doc_embeddings = embedder.encode(documents)

# Store in FAISS
store.add(doc_embeddings, documents)

# Query
query = "What are healthy foods?"
query_embedding = embedder.encode(query)

# Search similar docs
results = store.search(query_embedding)

print("Query:", query)
print("Results:", results)