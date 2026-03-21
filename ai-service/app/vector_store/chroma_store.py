import chromadb

class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("documents")

    def count(self):
        return self.collection.count()

    def add(self, embeddings, texts):
        ids = [str(i) for i in range(len(texts))]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=ids
        )

    def search(self, query_embedding, k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        return results["documents"][0]