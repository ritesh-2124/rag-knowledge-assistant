import faiss
import numpy as np
import os
import pickle

class FAISSStore:
    def __init__(self, dim, index_path="faiss.index", meta_path="faiss_meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path):
            print("✅ Loading FAISS index...")
            self.index = faiss.read_index(index_path)

            with open(meta_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            print("⚡ Creating new FAISS index...")
            self.index = faiss.IndexFlatIP(dim)
            self.texts = []

    def add(self, embeddings, texts):
        vectors = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.texts.extend(texts)

        # ✅ Save to disk
        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.texts, f)

    def search(self, query_embedding, query_text, k=5):
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k * 2)

        results = []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                text = self.texts[idx]

                keyword_score = sum(
                    1 for word in query_text.lower().split()
                    if word in text.lower()
                )

                final_score = float(distances[0][i]) + 0.1 * keyword_score
                results.append((text, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]