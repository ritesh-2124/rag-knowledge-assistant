import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, embeddings, texts):
        vectors = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding, query_text, k=5):
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k * 2)

        results = []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                text = self.texts[idx]

                # 🔥 keyword score
                keyword_score = sum(
                    1 for word in query_text.lower().split()
                    if word in text.lower()
                )

                # 🔥 combined score
                final_score = float(distances[0][i]) + 0.1 * keyword_score

                results.append((text, final_score))

        # 🔥 sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]