from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en", device="cpu")

    def encode(self, text):
        return self.model.encode(text)
        

# create object
embedder = Embedder()