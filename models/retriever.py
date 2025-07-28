from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def build_index(self, chunks):
        self.text_chunks = chunks
        embeddings = self.model.encode([chunk[2] for chunk in chunks], convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def query(self, question: str, top_k: int = 5):
        query_embedding = self.model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in I[0]]
