import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.ids = []
        self.texts = []
        self.matrix = None

    def add_passages(self, passages):
        self.ids.extend([p.passage_id for p in passages])
        self.texts.extend([p.text for p in passages])

        self.matrix = self.model.encode(
            self.texts,
            normalize_embeddings=True
        )

    def search(self, query, top_k=10):
        if not self.ids:
            return []

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = self.matrix @ query_vector
        ranked = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "passage_id": self.ids[index],
                "score": float(scores[index])
            }
            for index in ranked
        ]
