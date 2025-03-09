import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

class SKLearnVectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_store = None
        self.data = []

    def load_data(self, file_path = r"C:\Users\zakar\OneDrive\Desktop\Orange_rag\Rag\data.jsonl"
):
        """Charge les données JSONL et les indexe."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                self.data.append(doc)

        texts = [doc["title"] + " " + doc["content"] for doc in self.data]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Entraînement du modèle de recherche
        self.vector_store = NearestNeighbors(n_neighbors=3, metric="cosine")
        self.vector_store.fit(embeddings)

    def search(self, query, top_k=3):
        """Recherche les documents les plus pertinents."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.vector_store.kneighbors(query_embedding, n_neighbors=top_k)
        
        results = []
        for i in indices[0]:
            results.append(self.data[i])
        
        return results
