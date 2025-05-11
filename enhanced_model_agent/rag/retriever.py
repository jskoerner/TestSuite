import numpy as np
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, csv_path="../data/embeddings.csv", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.knowledge_df = pd.read_csv(csv_path)
        self.knowledge_embeddings = np.array(
            self.knowledge_df['embedding'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)).tolist()
        )
        self.knowledge_texts = self.knowledge_df['sentence_chunk'].tolist()

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode([query])[0]

    def retrieve(self, query: str, top_k=1) -> list:
        query_embedding = self.embed_query(query)
        similarities = np.dot(self.knowledge_embeddings, query_embedding) / (
            np.linalg.norm(self.knowledge_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.knowledge_texts[i], similarities[i]) for i in top_indices]
