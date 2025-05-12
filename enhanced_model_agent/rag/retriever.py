import os
import random
import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

class Retriever:
    def __init__(self, csv_path="../data/embeddings.csv", model_name="BAAI/bge-large-en-v1.5"):
        self.embed_model = SentenceTransformer(model_name)
        self.knowledge_df = pd.read_csv(csv_path)
        self.knowledge_embeddings = np.array(
            self.knowledge_df['embedding'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)).tolist(),
            dtype=np.float32
        )
        faiss.normalize_L2(self.knowledge_embeddings)
        self.knowledge_texts = self.knowledge_df['sentence_chunk'].tolist()
        self.knowledge_pages = self.knowledge_df['page_number'].tolist() if 'page_number' in self.knowledge_df else [None]*len(self.knowledge_texts)
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.knowledge_embeddings.shape[1])
        self.index.add(self.knowledge_embeddings)
        # Prepare cross-encoder for reranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def retrieve(self, query: str, top_k=3) -> list:
        query_embedding = self.embed_query(query).reshape(1, -1)
        print("Knowledge embeddings shape:", self.knowledge_embeddings.shape)
        print("Query embedding shape:", query_embedding.shape)
        D, I = self.index.search(query_embedding, 20)  # Retrieve top-20
        candidates = [
            (self.knowledge_texts[i], self.knowledge_pages[i], float(D[0][idx]))
            for idx, i in enumerate(I[0])
        ]
        # Cross-encoder rerank
        pairs = [(query, chunk) for chunk, _, _ in candidates]
        scores = self.cross_encoder.predict(pairs)
        # Get indices of top-3 by cross-encoder score
        top3_indices = np.argsort(scores)[-top_k:][::-1]
        return [(candidates[i][0], candidates[i][1], float(scores[i])) for i in top3_indices]
