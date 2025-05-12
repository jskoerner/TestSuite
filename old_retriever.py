import numpy as np, ast, pandas as pd
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self,
                 csv_path="../data/embeddings.csv",  # (stronger embedder optional)
                 model_name="BAAI/bge-large-en-v1.5"):
        self.embed_model = SentenceTransformer(model_name)
        df = pd.read_csv(csv_path)
        self.knowledge_embeddings = np.array(
            df["embedding"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)).tolist(),
            dtype=np.float32
        )
        self.knowledge_texts = df["sentence_chunk"].tolist()  # <- already unit-length

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            dtype=np.float32
        )[0]

    def retrieve(self, query: str, top_k: int = 3):  # grab a few, not just 1
        q_vec = self.embed_query(query)
        # cosine = dot because both sides are unit vectors
        sims = np.dot(self.knowledge_embeddings, q_vec)
        idx = np.argsort(sims)[-top_k:][::-1]
        return [(self.knowledge_texts[i], sims[i]) for i in idx] 