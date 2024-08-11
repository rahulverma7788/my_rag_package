import numpy as np
from embeddings import EmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity

class Reranker:

    def __init__(self, model_name="bert-base-uncased"):
        self.embedding_model = EmbeddingModel(model_name=model_name)


    def rerank(self, query: str, retrieved_documents: list, retrieved_embeddings: np.ndarray, top_k: int = 5):
        query_embedding = self.embedding_model.embed(query).numpy()
        similarities = cosine_similarity(query_embedding.reshape(1, -1), retrieved_embeddings).flatten()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return [retrieved_documents[i] for i in top_k_indices], [similarities[i] for i in top_k_indices]