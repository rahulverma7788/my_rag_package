import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import EmbeddingModel

class Retriever:
    def __init__(self, documents: list, model_name="bert-base-uncased"):
        self.documents = documents
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.document_embeddings = self._embed_documents(documents)

    def _embed_documents(self, documents: list):
        embeddings = []
        for doc in documents:
            embedding = self.embedding_model.embed(doc).numpy()
            embeddings.append(embedding)
        return np.vstack(embeddings)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.embed(query).numpy()
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
