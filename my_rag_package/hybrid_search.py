from embeddings import EmbeddingModel
from retrieval import Retriever
from reranker import Reranker
from query_transformations import QueryTransformer

class HybridSearch:
    def __init__(self, documents: list, model_name="bert-base-uncased"):
        self.retriever = Retriever(documents, model_name)
        self.reranker = Reranker(model_name=model_name)
        self.query_transformer = QueryTransformer(model_name=model_name)
    
    def search(self, query: str, top_k: int = 10):
        transformed_query = self.query_transformer.transform(query)
        # Retrieve relevant documents
        retrieved_docs, retrieved_doc_scores = self.retriever.retrieve(transformed_query, top_k=top_k)
        # Rerank the retrieved documents
        reranked_docs = self.reranker.rerank(transformed_query, retrieved_docs, retrieved_doc_scores)
        return reranked_docs
