from flask import Flask, request, jsonify
from embeddings import EmbeddingModel
from retrieval import Retriever
from reranker import Reranker
from query_transformations import QueryTransformer
from hybrid_search import HybridSearch

def create_app():
    app = Flask(__name__)

    # Sample documents or load them from a data source
    documents = [
        "Sample document 1",
        "Sample document 2"
        # Add more documents as needed
    ]
    
    # Initialize components
    embedding_model = EmbeddingModel(model_name="bert-base-uncased")
    retriever = Retriever(documents=documents, model_name="bert-base-uncased")
    reranker = Reranker(model_name="bert-base-uncased")
    query_transformer = QueryTransformer(model_name="bert-base-uncased")
    hybrid_search = HybridSearch(documents=documents, model_name="bert-base-uncased")

    @app.route('/search', methods=['POST'])
    def search():
        try:
            data = request.json
            query = data.get('query')
            if not query:
                return jsonify({"error": "Query parameter is missing"}), 400

            # Transform the query
            transformed_query, query_embedding = query_transformer.transform(query)

            # Retrieve relevant documents
            retrieved_docs, retrieved_doc_scores = retriever.retrieve(transformed_query)
            if not retrieved_docs:
                return jsonify({"error": "No documents retrieved"}), 404

            # Rerank the retrieved documents
            reranked_docs, reranked_scores = reranker.rerank(transformed_query, retrieved_docs, retrieved_doc_scores)
            
            # Format and return the results
            results = {
                "query": query,
                "transformed_query": transformed_query,
                "retrieved_docs": retrieved_docs,
                "reranked_docs": reranked_docs,
                "scores": reranked_scores
            }
            return jsonify(results)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
