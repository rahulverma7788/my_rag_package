from embeddings import EmbeddingModel
from transformers import pipeline

class QueryTransformer:
    def __init__(self, model_name="bert-base-uncased"):
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.paraphrase_model = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

    def transform(self, query: str):
        paraphrased_query = self.paraphrase_model(query, max_length=50, num_return_sequences=1)[0]['generated_text']
        query_embedding = self.embedding_model.embed(paraphrased_query)
        return paraphrased_query, query_embedding
