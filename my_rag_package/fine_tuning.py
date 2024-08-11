import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from embeddings import EmbeddingModel
from reranker import Reranker
from retrieval import Retriever
from hybrid_search import HybridSearch
from query_transformations import QueryTransformer

class FineTuningDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FineTuner:
    def __init__(self, model_name="bert-base-uncased", learning_rate=5e-5):
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.optimizer = AdamW(self.embedding_model.model.parameters(), lr=learning_rate)
    
    def fine_tune(self, data, epochs=3, batch_size=8):
        dataset = FineTuningDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.embedding_model.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                queries, documents = zip(*batch)
                query_embeddings = self.embedding_model.embed_batch(queries)
                document_embeddings = self.embedding_model.embed_batch(documents)

                loss = self._compute_loss(query_embeddings, document_embeddings)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    def _compute_loss(self, query_embeddings, document_embeddings):
        cosine_sim = torch.nn.functional.cosine_similarity(query_embeddings, document_embeddings)
        loss = 1 - cosine_sim.mean()
        return loss

    def save_model(self, output_path):
        self.embedding_model.model.save_pretrained(output_path)

    def load_model(self, input_path):
        self.embedding_model.model = AutoModel.from_pretrained(input_path)
