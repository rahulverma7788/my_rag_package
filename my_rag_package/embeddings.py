from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)        
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)        
        return embeddings
    def batch_embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)        
        with torch.no_grad():
            outputs = self.model(**inputs)        
        last_hidden_state = outputs.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)        
        return embeddings
    def embed_sentence(self, text, layer=-1):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)        
        with torch.no_grad():
            outputs = self.model(**inputs)        
        hidden_states = outputs.hidden_states[layer] if 'hidden_states' in outputs else outputs.last_hidden_state        
        return hidden_states
    def save_embeddings(self, embeddings, file_path):
        torch.save(embeddings, file_path)
    def load_embeddings(self, file_path):
        return torch.load(file_path)
