from huggingface import AutoTokenizer, AutoModel
from configs import CONFIG
from model_definitions.base_model import BaseModel

import torch
import torch.nn.functional as F



class ModernBertModel(BaseModel):
    name = 'bert'
    
    
    def __init__(self, gpu_number, device='cuda', padding=True, dimensions=768):
        super().__init__(gpu_number, device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = AutoModel.from_pretrained("answerdotai/ModernBERT-base").to(self.device)
        
        self.padding = padding
        self.dimensions = dimensions
        
    def embed(self, texts:list[str]):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=self.padding).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        
        
        normalized = F.normalize(mean_pooled, p=2, dim=1).cpu()

        if normalized.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedding output count mismatch: "
                f"received {len(texts)} inputs but produced {normalized.shape[0]} embeddings."
            )

        return normalized
    
    
    def forward(self, process_name, texts=None):
        if process_name == 'embed':
            return self.embed(texts)
        else:
            raise ValueError(f"Process {process_name} not supported for {self.name}")
        
    @classmethod
    def list_processes(cls):
        return ['embed']
    
    