from typing import Literal

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MistralEmbedder:
    
    def __init__(self, 
                 model_id: str='Salesforce/SFR-Embedding-Mistral', 
                 cache_dir: str='./models',
                 device: Literal['auto', 'cpu']='auto'):
        
        self.model_id = model_id
        self.model = self.get_model(cache_dir, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        
    def to(self, device: Literal['cpu', 'auto']):
        self.model = self.get_model(device)
        return self
    
    def embed(self, input_texts, max_length: int=1536) -> np.ndarray:
        batch_dict = self.tokenizer(input_texts, max_length=max_length, 
                                    padding='max_length', truncation=True, return_tensors="pt")
        outputs = self.model(**batch_dict)
        embeddings = self.pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = embeddings.to(dtype=torch.float)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = np.array([embed.detach().numpy().tolist() for embed in embeddings])
        return embeddings
    
    def pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_model(self, cache_dir, device: Literal['cpu', 'auto']):
        if device == 'auto':
            model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                cache_dir=cache_dir, 
                device_map='auto', 
                torch_dtype=torch.float16)
        elif device == 'cpu':
            model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                cache_dir=cache_dir)
        else:
            raise ValueError("Device must be 'auto' or 'cpu'.")
        return model
    
    def __repr__(self):
        return f"MistralEmbedder(model_id={self.model_id})"