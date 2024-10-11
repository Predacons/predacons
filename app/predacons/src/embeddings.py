from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from typing import List


class PredaconsEmbedding():
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self,sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # check if sentences is list
        if isinstance(sentences, list):
            return sentence_embeddings.tolist()
        else:
            return sentence_embeddings.tolist()[0]
    
    def embed_query(self, query):
        return self.get_embedding(query)

    def embed_documents(
        self, texts: List[str], chunk_size: int | None = None
    ) -> List[List[float]]:
        chunk_size = chunk_size or 300
        return self.get_embedding(texts)