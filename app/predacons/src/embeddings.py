from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from typing import List


class PredaconsEmbedding():
    """
    A class to generate embeddings for sentences using a pre-trained transformer model.

    Attributes:
        model (AutoModel): The pre-trained transformer model.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained model.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"):
        """
        Initializes the PredaconsEmbedding with a specified pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        """
        Applies mean pooling to the model output, taking the attention mask into account.

        Args:
            model_output: The output from the transformer model.
            attention_mask: The attention mask from the tokenizer.

        Returns:
            torch.Tensor: The mean pooled embeddings.
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences):
        """
        Generates embeddings for the given sentences.

        Args:
            sentences (str or List[str]): The sentence(s) to generate embeddings for.

        Returns:
            List[float] or List[List[float]]: The embeddings for the sentence(s).
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Check if sentences is a list
        if isinstance(sentences, list):
            return sentence_embeddings.tolist()
        else:
            return sentence_embeddings.tolist()[0]
    
    def embed_query(self, query):
        """
        Generates embeddings for a single query.

        Args:
            query (str): The query to generate embeddings for.

        Returns:
            List[float]: The embeddings for the query.
        """
        return self.get_embedding(query)

    def embed_documents(self, texts: List[str], chunk_size: int | None = None) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): The documents to generate embeddings for.
            chunk_size (int, optional): The chunk size for processing documents. Defaults to 300.

        Returns:
            List[List[float]]: The embeddings for the documents.
        """
        chunk_size = chunk_size or 300
        return self.get_embedding(texts)