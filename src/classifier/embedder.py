"""
PhoBERT embedding generator for Vietnamese text
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Union
import sys

from src.config import PHOBERT_MODEL, EMBEDDING_DIM, MAX_SEQ_LENGTH, BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhoBertEmbedder:
    """
    Generate embeddings using PhoBERT model for Vietnamese text
    """
    
    def __init__(self, model_name: str = PHOBERT_MODEL, batch_size: int = BATCH_SIZE):
        """
        Initialize PhoBERT embedder
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading PhoBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info(f"Using device: {self.device}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling of token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_texts(self, texts: Union[List[str], str], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for Vietnamese texts
        
        Args:
            texts: Single text or list of texts to embed
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
            
            # Move to device
            for key in encoded:
                encoded[key] = encoded[key].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded)
            
            # Mean pooling
            batch_embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
            
            # Normalize if requested
            if normalize:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        result = np.vstack(embeddings) if embeddings else np.array([])
        
        logger.info(f"Generated {len(texts)} embeddings of shape {result.shape}")
        return result
    
    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            normalize: Whether to normalize embedding
            
        Returns:
            1D numpy array of embedding
        """
        embeddings = self.embed_texts([text], normalize=normalize)
        return embeddings[0]


def get_embedder(model_name: str = PHOBERT_MODEL) -> PhoBertEmbedder:
    """Factory function to get embedder instance"""
    return PhoBertEmbedder(model_name)


if __name__ == "__main__":
    # Test embedder
    embedder = PhoBertEmbedder()
    
    test_texts = [
        "Làm cách nào để nâng cao hiệu suất?",
        "Tôi cần giúp đỡ về kỹ thuật",
        "Giá sản phẩm là bao nhiêu?"
    ]
    
    embeddings = embedder.embed_texts(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
