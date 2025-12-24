"""Embedding generation using Sentence Transformers."""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModel:
    """Wrapper for Sentence Transformer embedding model."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model.
                       Defaults to value from settings.
        """
        self.model_name = model_name or settings.embedding_model
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use.
        
        Returns:
            Loaded SentenceTransformer model
        """
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension size
        """
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress_bar: bool = False,
               normalize: bool = True) -> np.ndarray:
        """Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for processing multiple texts
            show_progress_bar: Whether to show progress bar
            normalize: Whether to normalize embeddings to unit length
        
        Returns:
            NumPy array of embeddings. Shape: (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize
        )
        
        # Ensure 2D array even for single text
        if isinstance(texts, str):
            embeddings = embeddings.reshape(1, -1)
        
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text.
        
        Convenience method for encoding single texts without batching.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize embedding to unit length
        
        Returns:
            1D NumPy array of embedding
        """
        embedding = self.encode(text, normalize=normalize)
        return embedding[0]  # Return 1D array
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize if not already normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0:
            embedding1 = embedding1 / norm1
        if norm2 > 0:
            embedding2 = embedding2 / norm2
        
        # Compute dot product (cosine similarity for normalized vectors)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_most_similar(self,
                         query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> List[tuple]:
        """Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector (1D)
            candidate_embeddings: Array of candidate embeddings (2D)
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        # Ensure query is 2D for consistent operations
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities with all candidates
        similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return list of (index, score) tuples
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results


# Global instance for reuse
_embedding_model_instance = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the global embedding model instance.
    
    Returns:
        Shared EmbeddingModel instance
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = EmbeddingModel()
    return _embedding_model_instance
