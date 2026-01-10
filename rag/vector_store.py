"""
FAISS vector store for semantic search.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple


class VectorStore:
    """FAISS-based vector store for document retrieval."""
    
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (768 for all-mpnet-base-v2, 384 for all-MiniLM-L6-v2)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def build_index(self, embeddings: np.ndarray, chunks: List[str]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings (n_chunks, embedding_dim)
            chunks: List of text chunks
        """
        print(f"Building index with embeddings shape: {embeddings.shape}, chunks: {len(chunks)}")
        
        try:
            import faiss
        except ImportError:
            print("ERROR: faiss not installed. Run: pip install faiss-cpu")
            raise ImportError("faiss-cpu is required for vector search")
        
        # Validate embeddings shape
        if embeddings.size == 0:
            print("ERROR: Empty embeddings array")
            raise ValueError("Empty embeddings array")
        
        # Ensure embeddings is 2D
        if embeddings.ndim == 1:
            print("Reshaping 1D embeddings to 2D")
            embeddings = embeddings.reshape(1, -1)
        
        # Verify dimensions match
        if embeddings.shape[1] != self.embedding_dim:
            print(f"Updating embedding_dim from {self.embedding_dim} to {embeddings.shape[1]}")
            self.embedding_dim = embeddings.shape[1]
        
        if len(chunks) != embeddings.shape[0]:
            print(f"ERROR: Mismatch between chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]})")
            raise ValueError("Number of chunks must match number of embeddings")
        
        self.chunks = chunks
        self.embeddings = embeddings
        
        # Normalize embeddings for cosine similarity
        print("Normalizing embeddings...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        embeddings_normalized = embeddings / norms
        
        # Create FAISS index
        print(f"Creating FAISS index with dimension {self.embedding_dim}...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.index.add(embeddings_normalized.astype('float32'))
        
        print(f"✓ Successfully built FAISS index with {len(chunks)} vectors (dim={self.embedding_dim})")
    
    def load_from_file(self, embeddings_path: str = "data/embeddings.pkl"):
        """
        Load embeddings from file and build index.
        
        Args:
            embeddings_path: Path to saved embeddings
        """
        print(f"Loading embeddings from {embeddings_path}...")
        
        if not Path(embeddings_path).exists():
            print(f"ERROR: Embeddings file not found: {embeddings_path}")
            return False
        
        try:
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Loaded data keys: {data.keys()}")
            print(f"Embeddings type: {type(data['embeddings'])}, shape: {data['embeddings'].shape}")
            print(f"Number of chunks: {len(data['chunks'])}")
            
            self.build_index(data['embeddings'], data['chunks'])
            return True
        except Exception as e:
            print(f"ERROR loading embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if self.index is None:
            print("ERROR: Index not built. Call build_index() first.")
            return []
        
        # Ensure query embedding is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            print("Warning: Zero-norm query embedding")
            return []
        query_normalized = query_embedding / norm
        query_normalized = query_normalized.reshape(1, -1).astype('float32')
        
        # Search
        k = min(k, len(self.chunks))  # Don't search for more than available
        scores, indices = self.index.search(query_normalized, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):  # Validate index
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save_index(self, index_path: str = "data/faiss_index"):
        """
        Save FAISS index to disk.
        
        Args:
            index_path: Path to save index
        """
        try:
            import faiss
        except ImportError:
            print("faiss not installed.")
            return
        
        if self.index is None:
            print("No index to save")
            return
        
        faiss.write_index(self.index, index_path)
        
        # Save chunks separately
        with open(f"{index_path}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str = "data/faiss_index"):
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to saved index
        """
        try:
            import faiss
        except ImportError:
            print("faiss not installed.")
            return False
        
        if not Path(index_path).exists():
            print(f"Index file not found: {index_path}")
            return False
        
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        with open(f"{index_path}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Index loaded with {len(self.chunks)} chunks")
        return True


if __name__ == "__main__":
    # Example usage
    store = VectorStore()
    
    # Load and build index
    if store.load_from_file():
        store.save_index()
        print("Vector store ready!")
    else:
        print("Run load_docs.py first to create embeddings")
