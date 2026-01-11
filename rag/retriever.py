"""
Retriever for RAG pipeline - supports both FAISS and Weaviate backends.
"""

import os
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Retriever:
    """Document retriever supporting both FAISS and Weaviate backends."""
    
    def __init__(self, model_name: str = None, use_reranker: bool = None, backend: str = None):
        """
        Initialize retriever.
        
        Args:
            model_name: Name of sentence transformer model
            use_reranker: Whether to use cross-encoder reranking
            backend: Vector store backend ('faiss' or 'weaviate')
        """
        if model_name is None:
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.model_name = model_name
        self.encoder = None
        
        # Determine backend
        if backend is None:
            backend = os.getenv('VECTOR_STORE', 'faiss').lower()
        self.backend = backend
        
        # Initialize appropriate vector store
        if self.backend == 'weaviate':
            from rag.weaviate_store import WeaviateVectorStore
            self.vector_store = WeaviateVectorStore()
        else:
            from rag.vector_store import VectorStore
            self.vector_store = VectorStore()
        
        # Reranker configuration
        if use_reranker is None:
            use_reranker = os.getenv('USE_RERANKER', 'false').lower() == 'true'
        self.use_reranker = use_reranker
        self.reranker = None
        
        self._load_encoder()
        if self.use_reranker:
            self._load_reranker()
        
        print(f"✓ Retriever initialized with {self.backend.upper()} backend")
    
    def _load_encoder(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
            print(f"Loaded encoder: {self.model_name}")
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    def _load_reranker(self):
        """Load cross-encoder reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            reranker_model = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.reranker = CrossEncoder(reranker_model)
            print(f"Loaded reranker: {reranker_model}")
        except ImportError:
            print("sentence-transformers not installed for reranking")
            self.use_reranker = False
    
    def load_vector_store(self, embeddings_path: str = "data/embeddings.pkl"):
        """Load vector store (only for FAISS backend)."""
        if self.backend == 'faiss':
            return self.vector_store.load_from_file(embeddings_path)
        else:
            # Weaviate is always "loaded" via connection
            return True
    
    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (final count after reranking)
            filters: Optional metadata filters (Weaviate only)
            
        Returns:
            List of (document_text, relevance_score, metadata) tuples
        """
        if self.encoder is None:
            print("Encoder not loaded")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Search based on backend
        if self.backend == 'weaviate':
            # Determine retrieval count based on reranking
            if self.use_reranker and self.reranker is not None:
                initial_k = int(os.getenv('INITIAL_RETRIEVAL_K', '20'))
                raw_results = self.vector_store.search(
                    query_embedding.tolist(), 
                    k=initial_k,
                    filters=filters
                )
                
                if raw_results:
                    import numpy as np
                    documents = [doc for doc, _, _ in raw_results]
                    metadata_list = [meta for _, _, meta in raw_results]
                    pairs = [(query, doc) for doc in documents]
                    raw_scores = self.reranker.predict(pairs)
                    
                    # Normalize scores
                    scores_array = np.array(raw_scores)
                    if scores_array.max() > scores_array.min():
                        normalized_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
                    else:
                        normalized_scores = np.ones_like(scores_array) * 0.5
                    
                    reranked = list(zip(documents, normalized_scores.tolist(), metadata_list))
                    reranked.sort(key=lambda x: x[1], reverse=True)
                    return reranked[:k]
            else:
                # No reranking
                results = self.vector_store.search(
                    query_embedding.tolist(), 
                    k=k,
                    filters=filters
                )
                return results
        else:
            # FAISS backend
            if self.use_reranker and self.reranker is not None:
                import numpy as np
                initial_k = int(os.getenv('INITIAL_RETRIEVAL_K', '20'))
                raw_results = self.vector_store.search(query_embedding, k=initial_k)
                
                if raw_results:
                    documents = [doc for doc, _ in raw_results]
                    pairs = [(query, doc) for doc in documents]
                    raw_scores = self.reranker.predict(pairs)
                    
                    # Normalize scores
                    scores_array = np.array(raw_scores)
                    if scores_array.max() > scores_array.min():
                        normalized_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
                    else:
                        normalized_scores = np.ones_like(scores_array) * 0.5
                    
                    reranked = list(zip(documents, normalized_scores.tolist(), [None] * len(documents)))
                    reranked.sort(key=lambda x: x[1], reverse=True)
                    return reranked[:k]
            else:
                results = self.vector_store.search(query_embedding, k=k)
                # Add None for metadata to match return signature
                return [(text, score, None) for text, score in results]
        
        return []
    
    def retrieve_with_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve documents and format as context string.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, k=k)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            if len(result) == 3:
                chunk, score, metadata = result
                meta_str = ""
                if metadata:
                    meta_str = f" [Source: {metadata.get('source', 'unknown')}, Page: {metadata.get('page', 'N/A')}]"
                context_parts.append(f"[Document {i}]{meta_str} (Relevance: {score:.3f})\n{chunk}\n")
            else:
                # Legacy format: (chunk, score)
                chunk, score = result
                context_parts.append(f"[Document {i}] (Relevance: {score:.3f})\n{chunk}\n")
        
        return "\n".join(context_parts)
    
    def close(self):
        """Close connections."""
        if self.backend == 'weaviate' and hasattr(self.vector_store, 'close'):
            self.vector_store.close()


if __name__ == "__main__":
    # Example usage
    retriever = Retriever()
    
    if retriever.load_vector_store():
        # Test query
        query = "What are common indicators of fraudulent transactions?"
        results = retriever.retrieve(query, k=3)
        
        print(f"Query: {query}\n")
        print("Retrieved documents:")
        for i, result in enumerate(results, 1):
            if len(result) == 3:
                chunk, score, metadata = result
                print(f"\n{i}. (Score: {score:.3f})")
                if metadata:
                    print(f"   Source: {metadata.get('source')}, Page: {metadata.get('page')}")
                print(chunk[:200] + "...")
            else:
                chunk, score = result
                print(f"\n{i}. (Score: {score:.3f})")
                print(chunk[:200] + "...")
    else:
        print("Vector store not available. Run ingestion pipeline first.")
    
    # Close connections
    retriever.close()
