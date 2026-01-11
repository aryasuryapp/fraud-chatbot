"""
Retriever for RAG pipeline - combines vector store with query encoding.
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv
from rag.vector_store import VectorStore

# Load environment variables
load_dotenv()


class Retriever:
    """Document retriever for RAG pipeline."""
    
    def __init__(self, model_name: str = None, use_reranker: bool = None):
        """
        Initialize retriever.
        
        Args:
            model_name: Name of sentence transformer model (defaults to EMBEDDING_MODEL env var)
            use_reranker: Whether to use cross-encoder reranking (defaults to USE_RERANKER env var)
        """
        if model_name is None:
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.model_name = model_name
        self.encoder = None
        self.vector_store = VectorStore()
        
        # Reranker configuration
        if use_reranker is None:
            use_reranker = os.getenv('USE_RERANKER', 'false').lower() == 'true'
        self.use_reranker = use_reranker
        self.reranker = None
        
        self._load_encoder()
        if self.use_reranker:
            self._load_reranker()
    
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
        """
        Load vector store from file.
        
        Args:
            embeddings_path: Path to embeddings file
        """
        return self.vector_store.load_from_file(embeddings_path)
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (final count after reranking)
            
        Returns:
            List of (document_text, relevance_score) tuples
        """
        if self.encoder is None:
            print("Encoder not loaded")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # If using reranker, retrieve more candidates
        if self.use_reranker and self.reranker is not None:
            # Retrieve more candidates for reranking
            initial_k = int(os.getenv('INITIAL_RETRIEVAL_K', '20'))
            results = self.vector_store.search(query_embedding, k=initial_k)
            
            # Rerank with cross-encoder
            if results:
                documents = [doc for doc, _ in results]
                # Create query-document pairs for cross-encoder
                pairs = [(query, doc) for doc in documents]
                scores = self.reranker.predict(pairs)
                
                # Combine documents with new scores and sort
                reranked = list(zip(documents, scores))
                reranked.sort(key=lambda x: x[1], reverse=True)
                
                # Return top k reranked results
                return reranked[:k]
        else:
            # Standard retrieval without reranking
            results = self.vector_store.search(query_embedding, k=k)
        
        return results
    
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
        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(f"[Document {i}] (Relevance: {score:.3f})\n{chunk}\n")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    # Example usage
    retriever = Retriever()
    
    if retriever.load_vector_store():
        # Test query
        query = "What are common indicators of fraudulent transactions?"
        results = retriever.retrieve(query, k=3)
        
        print(f"Query: {query}\n")
        print("Retrieved documents:")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n{i}. (Score: {score:.3f})")
            print(chunk[:200] + "...")
    else:
        print("Vector store not available. Run ingestion pipeline first.")
