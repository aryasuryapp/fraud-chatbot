"""
Weaviate vector store for semantic search with metadata support.
"""

import os
import uuid
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()


class WeaviateVectorStore:
    """Weaviate-based vector store with metadata support."""
    
    def __init__(self):
        """Initialize Weaviate client."""
        self.client = None
        self.collection_name = os.getenv('WEAVIATE_COLLECTION', 'FraudDocuments')
        self._connect()
    
    def _connect(self):
        """Connect to Weaviate instance."""
        try:
            import weaviate
            from weaviate.classes.init import Auth
        except ImportError:
            print("ERROR: weaviate-client not installed. Run: pip install weaviate-client")
            raise ImportError("weaviate-client is required")
        
        # Get Weaviate URL from environment
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
        
        try:
            if weaviate_api_key:
                # Cloud instance with authentication
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=weaviate_url,
                    auth_credentials=Auth.api_key(weaviate_api_key),
                )
            else:
                # Local instance without authentication
                host = weaviate_url.replace('http://', '').replace('https://', '').split(':')[0]
                port = int(weaviate_url.split(':')[-1]) if ':' in weaviate_url.split('/')[-1] else 8080
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                )
            
            print(f"✓ Connected to Weaviate at {weaviate_url}")
            print(f"  Collection: {self.collection_name}")
        except Exception as e:
            print(f"ERROR connecting to Weaviate: {e}")
            print("\nSetup options:")
            print("1. Local Docker: docker-compose up -d")
            print("2. Cloud: Set WEAVIATE_URL and WEAVIATE_API_KEY in .env")
            raise
    
    def create_schema(self, embedding_dim: int = 384):
        """
        Create Weaviate schema for documents.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        try:
            from weaviate.classes.config import Configure, Property, DataType, VectorDistances
            
            # Check if collection already exists
            if self.client.collections.exists(self.collection_name):
                print(f"Collection '{self.collection_name}' already exists. Skipping schema creation.")
                return
            
            # Create collection with properties
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=None,  # We'll provide our own vectors
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="chunk_id", data_type=DataType.INT),
                    Property(name="created_at", data_type=DataType.TEXT),
                ]
            )
            print(f"✓ Created collection '{self.collection_name}' with schema")
            
        except Exception as e:
            print(f"ERROR creating schema: {e}")
            raise
    
    def add_documents(
        self, 
        chunks: List[str], 
        embeddings: List[List[float]],
        metadata_list: List[Dict]
    ):
        """
        Add documents with embeddings and metadata to Weaviate.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata_list: List of metadata dicts with keys: source, page, chunk_id, created_at
        """
        if len(chunks) != len(embeddings) != len(metadata_list):
            raise ValueError("chunks, embeddings, and metadata_list must have same length")
        
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Batch insert for efficiency
            with collection.batch.dynamic() as batch:
                for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadata_list)):
                    batch.add_object(
                        properties={
                            "text": chunk,
                            "source": metadata.get("source", "unknown"),
                            "page": metadata.get("page", 0),
                            "chunk_id": metadata.get("chunk_id", i),
                            "created_at": metadata.get("created_at", ""),
                        },
                        vector=embedding
                    )
            
            print(f"✓ Added {len(chunks)} documents to Weaviate")
            
        except Exception as e:
            print(f"ERROR adding documents: {e}")
            raise
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters (e.g., {"source": "fraud_guide.pdf"})
            
        Returns:
            List of (text, score, metadata) tuples
        """
        try:
            collection = self.client.collections.get(self.collection_name)
            
            # Build filter if provided
            weaviate_filter = None
            if filters:
                from weaviate.classes.query import Filter
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(Filter.by_property(key).equal(value))
                
                # Combine filters with AND
                if len(filter_conditions) == 1:
                    weaviate_filter = filter_conditions[0]
                else:
                    weaviate_filter = Filter.all_of(filter_conditions)
            
            # Perform vector search
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=k,
                return_metadata=['distance'],
                filters=weaviate_filter
            )
            
            results = []
            for obj in response.objects:
                # Convert distance to similarity score (cosine: 1 - distance)
                score = 1.0 - obj.metadata.distance if obj.metadata.distance else 0.0
                
                metadata = {
                    "source": obj.properties.get("source"),
                    "page": obj.properties.get("page"),
                    "chunk_id": obj.properties.get("chunk_id"),
                    "created_at": obj.properties.get("created_at"),
                }
                
                results.append((obj.properties["text"], score, metadata))
            
            return results
            
        except Exception as e:
            print(f"ERROR searching: {e}")
            return []
    
    def delete_collection(self):
        """Delete the collection (useful for resetting)."""
        try:
            if self.client.collections.exists(self.collection_name):
                self.client.collections.delete(self.collection_name)
                print(f"✓ Deleted collection '{self.collection_name}'")
        except Exception as e:
            print(f"ERROR deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        try:
            collection = self.client.collections.get(self.collection_name)
            # Get total object count
            aggregate = collection.aggregate.over_all(total_count=True)
            
            return {
                "total_documents": aggregate.total_count,
                "collection_name": self.collection_name,
            }
        except Exception as e:
            print(f"ERROR getting stats: {e}")
            return {}
    
    def close(self):
        """Close Weaviate connection."""
        if self.client:
            self.client.close()
            print("✓ Closed Weaviate connection")


if __name__ == "__main__":
    # Example usage
    store = WeaviateVectorStore()
    
    # Get stats
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    store.close()
