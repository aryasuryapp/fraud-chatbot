"""
Debug script to check if specific PDF content is embedded correctly.
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def check_pdf_content():
    """Check if specific content from PDF is in embeddings."""
    
    # Load embeddings
    embeddings_path = "data/embeddings.pkl"
    
    if not Path(embeddings_path).exists():
        print(f"❌ {embeddings_path} not found. Run: python ingestion/load_docs.py")
        return
    
    print("📂 Loading embeddings...")
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    embeddings = data['embeddings']
    
    print(f"✅ Loaded {len(chunks)} chunks\n")
    
    # Search for table content
    search_terms = [
        "Lost or stolen card",
        "48%",
        "Identity theft",
        "Skimming",
        "Counterfeit card",
        "Mail intercept fraud",
        "Table 1",
        "Methods of Credit Card Fraud"
    ]
    
    print("🔍 Searching for table content in chunks...\n")
    
    found_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        matches = [term for term in search_terms if term.lower() in chunk_lower]
        
        if matches:
            found_chunks.append((i, chunk, matches))
    
    if found_chunks:
        print(f"✅ Found {len(found_chunks)} chunks containing table content:\n")
        for i, chunk, matches in found_chunks:
            print(f"{'='*70}")
            print(f"Chunk #{i} - Matches: {', '.join(matches)}")
            print(f"{'='*70}")
            print(chunk[:500])
            print(f"\n... (Length: {len(chunk)} chars)\n")
    else:
        print("❌ Table content NOT found in any chunks!")
        print("\n🔍 Possible issues:")
        print("1. PDF text extraction failed (tables may not extract well)")
        print("2. Chunks are too small and split the table")
        print("3. Text cleaning removed important content")
        
        # Show sample chunks
        print("\n📋 Sample chunks from your embeddings:")
        for i in range(min(3, len(chunks))):
            print(f"\n--- Chunk {i} ---")
            print(chunks[i][:300] + "...")
    
    # Now test semantic search
    print("\n\n🔎 Testing semantic search with query...")
    query = "What are the methods of credit card fraud?"
    
    model = SentenceTransformer('all-mpnet-base-v2')
    query_embedding = model.encode([query])[0]
    
    # Calculate similarity scores
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top 5 results
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    print(f"\nQuery: '{query}'")
    print(f"\nTop 5 most similar chunks:\n")
    
    for rank, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        print(f"{rank}. Score: {score:.3f} (Chunk #{idx})")
        print(f"   {chunks[idx][:200]}...")
        print()
    
    # Check if table content is in top results
    table_in_top5 = any(idx in [fc[0] for fc in found_chunks] for idx in top_indices)
    
    if found_chunks and not table_in_top5:
        print("⚠️  WARNING: Table content exists but NOT in top 5 results!")
        print("   This means semantic similarity is too low.")
        print("\n💡 Solutions:")
        print("   1. Use reranking (cross-encoder) - already implemented!")
        print("   2. Add table as separate chunk with better formatting")
        print("   3. Lower RELEVANCE_THRESHOLD in .env")
    elif not found_chunks:
        print("❌ CRITICAL: Table content was never extracted from PDF!")
        print("\n💡 Solutions:")
        print("   1. Check if PDF is text-based (not scanned image)")
        print("   2. Increase CHUNK_SIZE in .env to keep tables intact")
        print("   3. Manually add table data to a separate text file")

if __name__ == "__main__":
    try:
        check_pdf_content()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()