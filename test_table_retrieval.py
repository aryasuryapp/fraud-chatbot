"""
Test if reranker can find the fraud methods table.
"""

from rag.retriever import Retriever

def test_table_retrieval():
    """Test retrieval with and without reranker for table query."""
    
    query = "What are the methods of credit card fraud and their percentages?"
    
    print("="*80)
    print(f"Query: {query}")
    print("="*80)
    
    # Test WITHOUT reranker
    print("\n[1] WITHOUT RERANKER (Baseline - Bi-encoder only):")
    print("-"*80)
    retriever_baseline = Retriever(use_reranker=False)
    if not retriever_baseline.load_vector_store():
        print("Error: Could not load vector store")
        return
    
    results_baseline = retriever_baseline.retrieve(query, k=5)
    
    for i, (doc, score) in enumerate(results_baseline, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   {doc[:250]}...")
        
        # Check if this chunk has the table
        if "lost or stolen card" in doc.lower() and "48%" in doc:
            print("   ✅ THIS IS THE TABLE CHUNK!")
    
    # Test WITH reranker
    print("\n\n[2] WITH RERANKER (Cross-encoder):")
    print("-"*80)
    retriever_reranked = Retriever(use_reranker=True)
    retriever_reranked.load_vector_store()
    
    results_reranked = retriever_reranked.retrieve(query, k=5)
    
    table_found = False
    for i, (doc, score) in enumerate(results_reranked, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   {doc[:250]}...")
        
        # Check if this chunk has the table
        if "lost or stolen card" in doc.lower() and "48%" in doc:
            print("   ✅ THIS IS THE TABLE CHUNK!")
            table_found = True
    
    print("\n" + "="*80)
    if table_found:
        print("✅ SUCCESS: Reranker found the table chunk!")
    else:
        print("❌ FAIL: Table still not in top 5 even with reranking")
    print("="*80)

if __name__ == "__main__":
    test_table_retrieval()
