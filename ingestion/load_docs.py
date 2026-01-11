"""
Load PDF documents, chunk them, and create embeddings for RAG.
Supports both FAISS (legacy) and Weaviate (with metadata) backends.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from datetime import datetime
from dotenv import load_dotenv


def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers/headers
    text = re.sub(r'Page \d+', '', text)
    # Normalize unicode
    text = text.encode('ascii', 'ignore').decode()
    return text.strip()


def load_pdf_documents(pdf_dir: str) -> List[str]:
    """
    Load PDF documents from directory (legacy - no metadata).
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of document texts
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        return []
    
    documents = []
    pdf_path = Path(pdf_dir)
    
    if not pdf_path.exists():
        print(f"Directory {pdf_dir} not found")
        return documents
    
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"Loading {pdf_file.name}...")
        reader = PdfReader(str(pdf_file))
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        text = clean_text(text)
        documents.append(text)
        print(f"  Extracted {len(text)} characters")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def load_pdf_documents_with_metadata(pdf_dir: str) -> List[Tuple[str, Dict]]:
    """
    Load PDF documents from directory with metadata tracking (page-by-page).
    
    NOTE: This creates smaller, fragmented chunks with lower relevance scores.
    Use load_pdf_documents_with_smart_metadata() for better results.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of (page_text, metadata) tuples
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        return []
    
    documents = []
    pdf_path = Path(pdf_dir)
    
    if not pdf_path.exists():
        print(f"Directory {pdf_dir} not found")
        return documents
    
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"Loading {pdf_file.name}...")
        reader = PdfReader(str(pdf_file))
        
        # Extract each page with metadata
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            text = clean_text(text)
            
            if text:  # Only add non-empty pages
                metadata = {
                    "source": pdf_file.name,
                    "page": page_num,
                    "total_pages": len(reader.pages),
                }
                documents.append((text, metadata))
        
        print(f"  Extracted {len(reader.pages)} pages")
    
    print(f"Total pages loaded: {len(documents)}")
    return documents


def load_pdf_documents_with_smart_metadata(pdf_dir: str) -> Tuple[List[str], List[Dict]]:
    """
    Load PDFs as full documents but track metadata for page estimation.
    
    This approach:
    - Chunks the FULL document (like FAISS) for better semantic coherence
    - Tracks which page(s) each chunk comes from for citations
    - Best of both worlds: high scores + metadata
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        Tuple of (documents, metadata_list) where each document has associated metadata
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        return [], []
    
    documents = []
    doc_metadata = []
    pdf_path = Path(pdf_dir)
    
    if not pdf_path.exists():
        print(f"Directory {pdf_dir} not found")
        return documents, doc_metadata
    
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"Loading {pdf_file.name}...")
        reader = PdfReader(str(pdf_file))
        
        # Concatenate all pages (like FAISS does)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        full_text = clean_text(full_text)
        documents.append(full_text)
        
        # Store document-level metadata
        metadata = {
            "source": pdf_file.name,
            "total_pages": len(reader.pages),
        }
        doc_metadata.append(metadata)
        
        print(f"  Extracted {len(full_text)} characters from {len(reader.pages)} pages")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents, doc_metadata


def chunk_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split documents into smaller chunks (legacy - no metadata).
    
    Uses RecursiveCharacterTextSplitter with a hierarchy of semantic separators
    to create more meaningful chunks that respect document structure:
    - Paragraph breaks (\n\n)
    - Line breaks (\n)
    - Sentence endings (. ! ?)
    - Clauses (; ,)
    - Words and characters as fallback
    
    Args:
        documents: List of document texts
        chunk_size: Size of each chunk in characters (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        
    Returns:
        List of text chunks
    """
    # Try multiple import paths for different langchain versions
    RecursiveCharacterTextSplitter = None
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            print("ERROR: langchain text splitter not found.")
            print("Install with: pip install langchain-text-splitters")
            return []
    
    # Load environment variables for configurable parameters
    load_dotenv()
    chunk_size = int(os.getenv('CHUNK_SIZE', chunk_size))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', chunk_overlap))
    
    # Semantic separators in priority order:
    # 1. Paragraph breaks (strongest semantic boundary)
    # 2. Line breaks
    # 3. Sentence endings (period, exclamation, question)
    # 4. Clause separators (semicolon, comma)
    # 5. Word boundaries (space)
    # 6. Character-level (fallback)
    separators = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "! ",    # Exclamations
        "? ",    # Questions
        "; ",    # Semicolons (clauses)
        ", ",    # Commas (lists, clauses)
        " ",     # Word boundaries
        ""       # Character-level fallback
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc)
        chunks.extend(doc_chunks)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"Using semantic separators for better context preservation")
    return chunks


def chunk_documents_with_metadata(
    documents: List[Tuple[str, Dict]], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Tuple[str, Dict]]:
    """
    Split documents into chunks while preserving metadata (page-by-page).
    
    NOTE: This creates smaller, fragmented chunks with lower relevance scores.
    Use chunk_documents_with_smart_metadata() for better results.
    
    Args:
        documents: List of (page_text, metadata) tuples
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of (chunk_text, metadata) tuples
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            print("ERROR: langchain text splitter not found.")
            print("Install with: pip install langchain-text-splitters")
            return []
    
    # Load environment variables for configurable parameters
    load_dotenv()
    chunk_size = int(os.getenv('CHUNK_SIZE', chunk_size))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', chunk_overlap))
    
    separators = [
        "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )
    
    chunks_with_metadata = []
    chunk_id = 0
    
    for page_text, page_metadata in documents:
        # Split page into chunks
        page_chunks = text_splitter.split_text(page_text)
        
        # Add metadata to each chunk
        for chunk_text in page_chunks:
            chunk_metadata = page_metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["created_at"] = datetime.now().isoformat()
            
            chunks_with_metadata.append((chunk_text, chunk_metadata))
            chunk_id += 1
    
    print(f"Created {len(chunks_with_metadata)} chunks from {len(documents)} pages")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    return chunks_with_metadata


def chunk_documents_with_smart_metadata(
    documents: List[str],
    doc_metadata: List[Dict],
    chunk_size: int = 2000,
    chunk_overlap: int = 200
) -> List[Tuple[str, Dict]]:
    """
    Chunk full documents while estimating page numbers for each chunk.
    
    This maintains the same chunking strategy as FAISS (full document)
    while adding approximate page tracking for citations.
    
    Args:
        documents: List of full document texts
        doc_metadata: List of metadata dicts for each document
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of (chunk_text, metadata) tuples
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            print("ERROR: langchain text splitter not found.")
            print("Install with: pip install langchain-text-splitters")
            return []
    
    load_dotenv()
    chunk_size = int(os.getenv('CHUNK_SIZE', chunk_size))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', chunk_overlap))
    
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )
    
    chunks_with_metadata = []
    chunk_id = 0
    
    for doc_text, metadata in zip(documents, doc_metadata):
        doc_chunks = text_splitter.split_text(doc_text)
        
        # Estimate average chars per page
        total_pages = metadata["total_pages"]
        chars_per_page = len(doc_text) / total_pages if total_pages > 0 else 1
        
        char_position = 0
        for chunk_text in doc_chunks:
            # Estimate which page this chunk is on
            estimated_page = int(char_position / chars_per_page) + 1
            estimated_page = min(estimated_page, total_pages)  # Cap at total pages
            
            chunk_metadata = {
                "source": metadata["source"],
                "page": estimated_page,  # Approximate page number
                "chunk_id": chunk_id,
                "created_at": datetime.now().isoformat(),
                "total_pages": total_pages,
            }
            
            chunks_with_metadata.append((chunk_text, chunk_metadata))
            char_position += len(chunk_text) - chunk_overlap
            chunk_id += 1
    
    print(f"Created {len(chunks_with_metadata)} chunks from {len(documents)} documents")
    print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"Using SMART metadata (full-doc chunking + page estimation)")
    return chunks_with_metadata


def create_embeddings(chunks: List[str], output_path: str = "data/embeddings.pkl"):
    """
    Create embeddings for text chunks (legacy FAISS backend).
    
    Args:
        chunks: List of text chunks
        output_path: Path to save embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: pip install sentence-transformers")
        return
    
    # Load environment variables
    load_dotenv()
    model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Save embeddings and chunks
    data = {
        'chunks': chunks,
        'embeddings': embeddings
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Embeddings saved to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.2f} KB")


def upload_to_weaviate(chunks_with_metadata: List[Tuple[str, Dict]]):
    """
    Create embeddings and upload to Weaviate.
    
    Args:
        chunks_with_metadata: List of (chunk_text, metadata) tuples
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: pip install sentence-transformers")
        return
    
    try:
        import sys
        from pathlib import Path as PathLib
        # Add project root to Python path
        project_root = PathLib(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from rag.weaviate_store import WeaviateVectorStore
    except ImportError as e:
        print(f"WeaviateVectorStore not found: {e}")
        print("Check rag/weaviate_store.py")
        return
    
    # Load environment variables
    load_dotenv()
    model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    print(f"\nLoading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # Extract chunks and metadata
    chunks = [chunk for chunk, _ in chunks_with_metadata]
    metadata_list = [metadata for _, metadata in chunks_with_metadata]
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=False)
    embeddings_list = [emb.tolist() for emb in embeddings]
    
    # Connect to Weaviate and upload
    print("\nConnecting to Weaviate...")
    store = WeaviateVectorStore()
    
    # Create schema
    store.create_schema(embedding_dim=embedding_dim)
    
    # Upload documents
    print("Uploading to Weaviate...")
    store.add_documents(chunks, embeddings_list, metadata_list)
    
    # Show stats
    stats = store.get_collection_stats()
    print(f"\n✓ Upload complete!")
    print(f"  Total documents in Weaviate: {stats.get('total_documents', 'N/A')}")
    
    store.close()


if __name__ == "__main__":
    # Check dependencies first
    print("🔍 Checking dependencies...")
    missing_deps = []
    
    try:
        from pypdf import PdfReader
    except ImportError:
        missing_deps.append("pypdf")
    
    try:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        missing_deps.append("langchain-text-splitters")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"\n📦 Install with:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("\nOr install all requirements:")
        print("   pip install -r requirements.txt")
        import sys
        sys.exit(1)
    
    print("✅ All dependencies installed\n")
    
    # Check for backend configuration
    load_dotenv()
    backend = os.getenv('VECTOR_STORE', 'faiss').lower()
    print(f"📊 Vector store backend: {backend.upper()}")
    
    # Check for PDFs
    pdf_dir = "data/pdfs"
    pdf_count = len(list(Path(pdf_dir).glob("*.pdf"))) if Path(pdf_dir).exists() else 0
    print(f"📁 Found {pdf_count} PDF file(s) in {pdf_dir}")
    
    if pdf_count == 0:
        print(f"\n⚠️  No PDF files found. Add PDFs to {pdf_dir}/")
        import sys
        sys.exit(1)
    
    # Process based on backend
    if backend == 'weaviate':
        print("\n� Using WEAVIATE backend with SMART metadata tracking")
        print("   (Full document chunking + page estimation for high scores)\n")
        
        print("📄 Loading PDFs with smart metadata...")
        documents, doc_metadata = load_pdf_documents_with_smart_metadata(pdf_dir)
        
        if documents:
            print("\n📝 Chunking documents with smart metadata...")
            chunks_with_metadata = chunk_documents_with_smart_metadata(documents, doc_metadata)
            
            if chunks_with_metadata:
                # Check if weaviate-client is installed
                try:
                    import weaviate
                except ImportError:
                    print("\n❌ weaviate-client not installed")
                    print("📦 Install with: pip install weaviate-client")
                    import sys
                    sys.exit(1)
                
                # Check if Weaviate is running
                weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
                print(f"\n🔗 Checking Weaviate at {weaviate_url}...")
                try:
                    import requests
                    response = requests.get(f"{weaviate_url}/v1/.well-known/ready", timeout=2)
                    if response.status_code != 200:
                        raise Exception("Not ready")
                except:
                    print(f"\n❌ Cannot connect to Weaviate at {weaviate_url}")
                    print("📦 Start Weaviate with: docker-compose up -d")
                    import sys
                    sys.exit(1)
                
                upload_to_weaviate(chunks_with_metadata)
                print("\n✅ Done! Run: streamlit run ui/app.py")
            else:
                print("❌ No chunks created")
        else:
            print("❌ No documents to process")
    else:
        # Legacy FAISS backend
        print("\n📄 Loading PDFs (legacy mode - no metadata)...")
        documents = load_pdf_documents(pdf_dir)
        
        if documents:
            print("\n📝 Chunking documents...")
            chunks = chunk_documents(documents)
            
            if chunks:
                print("\n🔢 Creating embeddings...")
                create_embeddings(chunks)
                print("\n✅ Done! Embeddings ready for use.")
                print("   Run: streamlit run ui/app.py")
            else:
                print("❌ No chunks created")
        else:
            print("❌ No documents to process")
