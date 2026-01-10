"""
Load PDF documents, chunk them, and create embeddings for RAG.
"""

import os
from pathlib import Path
from typing import List
import pickle
from dotenv import load_dotenv


def load_pdf_documents(pdf_dir: str) -> List[str]:
    """
    Load PDF documents from directory.
    
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
        
        documents.append(text)
        print(f"  Extracted {len(text)} characters")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split documents into smaller chunks using semantic separators.
    
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


def create_embeddings(chunks: List[str], output_path: str = "data/embeddings.pkl"):
    """
    Create embeddings for text chunks.
    
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
    
    # Example usage
    pdf_dir = "data/pdfs"
    
    print(f"📁 Checking {pdf_dir} for PDF files...")
    from pathlib import Path
    pdf_count = len(list(Path(pdf_dir).glob("*.pdf"))) if Path(pdf_dir).exists() else 0
    print(f"   Found {pdf_count} PDF file(s)")
    
    if pdf_count == 0:
        print(f"\n⚠️  No PDF files found in {pdf_dir}/")
        print("   Add PDF files to process.")
        import sys
        sys.exit(1)
    
    # Load and process documents
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
