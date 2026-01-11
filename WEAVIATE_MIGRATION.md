# Weaviate Migration Guide

## ✅ Implementation Complete!

Your fraud-chatbot now uses **Weaviate** as the vector store with full metadata support (source file, page number, timestamps).

## What Changed

### New Files
- `docker-compose.yml` - Weaviate service configuration
- `rag/weaviate_store.py` - Weaviate vector store implementation with metadata

### Updated Files
- `ingestion/load_docs.py` - Now extracts page-level metadata
- `rag/retriever.py` - Supports both FAISS and Weaviate backends
- `llm/qa_chain.py` - Handles metadata in responses
- `requirements.txt` - Added weaviate-client>=4.4.0
- `.env.example` - Added Weaviate configuration

## Quick Start

### 1. Start Weaviate (Docker)
```bash
docker-compose up -d
```

Verify it's running:
```bash
curl http://localhost:8080/v1/meta
# Should show: {"version":"1.28.5",...}
```

### 2. Configure Environment
Ensure your `.env` has:
```bash
VECTOR_STORE=weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_COLLECTION=FraudDocuments
```

### 3. Ingest Documents
```bash
source .venv/bin/activate
python ingestion/load_docs.py
```

This will:
- Load PDFs from `data/pdfs/`
- Extract text with page-level metadata
- Create embeddings
- Upload to Weaviate

### 4. Run the App
```bash
source .venv/bin/activate
streamlit run ui/app.py
```

## Benefits Over FAISS

| Feature | FAISS (Old) | Weaviate (New) |
|---------|-------------|----------------|
| **Metadata** | ❌ None | ✅ Source, page, timestamps |
| **Persistence** | Manual (pickle) | Automatic (Docker volume) |
| **Filtered Search** | ❌ Not supported | ✅ `filters={"page": 2}` |
| **Citations** | ❌ No source tracking | ✅ Shows "Bhatla.pdf, Page 10" |
| **Updates** | Rebuild entire index | Real-time add/delete |
| **Production Ready** | Single machine only | Scalable architecture |

## Example Usage

### Basic Query
```python
from rag.retriever import Retriever

retriever = Retriever(backend='weaviate')
results = retriever.retrieve("What are fraud indicators?", k=5)

for text, score, metadata in results:
    print(f"Source: {metadata['source']}, Page: {metadata['page']}")
    print(f"Score: {score:.3f}")
    print(f"Text: {text[:200]}...\n")
```

### Filtered Search (Weaviate only)
```python
# Search only in specific document
results = retriever.retrieve(
    "fraud detection",
    k=5,
    filters={"source": "Bhatla.pdf"}
)

# Search only specific pages
results = retriever.retrieve(
    "fraud detection",
    k=5,
    filters={"source": "Bhatla.pdf", "page": 10}
)
```

### QA Chain (Same API)
```python
from llm.qa_chain import QAChain

qa = QAChain()
result = qa.ask("What are common fraud detection methods?")

print(result['answer'])
# Now shows: "[Document 1] [Source: Bhatla.pdf, Page: 10]..."
```

## Switching Between Backends

Change `VECTOR_STORE` in `.env`:
```bash
# Use Weaviate (recommended)
VECTOR_STORE=weaviate

# Use FAISS (legacy)
VECTOR_STORE=faiss
```

Both backends work with the same code - no changes needed!

## Docker Commands

```bash
# Start Weaviate
docker-compose up -d

# Stop Weaviate
docker-compose down

# View logs
docker-compose logs -f

# Restart after config changes
docker-compose restart

# Remove all data (fresh start)
docker-compose down -v
```

## Troubleshooting

### Weaviate not connecting
```bash
# Check if running
docker ps

# Check logs
docker-compose logs weaviate

# Restart
docker-compose restart
```

### Import errors
```bash
# Make sure weaviate-client is installed
source .venv/bin/activate
pip install weaviate-client
```

### Reset collection
```bash
source .venv/bin/activate
python -c "
from rag.weaviate_store import WeaviateVectorStore
store = WeaviateVectorStore()
store.delete_collection()
store.close()
"
# Then re-run ingestion
python ingestion/load_docs.py
```

## Next Steps

1. **Add more PDFs** to `data/pdfs/` and re-run ingestion
2. **Implement filtering** in the UI to search specific pages/documents
3. **Monitor usage** with `store.get_collection_stats()`
4. **Scale to cloud** using Weaviate Cloud (WCD) when ready

## Production Deployment

For production with Weaviate Cloud:

1. Sign up at https://console.weaviate.cloud
2. Create a cluster
3. Update `.env`:
   ```bash
   VECTOR_STORE=weaviate
   WEAVIATE_URL=https://your-cluster.weaviate.network
   WEAVIATE_API_KEY=your_api_key_here
   ```
4. Run ingestion to populate cloud instance

---

**Status**: ✅ Migration Complete
**Backend**: Weaviate 1.28.5 (Local Docker)
**Documents**: 35 chunks from Bhatla.pdf with metadata
**Metadata**: Source file, page number, chunk ID, timestamp
