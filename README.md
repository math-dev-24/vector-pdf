# OCR-Vector-Doc

PDF document processing pipeline: OCR extraction, chunking, vectorization, and storage in a vector database (Pinecone). Ideal for building RAG (Retrieval-Augmented Generation) applications.

## Features

- **PDF Extraction**: Native text, scans (Tesseract), or advanced OCR (Mistral AI)
- **Extraction Modes**: Basic, structured (title detection), PyMuPDF4LLM
- **Smart Chunking**: Standard or advanced (AI enrichment, context)
- **Vectorization**: OpenAI embeddings with cache and batch processing
- **Storage**: Pinecone with namespaces to organize documents
- **Interface**: Interactive CLI, GUI (PySide6), Python scripts

## Prerequisites

- **uv**: [Installation](https://docs.astral.sh/uv/getting-started/installation/)
- **Python** 3.13+
- **Tesseract OCR** (for scanned PDFs): [Installation](https://github.com/tesseract-ocr/tesseract)
- **API Keys**: OpenAI, Pinecone (Mistral optional for advanced OCR)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vector-pdf.git
cd vector-pdf

# Install dependencies with uv
uv sync
```

## Configuration

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your API keys:
   ```env
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   PINECONE_INDEX_NAME=pdf-documents
   PINECONE_DIMENSION=1536
   ```

3. Create a Pinecone index (dimension 1536, cosine metric) at [app.pinecone.io](https://app.pinecone.io/)

4. Place your PDFs in the `DATA/` folder (created automatically)

## Usage

### Pipeline CLI (generate.py)

```bash
uv run python generate.py
```

Interactive menu:
1. **PDF to MD** — Extraction only (PDF → Markdown)
2. **Vectorization** — Chunking + embeddings
3. **Go to DB** — Storage in Pinecone
4. **Full Pipeline** — All steps at once
5. **Cache Status** — Check cached data
6. **Clear Cache** — Remove chunks/embeddings

### Query the Database (ask.py)

```bash
uv run python ask.py
```

Ask questions in natural language and get the most relevant chunks via semantic search.

### Graphical Interface

```bash
uv run python ui.py
# or
uv run ocr-vector-ui
```

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `DATA/` | Source PDFs |
| `OUTPUT/` | Extracted Markdown files |
| `.cache/` | Cached chunks and embeddings |

## Architecture

```
src/
├── core/          # Config, logging, cache, retry, singletons
├── pipeline/      # Business logic (extraction, chunking, embedding, storage)
├── cli/           # CLI interface
├── extractors/    # PDF extractors (text, scan, Mistral OCR)
├── processors/    # Chunking, cleaning, enrichment
├── vectorization/ # Embeddings, Pinecone VectorStore
└── ui/            # PySide6 GUI
```

## Advanced Configuration (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_SIZE` | Chunk size (characters) | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `USE_SEMANTIC_CHUNKING` | Section-based chunking | false |
| `EMBEDDING_MODEL` | OpenAI model | text-embedding-3-small |
| `USE_MISTRAL_OCR` | Mistral OCR for scans | false |
| `MAX_WORKERS_EMBEDDINGS` | Threads for embeddings | 4 |

## License

MIT License — see [LICENSE](LICENSE) for details.
