# Architecture technique

## Vue d'ensemble

Le pipeline transforme des PDFs en vecteurs stockés dans Pinecone, prêts pour la recherche sémantique et le RAG.

```
DATA/ (PDFs)
    │
    ▼
[Extraction]  ──── text natif → pdfplumber / PyMuPDF / PyMuPDF4LLM
                └── scan OCR  → Tesseract | Mistral OCR (fallback auto)
    │
    ▼
OUTPUT/ (Markdown)
    │
    ▼
[Chunking]    ──── Standard : RecursiveCharacterTextSplitter (LangChain)
                └── Avancé   : détection de sections + enrichissement IA + contexte
    │
    ▼ (optimisations : token count, merge petits chunks, filtre qualité, priorisation)
    │
    ▼
[Embedding]   ──── OpenAI text-embedding-3-small (1536 dim)
                   cache SHA256(model+content) → .cache/embeddings/
                   batch processing + multithreading
    │
    ▼
[Storage]     ──── Classification namespace (IA ou mécanique)
                └── Pinecone upsert par namespace
```

---

## Modules

### `src/core/`

| Fichier | Rôle |
|---|---|
| `config.py` | `Settings` Pydantic — lit le `.env`, valide les champs |
| `logging_config.py` | `setup_logging()` et `get_logger()` |
| `exceptions.py` | `PipelineError`, `ConfigurationError`, `ValidationError`, `ErrorType` |
| `singletons.py` | Clients OpenAI / Pinecone / Mistral (pattern singleton) |
| `retry.py` | Décorateur `@retry_with_backoff` (max 3 tentatives, backoff exponentiel) |
| `cache.py` | `EmbeddingCache` — cache disque JSON par hash SHA256 |

### `src/pipeline/`

| Fichier | Rôle |
|---|---|
| `models.py` | Dataclasses et enums : `PipelineConfig`, `ExtractionResult`, `ChunkingResult`, `EmbeddingResult`, `StorageResult`, `NamespaceStrategy` |
| `interfaces.py` | Interfaces ABC pour l'architecture hexagonale |
| `services.py` | Implémentations : `ExtractionService`, `ChunkingService`, `EmbeddingService`, `StorageService` |
| `pipeline.py` | `Pipeline` — orchestre les 4 services en séquence |

### `src/extractors/`

| Fichier | Rôle |
|---|---|
| `text_extractor.py` | `extract_text_from_pdf()` — PDFs natifs via PyMuPDF |
| `text_extractor_v2.py` | `extract_structured_text_from_pdf()`, `extract_with_pymupdf4llm()`, `process_multiple_pdfs()` |
| `scan_extractor.py` | `extract_text_from_scan()` — OCR Tesseract |
| `mistral_ocr_extractor.py` | `extract_text_with_mistral_ocr()` — OCR via Mistral API |

### `src/processors/`

| Fichier | Rôle |
|---|---|
| `text_cleaner.py` | `clean_text()` — normalisation, suppression numéros de page, figures |
| `chunker.py` | `chunk_all_markdown_files()` — découpage standard |
| `advanced_chunker.py` | `AdvancedChunker` + `process_all_markdown_files()` — pipeline complet |
| `chunking_strategies.py` | `AdaptiveChunker`, `SemanticChunker`, `ContentTypeDetector`, `SentenceWindowChunker` |
| `token_based_chunker.py` | `TokenBasedChunker` — calcul token count via tiktoken |
| `chunk_quality_filter.py` | `ChunkQualityFilter` — filtre les chunks de faible qualité (score < seuil) |
| `chunk_merger.py` | `ChunkMerger` — fusionne les chunks trop petits |
| `chunk_prioritizer.py` | `ChunkPrioritizer` — trie les chunks par pertinence |
| `section_detector.py` | `SectionDetector` — détecte la hiérarchie de sections (H1/H2/H3) |
| `metadata_enricher.py` | `MetadataEnricher` — enrichit les métadonnées (keywords, topics, summary) |
| `contextual_augmenter.py` | `ContextualAugmenter` — ajoute du contexte au contenu des chunks |
| `state_manager.py` | `StateManager` — sérialise/désérialise chunks et embeddings (pickle) |

### `src/vectorization/`

| Fichier | Rôle |
|---|---|
| `embeddings.py` | `create_embeddings()`, `embed_chunks()`, `embed_all_files()` — appels OpenAI avec cache + threading |
| `vector_store.py` | `VectorStore` — CRUD Pinecone ; `add_chunks_distributed()` — dispatch multi-namespace |
| `smart_batching.py` | `SmartBatcher` — batch par tokens (vs batch par nombre fixe) |
| `namespace_classifier.py` | `classify_chunks()` — classification GPT-4o-mini → depannage / dimensionnement / general |

### `src/cli/` et `src/ui/`

| Fichier | Rôle |
|---|---|
| `cli/cli.py` | `CLIApplication` — menu interactif terminal |
| `ui/app.py` | `run_ui()` — point d'entrée PySide6 |
| `ui/main_window.py` | Fenêtre principale |
| `ui/dialogs.py` | Dialogues de configuration (extraction, chunking, namespace, storage) |
| `ui/workers.py` | `PipelineWorker` — exécution du pipeline dans un thread séparé |

---

## Flux de données détaillé

### 1. Extraction

```python
ExtractionService.extract(data_dir, output_dir, extraction_mode, pdf_filter)
    → analyze_pdfs()           # classe les PDFs : text | scan
    → extraction_func(pdf)     # selon extraction_mode
    → OUTPUT/*.md              # fichiers markdown
```

### 2. Chunking

```python
ChunkingService.chunk(output_dir, chunk_size, chunk_overlap, chunking_mode)
    → chunk_all_markdown_files() | process_all_markdown_files()
    → TokenBasedChunker.get_token_count()   # ajout token_count en metadata
    → ChunkMerger.merge_chunks()            # si merge_small_chunks=True
    → ChunkQualityFilter.filter_chunks()    # si filter_chunk_quality=True
    → ChunkPrioritizer.prioritize_chunks()  # si prioritize_chunks=True
    → List[Dict]  # chunks avec metadata enrichie
```

### 3. Embedding

```python
EmbeddingService.embed(chunks_data, model, batch_size)
    → embed_all_files()
        → SmartBatcher (optionnel) : batch par tokens
        → embed_chunks()
            → create_embeddings()        # avec cache + retry
                → cache.get(text, model) # SHA256 lookup
                → OpenAI embeddings API  # si pas en cache
                → cache.set(...)         # mise en cache
        → List[Dict]  # chunks + embedding: List[float]
```

### 4. Storage (namespace IA)

```python
StorageService.store(enriched_chunks, namespace_strategy=BY_AI)
    → VectorStore.add_chunks_distributed(chunks, strategy="by_ai")
        → classify_chunks(chunks, batch_size=25)
            # 25 chunks par appel GPT-4o-mini
            # → ["depannage", "general", "dimensionnement", ...]
        → group chunks by namespace
        → VectorStore.add_chunks(group, namespace="depannage")
        → VectorStore.add_chunks(group, namespace="dimensionnement")
        → VectorStore.add_chunks(group, namespace="general")
```

---

## Modèles de données

### `PipelineConfig`

```python
@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    extraction_mode: ExtractionMode      # BASIC | STRUCTURED | PYMUPDF4LLM | MISTRAL_OCR
    pdf_filter: PDFFilter                # ALL | TEXT | SCAN
    chunking_mode: ChunkingMode          # STANDARD | ADVANCED
    chunk_size: int                      # défaut 1000
    chunk_overlap: int                   # défaut 200
    embedding_model: str                 # défaut "text-embedding-3-small"
    embedding_batch_size: int            # défaut 100
    namespace: str                       # namespace explicite (strategy=NONE)
    namespace_strategy: NamespaceStrategy  # BY_AI | BY_FILE | BY_FOLDER | NONE
    namespace_prefix: str                # préfixe optionnel
    reset_namespace: bool                # vider le namespace avant upload
    verbose: bool
```

### Structure d'un chunk

```python
{
    "content": "texte du chunk...",
    "metadata": {
        "source": "/path/to/file.md",
        "file_name": "rapport.md",
        "chunk_index": 3,
        "total_chunks": 45,
        "chunk_size": 987,
        "token_count": 215,
        "quality_score": 0.82,
        "priority_score": 0.75,
        # optionnel (mode avancé) :
        "section_hierarchy": ["1. Introduction", "1.2 Contexte"],
        "keywords": ["pompe", "débit", "pression"],
        "content_type": "prose"
    },
    "embedding": [0.023, -0.041, ...]   # ajouté après embed_chunks()
}
```

---

## Cache

Le cache des embeddings est stocké dans `.cache/embeddings/` :

```
.cache/
├── embeddings/
│   ├── ab/
│   │   └── ab3f7c2e...json   # {hash, model, embedding, dimension}
│   └── ...
├── chunks_results.pkl
└── embeddings_results.pkl
```

La clé de cache est `SHA256(model + content)`. Un changement de modèle invalide automatiquement le cache (nouvelle clé).

---

## Gestion des erreurs

Toutes les erreurs du pipeline sont encapsulées dans `PipelineError(ErrorType, message)` avec les types :

| `ErrorType` | Quand |
|---|---|
| `PDF_EXTRACTION` | Erreur lecture PDF |
| `EMBEDDING` | Erreur appel OpenAI embeddings |
| `PINECONE` | Erreur connexion/upload Pinecone |
| `CONFIGURATION` | Clé API manquante, paramètre invalide |
| `UNKNOWN` | Autre exception non catchée |

Le décorateur `@retry_with_backoff` applique automatiquement 3 tentatives avec délai exponentiel (2s → 30s) sur les appels OpenAI.
