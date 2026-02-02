# OCR-Vector-Doc

Pipeline de traitement de documents PDF : extraction OCR, chunking, vectorisation et stockage dans une base de données vectorielle (Pinecone). Idéal pour construire des applications RAG (Retrieval-Augmented Generation).

## Fonctionnalités

- **Extraction PDF** : Texte natif, scans (Tesseract), ou OCR avancé (Mistral AI)
- **Modes d'extraction** : Basique, structurée (détection des titres), PyMuPDF4LLM
- **Chunking intelligent** : Standard ou avancé (enrichissement IA, contexte)
- **Vectorisation** : Embeddings OpenAI avec cache et batch processing
- **Stockage** : Pinecone avec namespaces pour organiser les documents
- **Interface** : CLI interactive, interface graphique (PySide6), scripts Python

## Prérequis

- **uv** : [Installation](https://docs.astral.sh/uv/getting-started/installation/)
- **Python** 3.13+
- **Tesseract OCR** (pour les PDFs scannés) : [Installation](https://github.com/tesseract-ocr/tesseract)
- **Clés API** : OpenAI, Pinecone (Mistral optionnel pour OCR avancé)

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-org/vector-pdf.git
cd vector-pdf

# Installer les dépendances avec uv
uv sync
```

## Configuration

1. Copier le fichier d'exemple :
   ```bash
   cp .env.example .env
   ```

2. Éditer `.env` avec vos clés API :
   ```env
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   PINECONE_INDEX_NAME=pdf-documents
   PINECONE_DIMENSION=1536
   ```

3. Créer un index Pinecone (dimension 1536, métrique cosine) sur [app.pinecone.io](https://app.pinecone.io/)

4. Placer vos PDFs dans le dossier `DATA/` (créé automatiquement)

## Utilisation

### Pipeline CLI (generate.py)

```bash
uv run python generate.py
```

Menu interactif :
1. **PDF to MD** — Extraction uniquement (PDF → Markdown)
2. **Vectorisation** — Chunking + embeddings
3. **Go to DB** — Stockage dans Pinecone
4. **Pipeline complet** — Tout en une fois
5. **État du cache** — Vérifier les données en cache
6. **Nettoyer le cache** — Supprimer chunks/embeddings

### Interroger la base (ask.py)

```bash
uv run python ask.py
```

Permet de poser des questions en langage naturel et d'obtenir les chunks les plus pertinents via recherche sémantique.

### Interface graphique

```bash
uv run python ui.py
# ou
uv run ocr-vector-ui
```

### Structure des dossiers

| Dossier | Rôle |
|---------|------|
| `DATA/` | PDFs sources |
| `OUTPUT/` | Fichiers Markdown extraits |
| `.cache/` | Chunks et embeddings en cache |

## Architecture

```
src/
├── core/          # Config, logging, cache, retry, singletons
├── pipeline/      # Services métier (extraction, chunking, embedding, storage)
├── cli/           # Interface CLI
├── extractors/    # Extracteurs PDF (texte, scan, Mistral OCR)
├── processors/    # Chunking, nettoyage, enrichissement
├── vectorization/ # Embeddings, VectorStore Pinecone
└── ui/            # Interface graphique PySide6
```

## Configuration avancée (.env)

| Variable | Description | Défaut |
|----------|-------------|--------|
| `CHUNK_SIZE` | Taille des chunks (caractères) | 1000 |
| `CHUNK_OVERLAP` | Chevauchement entre chunks | 200 |
| `USE_SEMANTIC_CHUNKING` | Chunking par sections | false |
| `EMBEDDING_MODEL` | Modèle OpenAI | text-embedding-3-small |
| `USE_MISTRAL_OCR` | OCR Mistral pour scans | false |
| `MAX_WORKERS_EMBEDDINGS` | Threads pour embeddings | 4 |

## Licence

MIT License — voir [LICENSE](LICENSE) pour les détails.
