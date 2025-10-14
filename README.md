# OCR-VECTOR-DOC

Pipeline complet pour extraire, chunker et vectoriser des documents PDF dans Pinecone pour du RAG (Retrieval-Augmented Generation).

## Fonctionnalités

- **Extraction PDF intelligente** : 3 modes (basique, structuré, PyMuPDF4LLM)
- **Détection automatique de titres** : Préserve la structure pour un meilleur chunking
- **OCR pour scans** : Support des PDFs scannés avec Tesseract
- **Chunking optimisé** : LangChain RecursiveCharacterTextSplitter
- **Vectorisation OpenAI** : Embeddings avec text-embedding-3-small
- **Stockage Pinecone** : Support des namespaces pour organiser vos données
- **Cache intelligent** : Réutilise les embeddings pour économiser les coûts API
- **Menu interactif** : Workflow modulaire (extraction, vectorisation, stockage séparés)

## Installation

### Prérequis

- Python 3.10+
- Tesseract OCR (pour les PDFs scannés)

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang
```

### Installation du projet

```bash
# Cloner le repo
git clone <your-repo>
cd OCR-VECTOR-DOC

# Installer uv (gestionnaire de packages rapide)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les d�pendances
uv sync

# (Optionnel) Pour PyMuPDF4LLM
pip install pymupdf4llm
```

### Configuration

```bash
# Copier le fichier d'exemple
cp .env.example .env

# �diter avec vos cl�s API
nano .env
```

Remplir :
```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=pdf-documents
```
