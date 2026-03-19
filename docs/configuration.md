# Référence de configuration

Toutes les options sont définies dans le fichier `.env` à la racine du projet.
Les valeurs par défaut sont appliquées si la variable n'est pas définie.

---

## Dossiers

| Variable | Défaut | Description |
|---|---|---|
| `DATA_DIR` | `./DATA` | Dossier contenant les PDFs source |
| `OUTPUT_DIR` | `./OUTPUT` | Dossier de sortie des fichiers Markdown |
| `CACHE_DIR` | `./.cache` | Cache local (chunks et embeddings) |

---

## Pinecone

| Variable | Défaut | Description |
|---|---|---|
| `PINECONE_API_KEY` | *(requis)* | Clé API Pinecone |
| `PINECONE_INDEX_NAME` | `pdf-documents` | Nom de l'index Pinecone |
| `PINECONE_DIMENSION` | `1536` | Dimension des vecteurs (dépend du modèle d'embedding) |
| `PINECONE_CLOUD` | `aws` | Cloud provider Pinecone (`aws`, `gcp`, `azure`) |
| `PINECONE_REGION` | `us-east-1` | Région Pinecone (plan gratuit : `us-east-1`) |
| `NAMESPACE_STRATEGY` | `by_ai` | Stratégie de namespace : `none`, `by_file`, `by_folder`, `by_ai` |
| `NAMESPACE_PREFIX` | *(vide)* | Préfixe optionnel pour les namespaces auto-générés |

### Stratégies de namespace

| Valeur | Comportement |
|---|---|
| `by_ai` | Chaque chunk est classifié par GPT-4o-mini → `depannage`, `dimensionnement`, ou `general` |
| `by_file` | Un namespace par fichier PDF (nom du fichier sans extension) |
| `by_folder` | Un namespace par dossier parent |
| `none` | Namespace unique défini manuellement |

---

## OpenAI

| Variable | Défaut | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(requis)* | Clé API OpenAI |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modèle d'embedding OpenAI |
| `EMBEDDING_BATCH_SIZE` | `100` | Nombre de textes par appel API (max 100) |
| `EMBEDDING_CACHE_ENABLED` | `true` | Activer le cache local des embeddings |
| `EMBEDDING_VERSION` | `1.0` | Version des embeddings (stockée en metadata Pinecone) |
| `ENABLE_ASYNC` | `true` | Utiliser les appels async pour les embeddings |
| `SMART_BATCHING` | `true` | Batching par tokens (vs nombre fixe de chunks) |

### Modèles d'embedding disponibles

| Modèle | Dimension | Coût |
|---|---|---|
| `text-embedding-3-small` | 1536 | $0.02 / 1M tokens |
| `text-embedding-3-large` | 3072 | $0.13 / 1M tokens |
| `text-embedding-ada-002` | 1536 | $0.10 / 1M tokens |

> Si vous changez de modèle, mettez à jour `PINECONE_DIMENSION` en conséquence et recréez l'index Pinecone.

---

## Mistral AI (OCR avancé)

| Variable | Défaut | Description |
|---|---|---|
| `MISTRAL_API_KEY` | *(optionnel)* | Clé API Mistral |
| `USE_MISTRAL_OCR` | `false` | Utiliser Mistral pour les PDFs scannés |
| `MISTRAL_OCR_FALLBACK` | `true` | Retomber sur Tesseract si Mistral échoue |

---

## Chunking

| Variable | Défaut | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Taille maximale d'un chunk (caractères) |
| `CHUNK_OVERLAP` | `200` | Chevauchement entre chunks consécutifs |
| `USE_TOKEN_BASED_CHUNKING` | `true` | Ajouter le `token_count` aux métadonnées de chaque chunk |
| `USE_SEMANTIC_CHUNKING` | `false` | Découper par sections sémantiques (recommandé pour longs documents) |
| `FILTER_CHUNK_QUALITY` | `true` | Supprimer les chunks de faible qualité |
| `MIN_CHUNK_QUALITY` | `0.5` | Score minimum de qualité (entre 0.0 et 1.0) |
| `MERGE_SMALL_CHUNKS` | `true` | Fusionner les chunks trop petits avant vectorisation |
| `PRIORITIZE_CHUNKS` | `true` | Trier les chunks par score de pertinence |

### Score de qualité

Le `ChunkQualityFilter` évalue chaque chunk sur 3 critères :
- **Longueur** : pénalise les chunks trop courts ou trop longs
- **Diversité lexicale** : ratio mots uniques / mots totaux
- **Structure** : présence de ponctuation, majuscules, organisation

---

## Performance

| Variable | Défaut | Description |
|---|---|---|
| `MAX_WORKERS` | *(auto)* | Nombre de threads pour les embeddings (défaut : min(4, nb_batches)) |
| `USE_MULTITHREADING` | `false` | Multithreading pour l'extraction PDF |

---

## Logging

| Variable | Défaut | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Niveau de log : `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | *(console)* | Chemin vers un fichier de log (ex: `logs/app.log`) |

---

## Exemple de fichier `.env` complet

```env
# === API Keys ===
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=pcsk_...
MISTRAL_API_KEY=...           # optionnel

# === Pinecone ===
PINECONE_INDEX_NAME=pdf-documents
PINECONE_DIMENSION=1536
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
NAMESPACE_STRATEGY=by_ai      # by_ai | by_file | by_folder | none

# === OpenAI Embedding ===
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=100
EMBEDDING_CACHE_ENABLED=true
SMART_BATCHING=true
ENABLE_ASYNC=true

# === Chunking ===
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
USE_TOKEN_BASED_CHUNKING=true
USE_SEMANTIC_CHUNKING=false
FILTER_CHUNK_QUALITY=true
MIN_CHUNK_QUALITY=0.5
MERGE_SMALL_CHUNKS=true
PRIORITIZE_CHUNKS=true

# === OCR ===
USE_MISTRAL_OCR=false
MISTRAL_OCR_FALLBACK=true

# === Performance ===
USE_MULTITHREADING=false

# === Logging ===
LOG_LEVEL=INFO
# LOG_FILE=logs/app.log
```
