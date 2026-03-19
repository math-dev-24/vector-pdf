# OCR-Vector-Doc

Pipeline de traitement de documents PDF : extraction OCR, chunking, vectorisation, et stockage dans une base vectorielle (Pinecone). Conçu pour des applications RAG (Retrieval-Augmented Generation).

---

## Fonctionnalités

- **Extraction PDF** : texte natif, scans (Tesseract), ou OCR avancé (Mistral AI)
- **Modes d'extraction** : basique, structuré (détection de titres), PyMuPDF4LLM
- **Chunking intelligent** : standard ou avancé (enrichissement IA, contexte)
- **Vectorisation** : embeddings OpenAI avec cache et traitement par batch
- **Namespaces automatiques** : classification IA en 3 catégories (Dépannage / Dimensionnement / Général), ou par fichier, par dossier, ou manuel
- **Stockage** : Pinecone serverless
- **Interfaces** : CLI interactive, GUI (PySide6), scripts Python

---

## Prérequis

- **uv** : [Installation](https://docs.astral.sh/uv/getting-started/installation/)
- **Python** 3.13+
- **Tesseract OCR** (pour les PDFs scannés) : [Installation](https://github.com/tesseract-ocr/tesseract)
- **Clés API** : OpenAI, Pinecone (Mistral optionnel pour l'OCR avancé)

---

## Installation

```bash
git clone https://github.com/your-org/vector-pdf.git
cd vector-pdf
uv sync
```

---

## Configuration

1. Copier le fichier d'exemple :
   ```bash
   cp .env.example .env
   ```

2. Renseigner les clés API dans `.env` :
   ```env
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   PINECONE_INDEX_NAME=pdf-documents
   ```

3. Placer les PDFs dans le dossier `DATA/` (créé automatiquement).

Voir [`docs/configuration.md`](docs/configuration.md) pour toutes les options disponibles.

---

## Utilisation

### CLI interactive

```bash
uv run python generate.py
```

Menu :
```
1. PDF to MD          — Extraction uniquement (PDF → Markdown)
2. Vectorisation      — Chunking + embeddings
3. Go to DB           — Stockage dans Pinecone
4. Pipeline complet   — Toutes les étapes d'un coup
5. État du cache      — Voir les données en cache
6. Nettoyer le cache  — Supprimer chunks/embeddings
```

### Interroger la base (semantic search)

```bash
uv run python ask.py
```

### Interface graphique

```bash
uv run python ui.py
# ou
uv run ocr-vector-ui
```

---

## Stratégie de namespace (Pinecone)

Lors du stockage dans Pinecone, chaque chunk est assigné à un namespace selon la stratégie choisie.

| Stratégie | Description | Exemple |
|---|---|---|
| **IA** *(défaut)* | GPT-4o-mini classifie chaque chunk | `depannage`, `dimensionnement`, `general` |
| **Par fichier** | 1 namespace = 1 fichier PDF | `rapport_2024`, `notice_installation` |
| **Par dossier** | 1 namespace = 1 dossier source | `contrats`, `manuels` |
| **Manuel** | Namespace unique saisi à la main | `mon-projet-v2` |

La stratégie IA utilise 3 namespaces fixes :
- `depannage` — diagnostic, codes d'erreur, résolution de pannes
- `dimensionnement` — calculs, sélection matériel, spécifications, débits
- `general` — présentation, normes, sécurité, tables des matières

---

## Structure du projet

```
vector-pdf/
├── DATA/                   # PDFs source (input)
├── OUTPUT/                 # Markdowns extraits
├── .cache/                 # Cache chunks + embeddings
├── docs/
│   ├── architecture.md     # Architecture technique détaillée
│   └── configuration.md    # Référence de toutes les options .env
├── src/
│   ├── core/               # Config, logging, cache, retry, singletons
│   ├── pipeline/           # Orchestration (extraction → chunking → embedding → storage)
│   ├── cli/                # Interface CLI
│   ├── extractors/         # Extracteurs PDF (texte, scan, Mistral OCR)
│   ├── processors/         # Chunking, nettoyage, enrichissement
│   ├── vectorization/      # Embeddings OpenAI, VectorStore Pinecone, classifier IA
│   └── ui/                 # Interface graphique PySide6
├── generate.py             # Point d'entrée CLI
├── ask.py                  # Recherche sémantique
└── ui.py                   # Point d'entrée GUI
```

---

## License

MIT License — voir [LICENSE](LICENSE) pour les détails.
