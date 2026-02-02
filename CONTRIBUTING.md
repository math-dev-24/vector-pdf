# Guide de contribution

Merci de votre intérêt pour contribuer à OCR-Vector-Doc. Ce document décrit les conventions et le processus de contribution.

## Démarrage rapide

1. **Fork** le dépôt et cloner votre fork
2. Installer les dépendances : `uv sync`
3. Créer une **branche** pour votre fonctionnalité : `git checkout -b feature/ma-fonctionnalite`
4. Faire vos **modifications** en respectant les conventions ci-dessous
5. **Tester** vos changements
6. **Commit** avec des messages clairs
7. **Push** et ouvrir une Pull Request

## Gestion des dépendances

Ce projet utilise **uniquement uv** :
- `uv sync` — installer/synchroniser les dépendances
- `uv run python script.py` — exécuter un script
- `uv add package` — ajouter une dépendance (met à jour `pyproject.toml`)

## Conventions de code

### Architecture

Le projet suit une architecture modulaire avec séparation des responsabilités :

- **`src/core/`** : Configuration, logging, cache, singletons, retry
- **`src/pipeline/`** : Logique métier pure (services, modèles)
- **`src/cli/`** : Interface utilisateur CLI uniquement
- **`src/extractors/`** : Extraction PDF
- **`src/processors/`** : Traitement des documents
- **`src/vectorization/`** : Embeddings et stockage

### Principes

- **DRY** : Pas de duplication de code
- **SRP** : Une responsabilité par classe/fonction
- **Services purs** : Pas de `print()` ni `input()` dans la logique métier
- **Type hints** : Toujours typer les paramètres et retours

### Configuration

Utiliser `src.core.settings` pour toute la configuration :

```python
from src.core import settings

settings.data_dir
settings.chunk_size
settings.embedding_model
```

### Logging

Utiliser `get_logger()` au lieu de `print()` dans le code métier :

```python
from src.core import get_logger

logger = get_logger(__name__)
logger.info("Message")
logger.error("Erreur", exc_info=True)
```

### Gestion d'erreurs

Utiliser `PipelineError` avec `ErrorType` :

```python
from src.core import PipelineError, ErrorType

raise PipelineError(
    ErrorType.EMBEDDING,
    "Message d'erreur",
    original_error=e
)
```

### Nouveau service

1. Définir l'interface dans `src/pipeline/interfaces.py`
2. Implémenter dans `src/pipeline/services.py`
3. Ajouter le modèle `Result` dans `src/pipeline/models.py`
4. Intégrer dans le `Pipeline`

## Format des commits

Messages de commit clairs et concis :

```
type: description courte

Description détaillée si nécessaire.
```

Types recommandés : `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Exemples :
- `feat: ajout du mode Mistral OCR pour les scans`
- `fix: correction du chunking pour les documents longs`
- `docs: mise à jour du README`

## Tests

Avant de soumettre une PR :

```bash
# Vérifier que les imports fonctionnent
uv run python -c "from src.pipeline import Pipeline; print('OK')"

# Lancer le pipeline CLI pour un test manuel
uv run python generate.py
```

## Pull Requests

- Décrire clairement les changements
- Référencer les issues liées si applicable
- S'assurer que le code respecte les conventions du projet
- Tester manuellement les flux principaux (extraction, vectorisation, ask)

## Questions

Pour toute question, ouvrir une issue sur le dépôt.
