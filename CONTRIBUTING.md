# Contributing Guide

Thank you for your interest in contributing to OCR-Vector-Doc. This document describes the conventions and contribution process.

## Quick Start

1. **Fork** the repository and clone your fork
2. Install dependencies: `uv sync`
3. Create a **branch** for your feature: `git checkout -b feature/my-feature`
4. Make your **changes** following the conventions below
5. **Test** your changes
6. **Commit** with clear messages
7. **Push** and open a Pull Request

## Dependency Management

This project uses **uv only**:
- `uv sync` — install/synchronize dependencies
- `uv run python script.py` — run a script
- `uv add package` — add a dependency (updates `pyproject.toml`)

## Code Conventions

### Architecture

The project follows a modular architecture with separation of concerns:

- **`src/core/`**: Configuration, logging, cache, singletons, retry
- **`src/pipeline/`**: Pure business logic (services, models)
- **`src/cli/`**: CLI user interface only
- **`src/extractors/`**: PDF extraction
- **`src/processors/`**: Document processing
- **`src/vectorization/`**: Embeddings and storage

### Principles

- **DRY**: No code duplication
- **SRP**: One responsibility per class/function
- **Pure services**: No `print()` or `input()` in business logic
- **Type hints**: Always type parameters and return values

### Configuration

Use `src.core.settings` for all configuration:

```python
from src.core import settings

settings.data_dir
settings.chunk_size
settings.embedding_model
```

### Logging

Use `get_logger()` instead of `print()` in business logic:

```python
from src.core import get_logger

logger = get_logger(__name__)
logger.info("Message")
logger.error("Error", exc_info=True)
```

### Error Handling

Use `PipelineError` with `ErrorType`:

```python
from src.core import PipelineError, ErrorType

raise PipelineError(
    ErrorType.EMBEDDING,
    "Error message",
    original_error=e
)
```

### New Service

1. Define the interface in `src/pipeline/interfaces.py`
2. Implement in `src/pipeline/services.py`
3. Add the `Result` model in `src/pipeline/models.py`
4. Integrate in the `Pipeline`

## Commit Format

Clear and concise commit messages:

```
type: short description

Detailed description if needed.
```

Recommended types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:
- `feat: add Mistral OCR mode for scans`
- `fix: fix chunking for long documents`
- `docs: update README`

## Tests

Before submitting a PR:

```bash
# Verify imports work
uv run python -c "from src.pipeline import Pipeline; print('OK')"

# Run pipeline CLI for manual test
uv run python generate.py
```

## Pull Requests

- Clearly describe the changes
- Reference related issues if applicable
- Ensure code follows project conventions
- Manually test main flows (extraction, vectorization, ask)

## Questions

For any questions, open an issue on the repository.
