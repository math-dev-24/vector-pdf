"""
Module pipeline - Architecture modulaire pour le traitement de documents.
"""

from .pipeline import Pipeline
from .services import (
    ExtractionService,
    ChunkingService,
    EmbeddingService,
    StorageService
)
from .models import PipelineConfig, PipelineResult

__all__ = [
    "Pipeline",
    "ExtractionService",
    "ChunkingService",
    "EmbeddingService",
    "StorageService",
    "PipelineConfig",
    "PipelineResult",
]
