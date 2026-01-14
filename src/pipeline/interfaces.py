"""
Interfaces abstraites pour l'architecture hexagonale (Ports).
Permet de facilement remplacer les implémentations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

from .models import (
    ExtractionResult,
    ChunkingResult,
    EmbeddingResult,
    StorageResult,
    ExtractionMode,
    PDFFilter,
    ChunkingMode
)


class IExtractionService(ABC):
    """Interface pour le service d'extraction."""
    
    @abstractmethod
    def extract(
        self,
        data_dir: Path,
        output_dir: Path,
        extraction_mode: ExtractionMode,
        pdf_filter: PDFFilter
    ) -> ExtractionResult:
        """Extrait le texte des PDFs."""
        pass


class IChunkingService(ABC):
    """Interface pour le service de chunking."""
    
    @abstractmethod
    def chunk(
        self,
        output_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
        chunking_mode: ChunkingMode
    ) -> ChunkingResult:
        """Découpe les documents en chunks."""
        pass


class IEmbeddingService(ABC):
    """Interface pour le service d'embeddings."""
    
    @abstractmethod
    def embed(
        self,
        chunks_data: List[Dict],
        model: str,
        batch_size: int
    ) -> EmbeddingResult:
        """Crée les embeddings pour les chunks."""
        pass


class IStorageService(ABC):
    """Interface pour le service de stockage."""
    
    @abstractmethod
    def store(
        self,
        enriched_chunks: List[Dict],
        namespace: str,
        reset: bool
    ) -> StorageResult:
        """Stocke les embeddings."""
        pass
