"""
Modèles de données pour le pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Literal
from enum import Enum


class ExtractionMode(str, Enum):
    """Modes d'extraction disponibles."""
    BASIC = "basic"
    STRUCTURED = "structured"
    PYMUPDF4LLM = "pymupdf4llm"
    MISTRAL_OCR = "mistral_ocr"


class PDFFilter(str, Enum):
    """Filtres pour les PDFs."""
    ALL = "all"
    TEXT = "text"
    SCAN = "scan"


class ChunkingMode(str, Enum):
    """Modes de chunking disponibles."""
    STANDARD = "standard"
    ADVANCED = "advanced"


@dataclass
class PipelineConfig:
    """Configuration du pipeline."""
    # Dossiers
    data_dir: Path
    output_dir: Path
    
    # Extraction
    extraction_mode: ExtractionMode = ExtractionMode.STRUCTURED
    pdf_filter: PDFFilter = PDFFilter.ALL
    
    # Chunking
    chunking_mode: ChunkingMode = ChunkingMode.ADVANCED
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    
    # Storage
    namespace: str = ""
    reset_namespace: bool = False
    
    # Options
    verbose: bool = True


@dataclass
class ExtractionResult:
    """Résultat de l'extraction."""
    output_files: List[str] = field(default_factory=list)
    text_pdfs_count: int = 0
    scan_pdfs_count: int = 0
    total_pages: int = 0


@dataclass
class ChunkingResult:
    """Résultat du chunking."""
    chunks: List[Dict] = field(default_factory=list)
    total_chunks: int = 0
    total_chars: int = 0


@dataclass
class EmbeddingResult:
    """Résultat de la vectorisation."""
    enriched_chunks: List[Dict] = field(default_factory=list)
    total_embeddings: int = 0
    dimension: int = 0


@dataclass
class StorageResult:
    """Résultat du stockage."""
    vector_store: Optional[object] = None
    total_vectors: int = 0
    namespace: str = ""


@dataclass
class PipelineResult:
    """Résultat complet du pipeline."""
    extraction: Optional[ExtractionResult] = None
    chunking: Optional[ChunkingResult] = None
    embedding: Optional[EmbeddingResult] = None
    storage: Optional[StorageResult] = None
    success: bool = False
    error: Optional[str] = None
