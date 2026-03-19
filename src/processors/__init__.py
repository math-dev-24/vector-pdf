"""
Modules de traitement de texte (nettoyage et chunking).
"""

# Modules de base
from .text_cleaner import clean_text
from .chunker import chunk_markdown_file, chunk_all_markdown_files
from .state_manager import StateManager

# Modules avancés
from .advanced_chunker import AdvancedChunker, process_all_markdown_files
from .section_detector import SectionDetector
from .metadata_enricher import MetadataEnricher
from .contextual_augmenter import ContextualAugmenter
from .chunking_strategies import (
    AdaptiveChunker,
    SemanticChunker,
    ContentTypeDetector,
    SentenceWindowChunker
)
from .token_based_chunker import TokenBasedChunker
from .chunk_quality_filter import ChunkQualityFilter
from .chunk_merger import ChunkMerger
from .chunk_prioritizer import ChunkPrioritizer

__all__ = [
    # Base
    'clean_text',
    'chunk_markdown_file',
    'chunk_all_markdown_files',
    'StateManager',
    # Avancé
    'AdvancedChunker',
    'process_all_markdown_files',
    'SectionDetector',
    'MetadataEnricher',
    'ContextualAugmenter',
    'AdaptiveChunker',
    'SemanticChunker',
    'ContentTypeDetector',
    'SentenceWindowChunker',
    # Optimisations
    'TokenBasedChunker',
    'ChunkQualityFilter',
    'ChunkMerger',
    'ChunkPrioritizer',
]
