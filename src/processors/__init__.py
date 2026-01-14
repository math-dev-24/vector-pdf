"""
Modules de traitement de texte (nettoyage et chunking).
"""

# Modules de base
from .text_cleaner import clean_text, clean_markdown_file
from .chunker import chunk_markdown_file, chunk_all_markdown_files
from .state_manager import StateManager

# Modules avancés
from .advanced_chunker import AdvancedChunker, process_all_markdown_files
from .section_detector import SectionDetector, add_section_metadata
from .metadata_enricher import MetadataEnricher, enrich_all_chunks
from .contextual_augmenter import ContextualAugmenter, augment_all_chunks
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
    'clean_markdown_file',
    'chunk_markdown_file',
    'chunk_all_markdown_files',
    'StateManager',
    # Avancé
    'AdvancedChunker',
    'process_all_markdown_files',
    'SectionDetector',
    'add_section_metadata',
    'MetadataEnricher',
    'enrich_all_chunks',
    'ContextualAugmenter',
    'augment_all_chunks',
    'AdaptiveChunker',
    'SemanticChunker',
    'ContentTypeDetector',
    'SentenceWindowChunker',
    # Optimisations
    'TokenBasedChunker',
    'ChunkQualityFilter',
    'ChunkMerger',
    'ChunkPrioritizer'
]
