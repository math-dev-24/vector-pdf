"""
Modules de vectorisation et stockage des embeddings.
"""

from .embeddings import create_embeddings, embed_chunks
from .vector_store import VectorStore

__all__ = ['create_embeddings', 'embed_chunks', 'VectorStore']
