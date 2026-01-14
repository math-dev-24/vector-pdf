"""
Système de métriques pour monitorer les performances.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from contextlib import contextmanager
from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class Metrics:
    """Métriques de performance du pipeline."""
    extraction_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    storage_time: float = 0.0
    total_time: float = 0.0
    
    # Compteurs
    total_pdfs: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_vectors: int = 0
    
    # Coûts (estimés)
    estimated_cost: float = 0.0
    
    # Erreurs
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convertit les métriques en dictionnaire."""
        return {
            'timings': {
                'extraction': self.extraction_time,
                'chunking': self.chunking_time,
                'embedding': self.embedding_time,
                'storage': self.storage_time,
                'total': self.total_time
            },
            'counts': {
                'pdfs': self.total_pdfs,
                'chunks': self.total_chunks,
                'embeddings': self.total_embeddings,
                'vectors': self.total_vectors
            },
            'cost': self.estimated_cost,
            'errors': self.errors
        }
    
    def print_summary(self) -> None:
        """Affiche un résumé des métriques."""
        logger.info("=" * 60)
        logger.info("MÉTRIQUES DE PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"\n⏱️  Temps d'exécution:")
        logger.info(f"  - Extraction: {self.extraction_time:.2f}s")
        logger.info(f"  - Chunking: {self.chunking_time:.2f}s")
        logger.info(f"  - Embedding: {self.embedding_time:.2f}s")
        logger.info(f"  - Storage: {self.storage_time:.2f}s")
        logger.info(f"  - Total: {self.total_time:.2f}s")
        logger.info(f"\n📊 Compteurs:")
        logger.info(f"  - PDFs: {self.total_pdfs}")
        logger.info(f"  - Chunks: {self.total_chunks}")
        logger.info(f"  - Embeddings: {self.total_embeddings}")
        logger.info(f"  - Vecteurs: {self.total_vectors}")
        if self.estimated_cost > 0:
            logger.info(f"\n💰 Coût estimé: ${self.estimated_cost:.4f}")
        if self.errors:
            logger.warning(f"\n⚠️  Erreurs: {len(self.errors)}")
        logger.info("=" * 60)


class MetricsCollector:
    """Collecteur de métriques."""
    
    def __init__(self):
        self.metrics = Metrics()
        self._start_time: Optional[float] = None
    
    def start(self) -> None:
        """Démarre le chronomètre global."""
        self._start_time = time.time()
    
    def stop(self) -> None:
        """Arrête le chronomètre global."""
        if self._start_time:
            self.metrics.total_time = time.time() - self._start_time
    
    @contextmanager
    def track_time(self, attribute: str):
        """
        Context manager pour tracker le temps d'une opération.
        
        Args:
            attribute: Nom de l'attribut dans Metrics (ex: 'extraction_time')
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            current = getattr(self.metrics, attribute, 0.0)
            setattr(self.metrics, attribute, current + elapsed)
    
    def add_error(self, error: str) -> None:
        """Ajoute une erreur aux métriques."""
        self.metrics.errors.append(error)
    
    def estimate_embedding_cost(self, num_tokens: int, model: str = "text-embedding-3-small") -> float:
        """
        Estime le coût des embeddings.
        
        Args:
            num_tokens: Nombre de tokens
            model: Modèle utilisé
            
        Returns:
            Coût estimé en dollars
        """
        # Prix par 1K tokens (approximatif, à mettre à jour selon les prix réels)
        pricing = {
            "text-embedding-3-small": 0.00002,  # $0.02 per 1M tokens
            "text-embedding-3-large": 0.00013,  # $0.13 per 1M tokens
            "text-embedding-ada-002": 0.0001,   # $0.10 per 1M tokens
        }
        
        price_per_1k = pricing.get(model, 0.00002)
        cost = (num_tokens / 1000) * price_per_1k
        self.metrics.estimated_cost += cost
        return cost
