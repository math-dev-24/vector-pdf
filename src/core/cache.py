"""
Système de cache pour les embeddings au niveau chunk.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, List
import logging

from .config import settings
from .exceptions import PipelineError, ErrorType

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache pour les embeddings au niveau chunk."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialise le cache.
        
        Args:
            cache_dir: Répertoire de cache (défaut: settings.cache_dir / "embeddings")
        """
        self.cache_dir = (cache_dir or settings.cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache d'embeddings initialisé: {self.cache_dir}")
    
    def _get_content_hash(self, content: str, model: str) -> str:
        """
        Génère un hash pour un contenu et un modèle.
        
        Args:
            content: Contenu du chunk
            model: Modèle d'embedding utilisé
            
        Returns:
            Hash SHA256
        """
        combined = f"{model}:{content}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, content_hash: str) -> Path:
        """
        Récupère le chemin du fichier de cache.
        
        Args:
            content_hash: Hash du contenu
            
        Returns:
            Chemin du fichier de cache
        """
        # Utiliser les 2 premiers caractères pour créer une structure de dossiers
        subdir = self.cache_dir / content_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{content_hash}.json"
    
    def get(self, content: str, model: str) -> Optional[List[float]]:
        """
        Récupère un embedding depuis le cache.
        
        Args:
            content: Contenu du chunk
            model: Modèle d'embedding utilisé
            
        Returns:
            Embedding si trouvé, None sinon
        """
        if not settings.embedding_cache_enabled:
            return None
        
        content_hash = self._get_content_hash(content, model)
        cache_path = self._get_cache_path(content_hash)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Vérifier que le hash correspond
                if data.get('hash') == content_hash and data.get('model') == model:
                    logger.debug(f"Cache hit pour hash {content_hash[:16]}...")
                    return data['embedding']
                else:
                    # Hash ou modèle différent, cache invalide
                    logger.warning(f"Cache invalide pour {content_hash[:16]}..., suppression")
                    cache_path.unlink()
                    return None
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture du cache: {e}")
            return None
    
    def set(self, content: str, model: str, embedding: List[float]) -> None:
        """
        Sauvegarde un embedding dans le cache.
        
        Args:
            content: Contenu du chunk
            model: Modèle d'embedding utilisé
            embedding: Vecteur d'embedding
        """
        if not settings.embedding_cache_enabled:
            return
        
        content_hash = self._get_content_hash(content, model)
        cache_path = self._get_cache_path(content_hash)
        
        try:
            data = {
                'hash': content_hash,
                'model': model,
                'embedding': embedding,
                'dimension': len(embedding)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            logger.debug(f"Embedding mis en cache: {content_hash[:16]}...")
        except Exception as e:
            logger.warning(f"Erreur lors de l'écriture du cache: {e}")
    
    def clear(self) -> None:
        """Vide le cache."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache d'embeddings vidé")
        except Exception as e:
            raise PipelineError(
                ErrorType.CACHE,
                f"Erreur lors du vidage du cache: {e}",
                original_error=e
            )
    
    def get_stats(self) -> dict:
        """
        Récupère les statistiques du cache.
        
        Returns:
            Dictionnaire avec les stats
        """
        if not self.cache_dir.exists():
            return {
                'total_cached': 0,
                'cache_size_mb': 0
            }
        
        cached_files = list(self.cache_dir.rglob("*.json"))
        total_size = sum(f.stat().st_size for f in cached_files)
        
        return {
            'total_cached': len(cached_files),
            'cache_size_mb': round(total_size / (1024 * 1024), 2)
        }


# Instance globale du cache
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Récupère l'instance globale du cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache
