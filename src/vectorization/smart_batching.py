"""
Batch processing intelligent selon la taille des chunks.
Optimise les batchs selon les tokens plutôt que nombre fixe.
"""

from typing import List, Dict
import tiktoken

from src.core import get_logger, settings

logger = get_logger(__name__)


class SmartBatcher:
    """Crée des batchs intelligents basés sur les tokens."""
    
    def __init__(self, max_tokens_per_batch: int = 8000):
        """
        Initialise le batcher.
        
        Args:
            max_tokens_per_batch: Nombre maximum de tokens par batch
        """
        self.max_tokens_per_batch = max_tokens_per_batch
        try:
            self.encoding = tiktoken.encoding_for_model(settings.embedding_model)
        except (KeyError, AttributeError):
            # Fallback
            logger.warning("Utilisation de l'encodage cl100k_base par défaut")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def create_smart_batches(
        self,
        chunks: List[Dict]
    ) -> List[List[Dict]]:
        """
        Crée des batchs optimisés selon les tokens.
        
        Args:
            chunks: Liste de chunks
            
        Returns:
            Liste de batchs
        """
        if not chunks:
            return []
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # Utiliser token_count si disponible, sinon calculer
            token_count = chunk.get('metadata', {}).get('token_count')
            if token_count is None:
                token_count = len(self.encoding.encode(content))
                # Mettre en cache pour éviter de recalculer
                chunk['metadata']['token_count'] = token_count
            
            # Si le chunk seul dépasse la limite, le mettre dans son propre batch
            if token_count > self.max_tokens_per_batch:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                batches.append([chunk])
                current_tokens = 0
            # Si on peut ajouter au batch actuel
            elif current_tokens + token_count <= self.max_tokens_per_batch:
                current_batch.append(chunk)
                current_tokens += token_count
            # Sinon, finaliser le batch actuel
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = token_count
        
        # Dernier batch
        if current_batch:
            batches.append(current_batch)
        
        logger.debug(f"Créé {len(batches)} batch(s) intelligent(s)")
        return batches
    
    def get_batch_stats(self, batches: List[List[Dict]]) -> Dict:
        """
        Retourne les statistiques des batchs.
        
        Args:
            batches: Liste de batchs
            
        Returns:
            Statistiques
        """
        if not batches:
            return {}
        
        batch_sizes = [len(batch) for batch in batches]
        batch_tokens = []
        
        for batch in batches:
            total_tokens = sum(
                chunk.get('metadata', {}).get('token_count', 0)
                for chunk in batch
            )
            batch_tokens.append(total_tokens)
        
        return {
            'num_batches': len(batches),
            'avg_batch_size': sum(batch_sizes) / len(batch_sizes),
            'min_batch_size': min(batch_sizes),
            'max_batch_size': max(batch_sizes),
            'avg_tokens_per_batch': sum(batch_tokens) / len(batch_tokens) if batch_tokens else 0,
            'min_tokens_per_batch': min(batch_tokens) if batch_tokens else 0,
            'max_tokens_per_batch': max(batch_tokens) if batch_tokens else 0
        }
