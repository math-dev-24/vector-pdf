"""
Fusionne intelligemment les petits chunks avec leurs voisins.
Réduit le nombre de chunks inutiles et améliore la cohérence.
"""

from typing import List, Dict, Optional

from src.core import get_logger

logger = get_logger(__name__)


class ChunkMerger:
    """Fusionne les chunks trop petits."""
    
    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1500,
        merge_strategy: str = "sequential"  # "sequential", "semantic", "hybrid"
    ):
        """
        Initialise le merger.
        
        Args:
            min_chunk_size: Taille minimale avant fusion (caractères)
            max_chunk_size: Taille maximale après fusion (caractères)
            merge_strategy: Stratégie de fusion
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.merge_strategy = merge_strategy
    
    def merge_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Fusionne les chunks trop petits.
        
        Args:
            chunks: Liste de chunks
            
        Returns:
            Chunks fusionnés
        """
        if not chunks:
            return []
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_size = len(current_chunk.get('content', ''))
            
            # Si trop petit, essayer de fusionner
            if current_size < self.min_chunk_size and i + 1 < len(chunks):
                # Chercher le meilleur voisin à fusionner
                best_neighbor_idx = self._find_best_neighbor(chunks, i)
                
                if best_neighbor_idx is not None:
                    # Fusionner
                    merged_chunk = self._merge_two_chunks(
                        current_chunk,
                        chunks[best_neighbor_idx]
                    )
                    
                    # Vérifier que la taille est acceptable
                    merged_size = len(merged_chunk['content'])
                    if merged_size <= self.max_chunk_size:
                        merged.append(merged_chunk)
                        # Marquer comme traité
                        chunks[best_neighbor_idx]['_processed'] = True
                        i += 1
                    else:
                        # Trop grand, garder séparé
                        merged.append(current_chunk)
                else:
                    merged.append(current_chunk)
            else:
                # Chunk de bonne taille, le garder
                if not current_chunk.get('_processed', False):
                    merged.append(current_chunk)
            
            i += 1
        
        return merged
    
    def _find_best_neighbor(
        self,
        chunks: List[Dict],
        current_idx: int
    ) -> Optional[int]:
        """
        Trouve le meilleur voisin à fusionner.
        
        Args:
            chunks: Liste de chunks
            current_idx: Index du chunk actuel
            
        Returns:
            Index du meilleur voisin ou None
        """
        # Stratégie séquentielle : fusionner avec le suivant
        if self.merge_strategy == "sequential":
            if current_idx + 1 < len(chunks):
                next_chunk = chunks[current_idx + 1]
                if not next_chunk.get('_processed', False):
                    # Vérifier que la fusion ne dépasse pas max_chunk_size
                    current_content = chunks[current_idx].get('content', '')
                    next_content = next_chunk.get('content', '')
                    merged_size = len(current_content + '\n\n' + next_content)
                    if merged_size <= self.max_chunk_size:
                        return current_idx + 1
        
        # TODO: Implémenter stratégie sémantique (utiliser embeddings)
        # TODO: Implémenter stratégie hybride
        
        return None
    
    def _merge_two_chunks(
        self,
        chunk1: Dict,
        chunk2: Dict
    ) -> Dict:
        """
        Fusionne deux chunks.
        
        Args:
            chunk1: Premier chunk
            chunk2: Deuxième chunk
            
        Returns:
            Chunk fusionné
        """
        content1 = chunk1.get('content', '')
        content2 = chunk2.get('content', '')
        merged_content = content1 + '\n\n' + content2
        
        # Fusionner les métadonnées
        merged_metadata = chunk1['metadata'].copy()
        
        # Mettre à jour les métadonnées
        merged_metadata.update({
            'merged': True,
            'merged_from': [
                chunk1['metadata'].get('chunk_index'),
                chunk2['metadata'].get('chunk_index')
            ],
            'chunk_size': len(merged_content),
            'original_chunk_sizes': [
                len(content1),
                len(content2)
            ]
        })
        
        # Garder les meilleures métadonnées
        if 'section_title' in chunk2['metadata']:
            merged_metadata['section_title'] = chunk2['metadata']['section_title']
        if 'section_level' in chunk2['metadata']:
            merged_metadata['section_level'] = chunk2['metadata']['section_level']
        
        return {
            'content': merged_content,
            'metadata': merged_metadata
        }
