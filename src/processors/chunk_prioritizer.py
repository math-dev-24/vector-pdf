"""
Priorise les chunks selon leur importance.
Permet de traiter les chunks les plus importants en premier.
"""

from typing import List, Dict

from src.core import get_logger

logger = get_logger(__name__)


class ChunkPrioritizer:
    """Priorise les chunks selon leur importance."""
    
    def prioritize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Ajoute un score de priorité aux chunks.
        
        Args:
            chunks: Liste de chunks
            
        Returns:
            Chunks avec score de priorité
        """
        for chunk in chunks:
            priority = self._calculate_priority(chunk)
            chunk['metadata']['priority_score'] = priority
        
        return chunks
    
    def _calculate_priority(self, chunk: Dict) -> float:
        """
        Calcule le score de priorité (0-1).
        
        Args:
            chunk: Chunk avec 'content' et 'metadata'
            
        Returns:
            Score de priorité (0-1)
        """
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        score = 0.5  # Base
        
        # Boost si contient un titre
        if metadata.get('section_title'):
            score += 0.2
        
        # Boost si niveau de section élevé (H1, H2)
        section_level = metadata.get('section_level', 99)
        if section_level <= 2:
            score += 0.15
        
        # Boost si contient des mots-clés importants
        keywords = [
            'résumé', 'conclusion', 'introduction', 'important', 
            'attention', 'note', 'remarque', 'définition'
        ]
        content_lower = content.lower()
        if any(kw in content_lower for kw in keywords):
            score += 0.1
        
        # Boost si longueur optimale (ni trop court ni trop long)
        length = len(content)
        if 200 <= length <= 1000:
            score += 0.1
        
        # Boost si qualité élevée
        quality_score = metadata.get('quality_score', 0.5)
        score += quality_score * 0.1
        
        # Boost si contient des chiffres/statistiques (probablement important)
        if any(char.isdigit() for char in content):
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def sort_by_priority(self, chunks: List[Dict]) -> List[Dict]:
        """
        Trie les chunks par priorité (décroissant).
        
        Args:
            chunks: Liste de chunks
            
        Returns:
            Chunks triés par priorité
        """
        # S'assurer que les priorités sont calculées
        chunks = self.prioritize_chunks(chunks)
        
        # Trier par priorité décroissante
        return sorted(
            chunks,
            key=lambda x: x['metadata'].get('priority_score', 0.5),
            reverse=True
        )
