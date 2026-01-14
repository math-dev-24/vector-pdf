"""
Filtre les chunks de faible qualité avant vectorisation.
Évite de vectoriser du contenu inutile ou de mauvaise qualité.
"""

import re
from typing import List, Dict, Tuple

from src.core import get_logger

logger = get_logger(__name__)


class ChunkQualityFilter:
    """Filtre les chunks selon leur qualité."""
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 2000,
        min_words: int = 10,
        filter_noise: bool = True
    ):
        """
        Initialise le filtre.
        
        Args:
            min_length: Longueur minimale (caractères)
            max_length: Longueur maximale (caractères)
            min_words: Nombre minimum de mots
            filter_noise: Filtrer le bruit (caractères spéciaux, etc.)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.filter_noise = filter_noise
    
    def score_chunk(self, chunk: Dict) -> float:
        """
        Score la qualité d'un chunk (0-1).
        
        Args:
            chunk: Chunk avec 'content' et 'metadata'
            
        Returns:
            Score de qualité (0-1)
        """
        content = chunk.get('content', '')
        
        if not content:
            return 0.0
        
        # Score de base
        score = 1.0
        
        # Pénalités pour longueur
        if len(content) < self.min_length:
            score *= 0.3
        elif len(content) > self.max_length:
            score *= 0.7
        
        # Nombre de mots
        words = content.split()
        word_count = len(words)
        if word_count < self.min_words:
            score *= 0.5
        
        # Ratio caractères spéciaux
        if self.filter_noise:
            special_chars = len(re.findall(r'[^\w\s]', content))
            char_ratio = special_chars / len(content) if content else 0
            if char_ratio > 0.3:  # Trop de caractères spéciaux
                score *= 0.4
        
        # Ratio majuscules (trop = probablement un titre/header)
        upper_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        if upper_ratio > 0.5 and len(content) < 200:
            score *= 0.6  # Probablement juste un titre
        
        # Contenu répétitif
        if word_count > 0:
            unique_ratio = len(set(words)) / word_count
            if unique_ratio < 0.3:  # Trop de répétition
                score *= 0.3
        
        # Détecter les patterns de bruit
        # Lignes vides excessives
        empty_lines = content.count('\n\n\n')
        if empty_lines > 3:
            score *= 0.7
        
        # Caractères non-ASCII excessifs (sauf si c'est normal)
        non_ascii_ratio = sum(1 for c in content if ord(c) > 127) / len(content) if content else 0
        if non_ascii_ratio > 0.5 and len(content) < 100:
            # Peut être normal pour du français, mais suspect si très court
            score *= 0.8
        
        return min(1.0, max(0.0, score))
    
    def filter_chunks(
        self,
        chunks: List[Dict],
        min_quality: float = 0.5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filtre les chunks selon leur qualité.
        
        Args:
            chunks: Liste de chunks
            min_quality: Score minimum pour garder un chunk
            
        Returns:
            Tuple (chunks gardés, chunks filtrés)
        """
        kept_chunks = []
        filtered_chunks = []
        
        for chunk in chunks:
            score = self.score_chunk(chunk)
            chunk['metadata']['quality_score'] = score
            
            if score >= min_quality:
                kept_chunks.append(chunk)
            else:
                filtered_chunks.append(chunk)
                logger.debug(
                    f"Chunk filtré (score: {score:.2f}): "
                    f"{chunk.get('content', '')[:50]}..."
                )
        
        return kept_chunks, filtered_chunks
