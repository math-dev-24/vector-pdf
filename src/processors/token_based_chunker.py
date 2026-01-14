"""
Chunker basé sur les tokens pour une meilleure optimisation des embeddings.
Les modèles d'embedding fonctionnent avec des tokens, pas des caractères.
"""

import tiktoken
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core import get_logger

logger = get_logger(__name__)


class TokenBasedChunker:
    """Chunker qui utilise les tokens au lieu des caractères."""
    
    def __init__(
        self,
        chunk_size_tokens: int = 500,  # ~500 tokens = optimal pour embeddings
        chunk_overlap_tokens: int = 50,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialise le chunker basé sur tokens.
        
        Args:
            chunk_size_tokens: Taille en tokens (défaut: 500)
            chunk_overlap_tokens: Overlap en tokens (défaut: 50)
            model: Modèle pour encoder (défaut: text-embedding-3-small)
        """
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback si le modèle n'est pas reconnu
            logger.warning(f"Modèle {model} non reconnu, utilisation de cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Découpe le texte en chunks basés sur les tokens.
        
        Args:
            text: Texte à chunker
            
        Returns:
            Liste de chunks
        """
        if not text.strip():
            return []
        
        # Encoder en tokens
        tokens = self.encoding.encode(text)
        
        if not tokens:
            return []
        
        chunks = []
        i = 0
        
        while i < len(tokens):
            # Prendre un chunk
            chunk_tokens = tokens[i:i + self.chunk_size_tokens]
            
            # Décoder en texte
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Avancer avec overlap
            i += self.chunk_size_tokens - self.chunk_overlap_tokens
            
            # Éviter les boucles infinies
            if i >= len(tokens):
                break
        
        return chunks
    
    def get_token_count(self, text: str) -> int:
        """
        Retourne le nombre de tokens d'un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Nombre de tokens
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def rechunk_with_tokens(
        self,
        chunks: List[Dict],
        max_tokens_per_chunk: int = 500
    ) -> List[Dict]:
        """
        Re-chunk une liste de chunks existants selon les tokens.
        
        Args:
            chunks: Liste de chunks existants
            max_tokens_per_chunk: Nombre maximum de tokens par chunk
            
        Returns:
            Nouveaux chunks basés sur tokens
        """
        rechunked = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            token_count = self.get_token_count(content)
            
            # Si le chunk est déjà de bonne taille, le garder
            if token_count <= max_tokens_per_chunk:
                # Ajouter le token_count aux métadonnées
                chunk['metadata']['token_count'] = token_count
                rechunked.append(chunk)
            else:
                # Re-chunker ce chunk trop grand
                sub_chunks = self.chunk_text(content)
                for i, sub_chunk in enumerate(sub_chunks):
                    new_chunk = {
                        'content': sub_chunk,
                        'metadata': chunk['metadata'].copy()
                    }
                    new_chunk['metadata']['token_count'] = self.get_token_count(sub_chunk)
                    new_chunk['metadata']['chunk_index'] = f"{chunk['metadata'].get('chunk_index', 0)}-{i}"
                    rechunked.append(new_chunk)
        
        return rechunked
