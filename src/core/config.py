"""
Configuration centralisée de l'application avec validation Pydantic.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)
else:
    load_dotenv(override=False)


class Settings(BaseSettings):
    """Configuration centralisée de l'application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Dossiers =====
    data_dir: Path = Field(default=Path("./DATA"), description="Répertoire contenant les PDFs")
    output_dir: Path = Field(default=Path("./OUTPUT"), description="Répertoire de sortie pour les markdown")
    cache_dir: Path = Field(default=Path("./.cache"), description="Répertoire de cache")

    # ===== Pinecone =====
    pinecone_index_name: str = Field(default="pdf-documents", description="Nom de l'index Pinecone")
    pinecone_dimension: int = Field(default=1536, description="Dimension des vecteurs")
    pinecone_api_key: Optional[str] = Field(default=None, description="Clé API Pinecone")
    pinecone_region: str = Field(default="us-east-1", description="Région Pinecone")
    pinecone_cloud: str = Field(default="aws", description="Cloud provider Pinecone")

    # ===== OpenAI =====
    openai_api_key: Optional[str] = Field(default=None, description="Clé API OpenAI")
    embedding_model: str = Field(default="text-embedding-3-small", description="Modèle d'embedding")
    embedding_version: str = Field(default="1.0", description="Version des embeddings")
    embedding_batch_size: int = Field(default=100, ge=1, le=100, description="Taille des batchs pour embeddings")
    embedding_cache_enabled: bool = Field(default=True, description="Activer le cache des embeddings")

    # ===== Mistral AI =====
    mistral_api_key: Optional[str] = Field(default=None, description="Clé API Mistral")
    use_mistral_ocr: bool = Field(default=False, description="Utiliser Mistral OCR pour les scans")
    mistral_ocr_fallback: bool = Field(default=True, description="Fallback vers Tesseract si Mistral échoue")

    # ===== Chunking =====
    chunk_size: int = Field(default=1000, ge=100, description="Taille des chunks en caractères")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap entre chunks")
    use_semantic_chunking: bool = Field(default=False, description="Chunking par sections sémantiques (idéal pour docs longs)")
    use_token_based_chunking: bool = Field(default=True, description="Utiliser chunking basé sur tokens")
    filter_chunk_quality: bool = Field(default=True, description="Filtrer les chunks de faible qualité")
    merge_small_chunks: bool = Field(default=True, description="Fusionner les petits chunks")
    min_chunk_quality: float = Field(default=0.5, ge=0.0, le=1.0, description="Score minimum de qualité")
    prioritize_chunks: bool = Field(default=True, description="Prioriser les chunks importants")
    
    # ===== Performance =====
    max_workers: Optional[int] = Field(default=None, description="Nombre max de workers (None = auto)")
    use_multithreading: bool = Field(default=False, description="Activer multithreading pour extraction PDF")
    enable_async: bool = Field(default=True, description="Utiliser async pour les APIs (expérimental)")
    smart_batching: bool = Field(default=True, description="Batch processing intelligent basé sur tokens")

    # ===== Logging =====
    log_level: str = Field(default="INFO", description="Niveau de logging")
    log_file: Optional[Path] = Field(default=None, description="Fichier de log (None = console uniquement)")

    @field_validator("data_dir", "output_dir", "cache_dir", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Convertit les strings en Path."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("log_file", mode="before")
    @classmethod
    def validate_log_file(cls, v):
        """Convertit les strings en Path pour log_file."""
        if isinstance(v, str):
            return Path(v)
        return v

    def __init__(self, **kwargs):
        """Initialise la configuration et crée les dossiers nécessaires."""
        super().__init__(**kwargs)
        # Créer les dossiers s'ils n'existent pas
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_api_keys(self) -> tuple[bool, list[str]]:
        """
        Valide que les clés API nécessaires sont présentes.
        
        Returns:
            Tuple (is_valid, missing_keys)
        """
        missing = []
        
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        
        return len(missing) == 0, missing


# Instance globale de configuration
settings = Settings()
