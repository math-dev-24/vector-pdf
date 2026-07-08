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
    namespace_strategy: str = Field(
        default="by_ai",
        description="Stratégie de namespace: none (unique), by_file (par fichier), by_folder (par dossier), by_ai (classification IA)"
    )
    namespace_prefix: str = Field(default="", description="Préfixe optionnel pour les namespaces auto-générés")
    namespace_definitions: str = Field(
        default="",
        description=(
            'Namespaces personnalisés au format JSON. '
            'Exemple: {"Mon label": {"id": "mon_id", "description": "..."}, ...}. '
            'Si vide, utilise les 3 namespaces par défaut (depannage/dimensionnement/general).'
        )
    )

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
    chunk_size: int = Field(default=1000, ge=100, description="Taille cible des chunks en caractères")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap entre chunks")
    chunk_min_size: int = Field(default=400, ge=50, description="Taille minimale avant fusion (caractères)")
    chunk_max_size: int = Field(default=2000, ge=500, description="Taille maximale d'un chunk (caractères)")
    chunk_max_tokens: int = Field(default=600, ge=100, description="Taille maximale d'un chunk (tokens embedding)")
    use_semantic_chunking: bool = Field(default=True, description="Chunking par sections (idéal docs techniques)")
    use_token_based_chunking: bool = Field(default=True, description="Utiliser chunking basé sur tokens")
    filter_chunk_quality: bool = Field(default=True, description="Filtrer les chunks de faible qualité")
    merge_small_chunks: bool = Field(default=True, description="Fusionner les petits chunks")
    min_chunk_quality: float = Field(default=0.5, ge=0.0, le=1.0, description="Score minimum de qualité")
    prioritize_chunks: bool = Field(default=True, description="Prioriser les chunks importants")
    chunk_merge_strategy: str = Field(
        default="hybrid",
        description="Stratégie de fusion des petits chunks: sequential, semantic, hybrid",
    )
    use_sentence_window_chunking: bool = Field(
        default=True,
        description="Fenêtre de phrases pour manuels de dépannage/procédures",
    )

    # ===== Enrichissement IA =====
    enable_ai_enrichment: bool = Field(default=True, description="Enrichissement métadonnées via GPT")
    ai_enrichment_batch_size: int = Field(
        default=15, ge=1, le=30, description="Chunks par appel GPT pour l'enrichissement",
    )
    enable_boundary_fallback: bool = Field(
        default=False,
        description="Fallback LLM pour découpage si structure de sections insuffisante",
    )
    boundary_fallback_section_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Score structure minimum avant fallback LLM (sections valides / attendues)",
    )

    # ===== Embedding / retrieval =====
    embed_with_context: bool = Field(
        default=True,
        description="Inclure hiérarchie sectionnelle dans le texte embeddé",
    )
    max_workers: Optional[int] = Field(default=None, description="Nombre max de workers (None = auto)")
    use_multithreading: bool = Field(default=False, description="Activer multithreading pour extraction PDF")
    enable_async: bool = Field(default=True, description="Utiliser async pour les APIs (expérimental)")
    smart_batching: bool = Field(default=True, description="Batch processing intelligent basé sur tokens")

    # ===== Extraction PDF =====
    pdf_margin_top: float = Field(default=72.0, description="Marge haute (pt) pour ignorer en-têtes de page")
    pdf_margin_bottom: float = Field(default=72.0, description="Marge basse (pt) pour ignorer pieds de page")
    pdf_table_strategy: str = Field(
        default="lines_strict",
        description="Stratégie de détection des tableaux pymupdf4llm (lines_strict, lines, text)",
    )

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
