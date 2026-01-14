"""
Classe principale du pipeline - Orchestration des services.
"""

from typing import Optional

from src.core import settings, get_logger, PipelineError, ErrorType
from src.processors import StateManager

from .models import (
    PipelineConfig,
    PipelineResult,
    ExtractionResult,
    ChunkingResult,
    EmbeddingResult,
    StorageResult
)
from .services import (
    ExtractionService,
    ChunkingService,
    EmbeddingService,
    StorageService
)

logger = get_logger(__name__)


class Pipeline:
    """
    Pipeline principal pour le traitement de documents.
    Orchestre les différents services.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        state_manager: Optional[StateManager] = None
    ):
        """
        Initialise le pipeline.
        
        Args:
            config: Configuration du pipeline
            state_manager: Gestionnaire d'état (optionnel)
        """
        self.config = config or PipelineConfig(
            data_dir=settings.data_dir,
            output_dir=settings.output_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedding_model=settings.embedding_model,
            embedding_batch_size=settings.embedding_batch_size
        )
        
        self.state_manager = state_manager or StateManager(str(settings.cache_dir))
        
        # Initialiser les services
        self.extraction_service = ExtractionService(verbose=self.config.verbose)
        self.chunking_service = ChunkingService(verbose=self.config.verbose)
        self.embedding_service = EmbeddingService(verbose=self.config.verbose)
        self.storage_service = StorageService(verbose=self.config.verbose)
    
    def run_full(
        self,
        save_intermediate: bool = True
    ) -> PipelineResult:
        """
        Exécute le pipeline complet (extraction → chunking → embedding → storage).
        
        Args:
            save_intermediate: Sauvegarder les résultats intermédiaires dans le cache
            
        Returns:
            Résultat complet du pipeline
        """
        result = PipelineResult()
        
        try:
            # Étape 1: Extraction
            result.extraction = self.extract()
            if not result.extraction.output_files:
                result.error = "Aucun fichier extrait"
                return result
            
            # Étape 2: Chunking
            result.chunking = self.chunk()
            if not result.chunking.chunks:
                result.error = "Aucun chunk créé"
                return result
            
            if save_intermediate:
                # Sauvegarder les chunks
                chunking_results = [{
                    'file_path': 'combined',
                    'file_name': 'combined',
                    'num_chunks': result.chunking.total_chunks,
                    'total_chars': result.chunking.total_chars,
                    'chunks': result.chunking.chunks
                }]
                self.state_manager.save_chunks(chunking_results)
            
            # Étape 3: Embedding
            result.embedding = self.embed(result.chunking.chunks)
            if not result.embedding.enriched_chunks:
                result.error = "Aucun embedding créé"
                return result
            
            if save_intermediate:
                # Sauvegarder les embeddings
                embedding_results = [{
                    'file_path': 'combined',
                    'file_name': 'combined',
                    'num_chunks': result.embedding.total_embeddings,
                    'total_chars': result.chunking.total_chars,
                    'chunks': result.embedding.enriched_chunks
                }]
                self.state_manager.save_embeddings(
                    embedding_results,
                    self.config.namespace
                )
            
            # Étape 4: Storage
            result.storage = self.store(
                result.embedding.enriched_chunks,
                reset=self.config.reset_namespace
            )
            
            result.success = True
            logger.info("Pipeline terminé avec succès")
            
        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.error(f"Erreur dans le pipeline: {e}", exc_info=True)
            raise PipelineError(
                ErrorType.UNKNOWN,
                f"Erreur dans le pipeline: {e}",
                original_error=e
            )
        
        return result
    
    def extract(self) -> ExtractionResult:
        """Exécute uniquement l'extraction."""
        return self.extraction_service.extract(
            data_dir=self.config.data_dir,
            output_dir=self.config.output_dir,
            extraction_mode=self.config.extraction_mode,
            pdf_filter=self.config.pdf_filter
        )
    
    def chunk(self) -> ChunkingResult:
        """Exécute uniquement le chunking."""
        return self.chunking_service.chunk(
            output_dir=self.config.output_dir,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunking_mode=self.config.chunking_mode
        )
    
    def embed(self, chunks: list) -> EmbeddingResult:
        """Exécute uniquement la vectorisation."""
        return self.embedding_service.embed(
            chunks_data=chunks,
            model=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size
        )
    
    def store(
        self,
        enriched_chunks: list,
        reset: bool = False
    ) -> StorageResult:
        """Exécute uniquement le stockage."""
        return self.storage_service.store(
            enriched_chunks=enriched_chunks,
            namespace=self.config.namespace,
            reset=reset
        )
