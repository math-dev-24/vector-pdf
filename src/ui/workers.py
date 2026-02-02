"""
Workers pour exécuter le pipeline en arrière-plan sans bloquer l'interface.
"""

from PySide6.QtCore import QObject, QThread, Signal
from typing import Optional, Any

from src.core import settings, get_logger
from src.processors import StateManager
from src.pipeline import Pipeline, PipelineConfig
from src.pipeline.models import (
    ExtractionMode,
    PDFFilter,
    ChunkingMode,
)

logger = get_logger(__name__)


class PipelineWorker(QObject):
    """Worker pour exécuter les opérations du pipeline."""

    finished = Signal(bool, str)  # success, message
    progress = Signal(str)

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def _emit_progress(self, msg: str) -> None:
        self.progress.emit(msg)

    def run_extraction(
        self,
        extraction_mode: ExtractionMode,
        pdf_filter: PDFFilter,
    ) -> None:
        """Exécute l'extraction PDF vers Markdown."""
        try:
            self._emit_progress("Configuration de l'extraction...")
            config = PipelineConfig(
                data_dir=settings.data_dir,
                output_dir=settings.output_dir,
                extraction_mode=extraction_mode,
                pdf_filter=pdf_filter,
            )
            pipeline = Pipeline(config=config, state_manager=self.state_manager)

            self._emit_progress("Extraction en cours...")
            result = pipeline.extract()

            if result.output_files:
                msg = (
                    f"Extraction terminée: {len(result.output_files)} fichier(s) créé(s)\n"
                    f"PDFs texte: {result.text_pdfs_count} | "
                    f"PDFs scannés: {result.scan_pdfs_count} | "
                    f"Total pages: {result.total_pages}"
                )
                self.finished.emit(True, msg)
            else:
                self.finished.emit(False, "Aucun fichier extrait")
        except Exception as e:
            logger.error(f"Erreur extraction: {e}", exc_info=True)
            self.finished.emit(False, str(e))

    def run_vectorization(
        self,
        namespace: str,
        chunking_mode: ChunkingMode,
        use_cached_chunks: bool,
        use_cached_embeddings: bool,
    ) -> None:
        """Exécute la vectorisation (chunking + embeddings)."""
        try:
            enriched_results = None

            if use_cached_embeddings and self.state_manager.has_embeddings(namespace):
                self._emit_progress("Chargement des embeddings en cache...")
                enriched_results = self.state_manager.load_embeddings(namespace)
                if enriched_results:
                    self.finished.emit(True, f"Embeddings chargés (namespace: {namespace or 'default'})")
                    return

            results = None
            if use_cached_chunks and self.state_manager.has_chunks():
                self._emit_progress("Chargement des chunks en cache...")
                results = self.state_manager.load_chunks()

            if not results:
                self._emit_progress("Chunking en cours...")
                config = PipelineConfig(
                    data_dir=settings.data_dir,
                    output_dir=settings.output_dir,
                    chunking_mode=chunking_mode,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                pipeline = Pipeline(config=config, state_manager=self.state_manager)
                chunking_result = pipeline.chunk()

                if not chunking_result.chunks:
                    self.finished.emit(
                        False,
                        "Aucun document markdown trouvé. Lancez d'abord l'extraction.",
                    )
                    return

                results = [
                    {
                        "file_path": "combined",
                        "file_name": "combined",
                        "num_chunks": chunking_result.total_chunks,
                        "total_chars": chunking_result.total_chars,
                        "chunks": chunking_result.chunks,
                    }
                ]
                self.state_manager.save_chunks(results)

            self._emit_progress("Vectorisation en cours...")
            config = PipelineConfig(
                data_dir=settings.data_dir,
                output_dir=settings.output_dir,
                embedding_model=settings.embedding_model,
                embedding_batch_size=settings.embedding_batch_size,
            )
            pipeline = Pipeline(config=config, state_manager=self.state_manager)
            embedding_result = pipeline.embed(results[0]["chunks"])

            if embedding_result.enriched_chunks:
                enriched_results = [
                    {
                        "file_path": "combined",
                        "file_name": "combined",
                        "num_chunks": embedding_result.total_embeddings,
                        "total_chars": results[0]["total_chars"],
                        "chunks": embedding_result.enriched_chunks,
                    }
                ]
                self.state_manager.save_embeddings(enriched_results, namespace)
                self.finished.emit(
                    True,
                    f"Vectorisation terminée: {embedding_result.total_embeddings} embeddings",
                )
            else:
                self.finished.emit(False, "Échec de la vectorisation")
        except Exception as e:
            logger.error(f"Erreur vectorisation: {e}", exc_info=True)
            self.finished.emit(False, str(e))

    def run_store(
        self,
        enriched_results: list,
        namespace: str,
        reset: bool,
    ) -> None:
        """Exécute le stockage dans Pinecone."""
        try:
            self._emit_progress("Stockage dans Pinecone...")
            config = PipelineConfig(
                data_dir=settings.data_dir,
                output_dir=settings.output_dir,
                namespace=namespace,
                reset_namespace=reset,
            )
            pipeline = Pipeline(config=config, state_manager=self.state_manager)
            chunks = enriched_results[0]["chunks"] if enriched_results else []
            storage_result = pipeline.store(chunks, reset=reset)

            if storage_result.vector_store:
                stats = storage_result.vector_store.get_stats()
                msg = (
                    f"Stockage terminé!\n"
                    f"Index: {stats['index_name']} | "
                    f"Vecteurs: {stats['total_vectors']} | "
                    f"Dimension: {stats['dimension']}"
                )
                self.finished.emit(True, msg)
            else:
                self.finished.emit(True, "Stockage terminé")
        except Exception as e:
            logger.error(f"Erreur stockage: {e}", exc_info=True)
            self.finished.emit(False, str(e))

    def run_full_pipeline(
        self,
        namespace: str,
        extraction_mode: ExtractionMode,
        pdf_filter: PDFFilter,
        chunking_mode: ChunkingMode,
        reset: bool,
    ) -> None:
        """Exécute le pipeline complet."""
        try:
            self._emit_progress("Démarrage du pipeline complet...")
            config = PipelineConfig(
                data_dir=settings.data_dir,
                output_dir=settings.output_dir,
                extraction_mode=extraction_mode,
                pdf_filter=pdf_filter,
                chunking_mode=chunking_mode,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                embedding_model=settings.embedding_model,
                embedding_batch_size=settings.embedding_batch_size,
                namespace=namespace,
                reset_namespace=reset,
            )
            pipeline = Pipeline(config=config, state_manager=self.state_manager)
            result = pipeline.run_full()

            if result.success and result.storage and result.storage.vector_store:
                stats = result.storage.vector_store.get_stats()
                msg = (
                    f"Pipeline terminé avec succès!\n"
                    f"Index: {stats['index_name']} | "
                    f"Vecteurs: {stats['total_vectors']}"
                )
                self.finished.emit(True, msg)
            elif result.success:
                self.finished.emit(True, "Pipeline terminé avec succès!")
            else:
                self.finished.emit(False, result.error or "Erreur inconnue")
        except Exception as e:
            logger.error(f"Erreur pipeline: {e}", exc_info=True)
            self.finished.emit(False, str(e))
