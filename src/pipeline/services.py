"""
Services métier pour le pipeline (logique pure, sans I/O utilisateur).
"""

from pathlib import Path
from typing import List, Dict, Optional

from src.core import settings, get_logger, PipelineError, ErrorType
from src.pdf_analyzer import analyze_pdfs
from src.extractors import extract_text_from_pdf, extract_text_from_scan
from src.extractors.text_extractor_v2 import (
    extract_structured_text_from_pdf,
    extract_with_pymupdf4llm,
    process_multiple_pdfs
)
from src.extractors.mistral_ocr_extractor import extract_text_with_mistral_ocr
from src.processors import chunk_all_markdown_files, process_all_markdown_files
from src.vectorization.embeddings import embed_all_files
from src.vectorization.vector_store import store_embeddings, VectorStore

from .models import (
    ExtractionMode,
    PDFFilter,
    ChunkingMode,
    ExtractionResult,
    ChunkingResult,
    EmbeddingResult,
    StorageResult
)
from .interfaces import (
    IExtractionService,
    IChunkingService,
    IEmbeddingService,
    IStorageService
)

logger = get_logger(__name__)


class ExtractionService(IExtractionService):
    """Service d'extraction de PDFs."""
    
    EXTRACTION_FUNCTIONS = {
        ExtractionMode.BASIC: extract_text_from_pdf,
        ExtractionMode.STRUCTURED: extract_structured_text_from_pdf,
        ExtractionMode.PYMUPDF4LLM: extract_with_pymupdf4llm,
        ExtractionMode.MISTRAL_OCR: extract_text_with_mistral_ocr,
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def extract(
        self,
        data_dir: Path,
        output_dir: Path,
        extraction_mode: ExtractionMode = ExtractionMode.STRUCTURED,
        pdf_filter: PDFFilter = PDFFilter.ALL
    ) -> ExtractionResult:
        """
        Extrait le texte des PDFs.
        
        Args:
            data_dir: Répertoire contenant les PDFs
            output_dir: Répertoire de sortie
            extraction_mode: Mode d'extraction
            pdf_filter: Filtre pour les PDFs
            
        Returns:
            Résultat de l'extraction
        """
        if self.verbose:
            logger.info("Début de l'extraction des PDFs")
            logger.info(f"Mode: {extraction_mode.value}, Filtre: {pdf_filter.value}")
        
        # Analyser les PDFs
        pdf_infos = analyze_pdfs(str(data_dir))
        
        if not pdf_infos:
            logger.warning(f"Aucun PDF trouvé dans {data_dir}")
            return ExtractionResult()
        
        # Séparer scans et textes natifs
        all_text_pdfs = [p for p in pdf_infos if p['page_type'] == 'text']
        all_scan_pdfs = [p for p in pdf_infos if p['page_type'] == 'scan']
        
        # Appliquer le filtre
        text_pdfs, scan_pdfs = self._apply_filter(all_text_pdfs, all_scan_pdfs, pdf_filter)
        
        output_files = []
        total_pages = 0
        
        # Extraire les PDFs natifs
        if text_pdfs:
            output_files.extend(self._extract_text_pdfs(
                text_pdfs,
                output_dir,
                extraction_mode
            ))
            total_pages += sum(p['num_pages'] for p in text_pdfs)
        
        # Extraire les scans
        if scan_pdfs:
            scan_files = self._extract_scan_pdfs(scan_pdfs, output_dir)
            output_files.extend(scan_files)
            total_pages += sum(p['num_pages'] for p in scan_pdfs)
        
        if self.verbose:
            logger.info(f"Extraction terminée: {len(output_files)} fichiers créés")
        
        return ExtractionResult(
            output_files=output_files,
            text_pdfs_count=len(text_pdfs),
            scan_pdfs_count=len(scan_pdfs),
            total_pages=total_pages
        )
    
    def _apply_filter(
        self,
        text_pdfs: List[Dict],
        scan_pdfs: List[Dict],
        pdf_filter: PDFFilter
    ) -> tuple[List[Dict], List[Dict]]:
        """Applique le filtre aux PDFs."""
        if pdf_filter == PDFFilter.TEXT:
            return text_pdfs, []
        elif pdf_filter == PDFFilter.SCAN:
            return [], scan_pdfs
        else:  # ALL
            return text_pdfs, scan_pdfs
    
    def _extract_text_pdfs(
        self,
        text_pdfs: List[Dict],
        output_dir: Path,
        extraction_mode: ExtractionMode
    ) -> List[str]:
        """Extrait les PDFs avec texte natif."""
        extraction_func = self.EXTRACTION_FUNCTIONS.get(
            extraction_mode,
            extract_structured_text_from_pdf
        )
        
        if len(text_pdfs) > 1:
            pdf_paths = [pdf['path'] for pdf in text_pdfs]
            return process_multiple_pdfs(
                pdf_paths=pdf_paths,
                output_dir=str(output_dir),
                extraction_func=extraction_func,
                verbose=self.verbose,
                use_multithreading=settings.use_multithreading
            )
        else:
            pdf = text_pdfs[0]
            try:
                output_path = extraction_func(pdf['path'], str(output_dir), verbose=self.verbose)
                return [output_path]
            except Exception as e:
                raise PipelineError(
                    ErrorType.PDF_EXTRACTION,
                    f"Erreur lors de l'extraction de {pdf['path']}: {e}",
                    original_error=e
                )
    
    def _extract_scan_pdfs(
        self,
        scan_pdfs: List[Dict],
        output_dir: Path
    ) -> List[str]:
        """
        Extrait les PDFs scannés (OCR).
        Utilise Mistral OCR si configuré, sinon Tesseract.
        Fallback automatique vers Tesseract si Mistral échoue.
        """
        output_files = []
        use_mistral = settings.use_mistral_ocr
        fallback_enabled = settings.mistral_ocr_fallback
        
        for pdf in scan_pdfs:
            success = False
            
            # Essayer Mistral OCR si configuré
            if use_mistral:
                try:
                    if self.verbose:
                        logger.info(f"Extraction Mistral OCR: {pdf['path']}")
                    output_path = extract_text_with_mistral_ocr(
                        pdf['path'],
                        str(output_dir),
                        language="fr",
                        verbose=self.verbose
                    )
                    output_files.append(output_path)
                    success = True
                except Exception as e:
                    logger.warning(f"Erreur Mistral OCR pour {pdf['path']}: {e}")
                    if fallback_enabled:
                        if self.verbose:
                            logger.info(f"Fallback vers Tesseract pour {pdf['path']}")
                    else:
                        # Pas de fallback, on continue avec l'erreur
                        logger.error(f"Erreur OCR (Mistral) pour {pdf['path']}: {e}")
            
            # Utiliser Tesseract si Mistral n'est pas configuré ou si fallback activé
            if not success:
                try:
                    if not use_mistral and self.verbose:
                        logger.info(f"Extraction Tesseract: {pdf['path']}")
                    output_path = extract_text_from_scan(
                        pdf['path'],
                        str(output_dir),
                        verbose=self.verbose
                    )
                    output_files.append(output_path)
                    success = True
                except Exception as e:
                    logger.error(f"Erreur OCR (Tesseract) pour {pdf['path']}: {e}")
            
            if not success:
                logger.error(f"Échec de l'extraction OCR pour {pdf['path']} (Mistral et Tesseract)")
        
        return output_files


class ChunkingService(IChunkingService):
    """Service de chunking des documents avec optimisations."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        # Importer les nouveaux modules
        from src.processors.token_based_chunker import TokenBasedChunker
        from src.processors.chunk_quality_filter import ChunkQualityFilter
        from src.processors.chunk_merger import ChunkMerger
        from src.processors.chunk_prioritizer import ChunkPrioritizer
        
        self.token_chunker = TokenBasedChunker()
        self.quality_filter = ChunkQualityFilter()
        self.merger = ChunkMerger()
        self.prioritizer = ChunkPrioritizer()
    
    def chunk(
        self,
        output_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
        chunking_mode: ChunkingMode = ChunkingMode.STANDARD,
        use_token_based: bool = None,
        filter_quality: bool = None,
        merge_small: bool = None
    ) -> ChunkingResult:
        """
        Découpe les documents en chunks avec optimisations.
        
        Args:
            output_dir: Répertoire contenant les markdowns
            chunk_size: Taille des chunks
            chunk_overlap: Overlap entre chunks
            chunking_mode: Mode de chunking
            use_token_based: Utiliser chunking basé sur tokens (défaut: settings)
            filter_quality: Filtrer les chunks de faible qualité (défaut: settings)
            merge_small: Fusionner les petits chunks (défaut: settings)
            
        Returns:
            Résultat du chunking
        """
        # Utiliser les settings si non spécifié
        use_token_based = use_token_based if use_token_based is not None else settings.use_token_based_chunking
        filter_quality = filter_quality if filter_quality is not None else settings.filter_chunk_quality
        merge_small = merge_small if merge_small is not None else settings.merge_small_chunks
        
        if self.verbose:
            logger.info(f"Début du chunking (mode: {chunking_mode.value})")
            logger.info(f"Optimisations: token_based={use_token_based}, filter={filter_quality}, merge={merge_small}")
        
        if chunking_mode == ChunkingMode.ADVANCED:
            results = process_all_markdown_files(
                directory=str(output_dir),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_adaptive_chunking=True,
                use_semantic_chunking=False,
                enable_ai_enrichment=True,
                enable_context_augmentation=True,
                augmentation_strategy="with_context",
                verbose=self.verbose
            )
        else:
            results = chunk_all_markdown_files(
                directory=str(output_dir),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                verbose=self.verbose
            )
        
        if not results:
            logger.warning("Aucun document à chunker")
            return ChunkingResult()
        
        # Collecter tous les chunks
        all_chunks = []
        total_chars = 0
        for result in results:
            all_chunks.extend(result['chunks'])
            total_chars += result.get('total_chars', 0)
        
        initial_count = len(all_chunks)
        
        # OPTIMISATIONS
        
        # 1. Ajouter token_count aux métadonnées
        if use_token_based:
            if self.verbose:
                logger.info("Calcul des tokens pour chaque chunk...")
            for chunk in all_chunks:
                content = chunk.get('content', '')
                token_count = self.token_chunker.get_token_count(content)
                chunk['metadata']['token_count'] = token_count
        
        # 2. Fusionner les petits chunks
        if merge_small:
            if self.verbose:
                logger.info("Fusion des petits chunks...")
            all_chunks = self.merger.merge_chunks(all_chunks)
            if self.verbose:
                logger.info(f"  {initial_count} → {len(all_chunks)} chunks après fusion")
        
        # 3. Filtrer les chunks de faible qualité
        filtered_count = 0
        if filter_quality:
            if self.verbose:
                logger.info("Filtrage de la qualité des chunks...")
            all_chunks, filtered = self.quality_filter.filter_chunks(
                all_chunks,
                min_quality=settings.min_chunk_quality
            )
            filtered_count = len(filtered)
            if self.verbose and filtered:
                logger.info(f"  {filtered_count} chunks filtrés (qualité < {settings.min_chunk_quality})")
        
        # 4. Prioriser les chunks
        if settings.prioritize_chunks:
            if self.verbose:
                logger.info("Priorisation des chunks...")
            all_chunks = self.prioritizer.prioritize_chunks(all_chunks)
        
        # Recalculer les totaux
        total_chars = sum(len(c.get('content', '')) for c in all_chunks)
        
        if self.verbose:
            logger.info(f"Chunking terminé: {len(all_chunks)} chunks créés")
            logger.info(f"  Réduction: {initial_count} → {len(all_chunks)} chunks ({((initial_count - len(all_chunks)) / initial_count * 100):.1f}%)")
            if all_chunks:
                avg_quality = sum(
                    c['metadata'].get('quality_score', 0.5) 
                    for c in all_chunks
                ) / len(all_chunks)
                logger.info(f"  Qualité moyenne: {avg_quality:.2f}")
        
        return ChunkingResult(
            chunks=all_chunks,
            total_chunks=len(all_chunks),
            total_chars=total_chars
        )


class EmbeddingService(IEmbeddingService):
    """Service de création d'embeddings."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def embed(
        self,
        chunks_data: List[Dict],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_smart_batching: bool = None
    ) -> EmbeddingResult:
        """
        Crée les embeddings pour les chunks.
        
        Args:
            chunks_data: Liste de chunks avec métadonnées
            model: Modèle d'embedding
            batch_size: Taille des batchs
            use_smart_batching: Utiliser smart batching (défaut: settings)
            
        Returns:
            Résultat de la vectorisation
        """
        if self.verbose:
            logger.info("Début de la vectorisation")
        
        model = model or settings.embedding_model
        batch_size = batch_size or settings.embedding_batch_size
        use_smart_batching = use_smart_batching if use_smart_batching is not None else settings.smart_batching
        
        # Réorganiser en format attendu par embed_all_files
        results = [{
            'file_path': 'combined',
            'file_name': 'combined',
            'num_chunks': len(chunks_data),
            'total_chars': sum(len(c.get('content', '')) for c in chunks_data),
            'chunks': chunks_data
        }]
        
        enriched_results = embed_all_files(
            results=results,
            model=model,
            batch_size=batch_size,
            verbose=self.verbose,
            use_smart_batching=use_smart_batching
        )
        
        if not enriched_results:
            logger.warning("Aucun embedding créé")
            return EmbeddingResult()
        
        enriched_chunks = enriched_results[0]['chunks']
        dimension = len(enriched_chunks[0]['embedding']) if enriched_chunks else 0
        
        if self.verbose:
            logger.info(f"Vectorisation terminée: {len(enriched_chunks)} embeddings créés")
        
        return EmbeddingResult(
            enriched_chunks=enriched_chunks,
            total_embeddings=len(enriched_chunks),
            dimension=dimension
        )


class StorageService(IStorageService):
    """Service de stockage dans Pinecone."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def store(
        self,
        enriched_chunks: List[Dict],
        namespace: str = "",
        reset: bool = False
    ) -> StorageResult:
        """
        Stocke les embeddings dans Pinecone.
        
        Args:
            enriched_chunks: Chunks avec embeddings
            namespace: Namespace Pinecone
            reset: Réinitialiser le namespace
            
        Returns:
            Résultat du stockage
        """
        if self.verbose:
            logger.info(f"Stockage dans Pinecone (namespace: {namespace or 'default'})")
        
        # Réorganiser en format attendu par store_embeddings
        enriched_results = [{
            'file_path': 'combined',
            'file_name': 'combined',
            'num_chunks': len(enriched_chunks),
            'total_chars': sum(len(c.get('content', '')) for c in enriched_chunks),
            'chunks': enriched_chunks
        }]
        
        vector_store = store_embeddings(
            enriched_results=enriched_results,
            index_name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
            namespace=namespace,
            reset=reset,
            embedding_version=settings.embedding_version
        )
        
        stats = vector_store.get_stats()
        
        if self.verbose:
            logger.info(f"Stockage terminé: {stats['total_vectors']} vecteurs")
        
        return StorageResult(
            vector_store=vector_store,
            total_vectors=stats['total_vectors'],
            namespace=namespace
        )
