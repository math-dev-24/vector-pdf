"""
Chunker avancé qui orchestre tous les modules d'amélioration:
- Nettoyage avancé du texte
- Détection de structure
- Chunking adaptatif
- Enrichissement de métadonnées
- Augmentation contextuelle
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core import ProgressBar, get_logger

from .text_cleaner import clean_markdown_extraction
from .section_detector import SectionDetector
from .chunking_strategies import AdaptiveChunker, SemanticChunker, ContentTypeDetector
from .metadata_enricher import MetadataEnricher
from .contextual_augmenter import ContextualAugmenter

logger = get_logger(__name__)


class AdvancedChunker:
    """
    Chunker avancé avec toutes les optimisations.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_adaptive_chunking: bool = True,
        use_semantic_chunking: bool = False,
        enable_ai_enrichment: bool = True,
        enable_context_augmentation: bool = True,
        augmentation_strategy: str = "with_context"  # "with_context", "embedding_optimized", "hybrid"
    ):
        """
        Initialise le chunker avancé.

        Args:
            chunk_size: Taille par défaut des chunks
            chunk_overlap: Overlap entre chunks
            use_adaptive_chunking: Adapter le chunking au type de contenu
            use_semantic_chunking: Utiliser le chunking sémantique par sections
            enable_ai_enrichment: Activer l'enrichissement IA (nécessite OpenAI API)
            enable_context_augmentation: Ajouter contexte aux chunks
            augmentation_strategy: Stratégie d'augmentation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_adaptive_chunking = use_adaptive_chunking
        self.use_semantic_chunking = use_semantic_chunking
        self.enable_ai_enrichment = enable_ai_enrichment
        self.enable_context_augmentation = enable_context_augmentation
        self.augmentation_strategy = augmentation_strategy

        from src.core import settings

        self.adaptive_chunker = (
            AdaptiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if use_adaptive_chunking else None
        )
        self.semantic_chunker = (
            SemanticChunker(
                min_chunk_size=settings.chunk_min_size,
                max_chunk_size=settings.chunk_max_size,
                chunk_overlap=chunk_overlap,
            )
            if use_semantic_chunking else None
        )
        self.section_detector = SectionDetector()
        self.metadata_enricher = MetadataEnricher(use_ai=enable_ai_enrichment)
        self.contextual_augmenter = ContextualAugmenter()

    def process_markdown_file(
        self,
        file_path: str,
        verbose: bool = False
    ) -> Dict:
        """
        Traite un fichier markdown avec le pipeline complet.

        Args:
            file_path: Chemin du fichier markdown
            verbose: Afficher progression

        Returns:
            Dictionnaire avec chunks et métadonnées
        """
        if verbose:
            print(f"\n📄 Traitement de {os.path.basename(file_path)}")

        # 1. Lire et nettoyer le texte
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        original_length = len(text)

        # Nettoyage léger (l'extraction PDF a déjà nettoyé le markdown)
        is_ocr = 'scan' in file_path.lower() or '_ocr' in file_path.lower()
        text = clean_markdown_extraction(text, is_ocr=is_ocr)

        cleaned_length = len(text)
        reduction_pct = 100 * (1 - cleaned_length / original_length) if original_length > 0 else 0

        if reduction_pct > 50:
            logger.warning(
                f"{os.path.basename(file_path)}: nettoyage agressif — "
                f"{reduction_pct:.1f}% du texte supprimé ({original_length} → {cleaned_length} chars). "
                "Vérifier les filtres de text_cleaner."
            )

        if verbose:
            print(f"  🧹 Nettoyage: {reduction_pct:.1f}% de réduction")

        if cleaned_length < 50:
            if verbose:
                print(f"  ⚠️  Texte quasi-vide après nettoyage ({cleaned_length} chars), fichier ignoré")
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'num_chunks': 0,
                'total_chars': 0,
                'chunks': [],
                'sections': [],
                'original_length': original_length,
                'cleaned_length': cleaned_length
            }

        # 2. Détecter la structure (sections)
        self.section_detector.parse_document(text)

        if verbose and self.section_detector.sections:
            print(f"  📑 Structure: {len(self.section_detector.sections)} sections détectées")

        # 3. Chunking adaptatif ou sémantique
        if self.use_semantic_chunking and self.semantic_chunker:
            # Chunking par sections
            section_chunks = self.semantic_chunker.chunk_by_sections(text)
            chunks_text = [content for _, content in section_chunks]
            chunks_titles = [title for title, _ in section_chunks]

            if verbose:
                print(f"  ✂️  Chunking sémantique: {len(chunks_text)} sections")

        elif self.use_adaptive_chunking and self.adaptive_chunker:
            # Détecter le type de contenu
            content_type = ContentTypeDetector.detect_content_type(text)
            chunks_text = self.adaptive_chunker.chunk_text(text, content_type)
            chunks_titles = [None] * len(chunks_text)

            if verbose:
                print(f"  ✂️  Chunking adaptatif ({content_type}): {len(chunks_text)} chunks")

        else:
            # Chunking standard (fallback)
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""]
            )
            chunks_text = splitter.split_text(text)
            chunks_titles = [None] * len(chunks_text)

            if verbose:
                print(f"  ✂️  Chunking standard: {len(chunks_text)} chunks")

        # 4. Créer chunks avec métadonnées de base
        chunks_with_metadata = []
        search_start = 0  # Fix #2: recherche réelle dans le texte nettoyé

        for i, chunk_text in enumerate(chunks_text):
            # Trouver la position réelle du chunk dans le texte nettoyé
            lookup = chunk_text[:min(80, len(chunk_text))]
            pos = text.find(lookup, search_start)
            if pos != -1:
                char_position = pos
                # Avancer d'au moins 1 char pour que le prochain chunk puisse se trouver après
                search_start = pos + max(1, len(chunk_text) - self.chunk_overlap)
            else:
                char_position = search_start

            # Trouver la section correspondante
            section = self.section_detector.get_section_for_text_position(text, char_position)

            base_metadata = {
                'source': file_path,
                'file_name': os.path.basename(file_path),
                'chunk_index': i,
                'total_chunks': len(chunks_text),
                'chunk_size': len(chunk_text)
            }

            # Ajouter métadonnées de section
            if section:
                base_metadata['section_title'] = section.title
                base_metadata['section_level'] = section.level
                base_metadata['section_hierarchy'] = section.get_hierarchy_path()
                base_metadata['section_hierarchy_string'] = section.get_hierarchy_string()

            # Ajouter titre personnalisé (chunking sémantique)
            if chunks_titles[i]:
                base_metadata['chunk_title'] = chunks_titles[i]

            chunks_with_metadata.append({
                'content': chunk_text,
                'metadata': base_metadata
            })

        # 5. Enrichissement de métadonnées (IA + basique)
        if verbose:
            print(f"  🧠 Enrichissement de métadonnées...")

        chunks_with_metadata = self.metadata_enricher.enrich_batch(
            chunks=chunks_with_metadata,
            use_ai=self.enable_ai_enrichment,
            verbose=verbose,
        )

        # 6. Augmentation contextuelle
        # Fix #3 : le contenu brut reste dans 'content' (pour l'embedding),
        # la version augmentée va dans metadata['display_content'] (pour l'affichage).
        if self.enable_context_augmentation:
            if verbose:
                print(f"  ✨ Augmentation contextuelle ({self.augmentation_strategy})...")

            augment_progress = ProgressBar(
                len(chunks_with_metadata),
                prefix="Augmentation",
                enabled=verbose,
            )
            for i, chunk in enumerate(chunks_with_metadata):
                original_content = chunk['content']

                if self.augmentation_strategy == "embedding_optimized":
                    # Stocker la version enrichie pour l'affichage seulement
                    optimized_text = self.contextual_augmenter.create_embedding_optimized_text(chunk)
                    chunk['metadata']['display_content'] = optimized_text
                    chunks_with_metadata[i] = chunk
                else:
                    # with_context ou hybrid : construire le header contextuel
                    augmented = self.contextual_augmenter.augment_chunk(chunk, preserve_original=False)
                    # Stocker l'augmenté pour l'affichage, restaurer l'original pour l'embedding
                    augmented['metadata']['display_content'] = augmented['content']
                    augmented['content'] = original_content
                    chunks_with_metadata[i] = augmented

                augment_progress.update(i + 1)
            augment_progress.finish("✓")

        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'num_chunks': len(chunks_with_metadata),
            'total_chars': len(text),
            'chunks': chunks_with_metadata,
            'sections': self.section_detector.sections,
            'original_length': original_length,
            'cleaned_length': len(text)
        }

        if verbose:
            avg_quality = sum(c['metadata'].get('chunk_quality_score', 0) for c in chunks_with_metadata) / len(chunks_with_metadata) if chunks_with_metadata else 0
            print(f"  ✅ Terminé: {len(chunks_with_metadata)} chunks (qualité moyenne: {avg_quality:.2f})")

        return result


def process_all_markdown_files(
    directory: str = "./OUTPUT",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_adaptive_chunking: bool = True,
    use_semantic_chunking: bool = False,
    enable_ai_enrichment: bool = True,
    enable_context_augmentation: bool = True,
    augmentation_strategy: str = "with_context",
    verbose: bool = True,
    max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Traite tous les fichiers markdown d'un répertoire avec le pipeline avancé.

    Args:
        directory: Répertoire contenant les markdowns
        chunk_size: Taille des chunks
        chunk_overlap: Overlap
        use_adaptive_chunking: Chunking adaptatif
        use_semantic_chunking: Chunking sémantique
        enable_ai_enrichment: Enrichissement IA
        enable_context_augmentation: Augmentation contextuelle
        augmentation_strategy: Stratégie d'augmentation
        verbose: Afficher progression
        max_workers: Threads (None = auto)

    Returns:
        Liste des résultats
    """
    output_dir = Path(directory)

    if not output_dir.exists():
        print(f"❌ Le répertoire {directory} n'existe pas.")
        return []

    md_files = list(output_dir.glob("*.md"))

    if not md_files:
        print(f"❌ Aucun fichier .md trouvé dans {directory}")
        return []

    if verbose:
        print("\n" + "=" * 70)
        print("🚀 CHUNKING AVANCÉ")
        print("=" * 70)
        print(f"\n📁 Répertoire: {directory}")
        print(f"📄 Fichiers: {len(md_files)}")
        print(f"\n⚙️  Configuration:")
        print(f"  - Chunking: {'Sémantique' if use_semantic_chunking else 'Adaptatif' if use_adaptive_chunking else 'Standard'}")
        print(f"  - Taille: {chunk_size} caractères (overlap: {chunk_overlap})")
        print(f"  - Enrichissement IA: {'✅' if enable_ai_enrichment else '❌'}")
        print(f"  - Augmentation: {augmentation_strategy if enable_context_augmentation else 'Désactivée'}")

    # Créer le chunker
    chunker = AdvancedChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_adaptive_chunking=use_adaptive_chunking,
        use_semantic_chunking=use_semantic_chunking,
        enable_ai_enrichment=enable_ai_enrichment,
        enable_context_augmentation=enable_context_augmentation,
        augmentation_strategy=augmentation_strategy
    )

    results = []

    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    file_progress = ProgressBar(len(md_files), prefix="Chunking avancé", enabled=verbose)

    if enable_ai_enrichment or len(md_files) == 1:
        for i, md_file in enumerate(md_files):
            if verbose:
                print(f"\n→ {md_file.name}")
            try:
                result = chunker.process_markdown_file(str(md_file), verbose=verbose)
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Erreur: {e}")
            file_progress.update(i + 1)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(chunker.process_markdown_file, str(md_file), False): md_file
                for md_file in md_files
            }

            for i, future in enumerate(as_completed(future_to_file), 1):
                md_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  ❌ {md_file.name}: {e}")
                file_progress.update(i)

    file_progress.finish("✓")

    if verbose:
        total_chunks = sum(r['num_chunks'] for r in results)
        total_sections = sum(len(r.get('sections', [])) for r in results)
        avg_quality = sum(
            sum(c['metadata'].get('chunk_quality_score', 0) for c in r['chunks']) / len(r['chunks'])
            for r in results if r['chunks']
        ) / len(results) if results else 0

        print("\n" + "=" * 70)
        print("✅ TRAITEMENT TERMINÉ")
        print("=" * 70)
        print(f"\n📊 Statistiques:")
        print(f"  - Fichiers traités: {len(results)}")
        print(f"  - Chunks générés: {total_chunks}")
        print(f"  - Sections détectées: {total_sections}")
        print(f"  - Qualité moyenne: {avg_quality:.2f}/1.00")
        print()

    return results


if __name__ == "__main__":
    # Test du chunker avancé
    import sys

    # Vérifier les arguments
    directory = sys.argv[1] if len(sys.argv) > 1 else "./OUTPUT"

    results = process_all_markdown_files(
        directory=directory,
        chunk_size=1000,
        chunk_overlap=200,
        use_adaptive_chunking=True,
        use_semantic_chunking=False,
        enable_ai_enrichment=True,  # Activer enrichissement IA
        enable_context_augmentation=True,
        augmentation_strategy="with_context",
        verbose=True
    )

    # Afficher un exemple de chunk enrichi
    if results and results[0]['chunks']:
        print("\n" + "=" * 70)
        print("📝 EXEMPLE DE CHUNK ENRICHI")
        print("=" * 70)

        example_chunk = results[0]['chunks'][0]

        print("\n**CONTENU:**")
        print(example_chunk['content'][:300] + "..." if len(example_chunk['content']) > 300 else example_chunk['content'])

        print("\n**MÉTADONNÉES:**")
        import json
        # Afficher seulement les métadonnées pertinentes
        relevant_meta = {k: v for k, v in example_chunk['metadata'].items() if k != 'original_content'}
        print(json.dumps(relevant_meta, indent=2, ensure_ascii=False))