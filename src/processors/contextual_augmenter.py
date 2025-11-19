"""
Module pour augmenter les chunks avec du contexte additionnel.
Améliore drastiquement la qualité de la recherche vectorielle.
"""

from typing import List, Dict


class ContextualAugmenter:
    """Ajoute du contexte sémantique aux chunks."""

    def __init__(
        self,
        add_document_title: bool = True,
        add_section_hierarchy: bool = True,
        add_metadata_header: bool = True,
        separator_style: str = "markdown"  # "markdown" ou "plain"
    ):
        """
        Initialise l'augmenteur contextuel.

        Args:
            add_document_title: Ajouter le nom du document
            add_section_hierarchy: Ajouter la hiérarchie de sections
            add_metadata_header: Ajouter un header avec métadonnées
            separator_style: Style de séparation ("markdown" ou "plain")
        """
        self.add_document_title = add_document_title
        self.add_section_hierarchy = add_section_hierarchy
        self.add_metadata_header = add_metadata_header
        self.separator_style = separator_style

    def create_context_header(self, metadata: Dict) -> str:
        """
        Crée un header contextuel à partir des métadonnées.

        Args:
            metadata: Métadonnées du chunk

        Returns:
            Header formaté
        """
        header_parts = []

        # Nom du document
        if self.add_document_title and metadata.get('file_name'):
            doc_name = metadata['file_name'].replace('.md', '').replace('_', ' ')
            header_parts.append(f"Document: {doc_name}")

        # Hiérarchie de sections
        if self.add_section_hierarchy and metadata.get('section_hierarchy_string'):
            header_parts.append(f"Section: {metadata['section_hierarchy_string']}")

        # Type de document (si détecté par IA)
        if metadata.get('document_type') and metadata['document_type'] != 'unknown':
            header_parts.append(f"Type: {metadata['document_type']}")

        # Sujets principaux (si détectés par IA)
        if metadata.get('topics'):
            topics_str = ', '.join(metadata['topics'][:3])  # Max 3 topics
            header_parts.append(f"Sujets: {topics_str}")

        if not header_parts:
            return ""

        # Formater selon le style
        if self.separator_style == "markdown":
            header = "---\n" + "\n".join(header_parts) + "\n---\n\n"
        else:
            header = "[" + " | ".join(header_parts) + "]\n\n"

        return header

    def augment_chunk(self, chunk: Dict, preserve_original: bool = True) -> Dict:
        """
        Augmente un chunk avec contexte.

        Args:
            chunk: Chunk avec 'content' et 'metadata'
            preserve_original: Garder le contenu original dans les métadonnées

        Returns:
            Chunk augmenté
        """
        content = chunk['content']
        metadata = chunk['metadata'].copy()

        # Sauvegarder l'original
        if preserve_original and 'original_content' not in metadata:
            metadata['original_content'] = content

        # Créer le header contextuel
        context_header = self.create_context_header(metadata)

        # Augmenter le contenu
        augmented_content = context_header + content

        return {
            'content': augmented_content,
            'metadata': metadata
        }

    def augment_batch(
        self,
        chunks: List[Dict],
        preserve_original: bool = True
    ) -> List[Dict]:
        """
        Augmente un lot de chunks.

        Args:
            chunks: Liste de chunks
            preserve_original: Garder contenu original

        Returns:
            Chunks augmentés
        """
        return [self.augment_chunk(chunk, preserve_original) for chunk in chunks]

    def create_embedding_optimized_text(self, chunk: Dict) -> str:
        """
        Crée une version du texte optimisée pour l'embedding.
        Combine contenu + métadonnées clés pour améliorer la recherche.

        Args:
            chunk: Chunk avec métadonnées

        Returns:
            Texte optimisé pour embedding
        """
        content = chunk['content']
        metadata = chunk['metadata']

        # Éléments à inclure
        elements = []

        # 1. Hiérarchie de sections
        if metadata.get('section_hierarchy'):
            hierarchy = ' > '.join(metadata['section_hierarchy'])
            elements.append(f"[Section: {hierarchy}]")

        # 2. Mots-clés (IA ou basiques)
        keywords = metadata.get('keywords_ai') or metadata.get('keywords', [])
        if keywords:
            kw_str = ', '.join(keywords[:5])
            elements.append(f"[Mots-clés: {kw_str}]")

        # 3. Sujets
        if metadata.get('topics'):
            topics_str = ', '.join(metadata['topics'][:3])
            elements.append(f"[Sujets: {topics_str}]")

        # 4. Type de document
        if metadata.get('document_type') and metadata['document_type'] != 'unknown':
            elements.append(f"[Type: {metadata['document_type']}]")

        # 5. Résumé (si disponible)
        if metadata.get('summary'):
            elements.append(f"[Résumé: {metadata['summary']}]")

        # Assembler
        prefix = ' '.join(elements)
        if prefix:
            return f"{prefix}\n\n{content}"
        else:
            return content


class HybridTextGenerator:
    """Génère différentes versions du texte pour embedding hybride."""

    @staticmethod
    def generate_variants(chunk: Dict) -> Dict[str, str]:
        """
        Génère plusieurs variantes du texte pour différentes stratégies.

        Args:
            chunk: Chunk avec métadonnées

        Returns:
            Dictionnaire {strategie: texte}
        """
        content = chunk['content']
        metadata = chunk['metadata']

        variants = {
            'raw': content,  # Texte brut
            'with_context': '',  # Avec contexte
            'keywords_focused': '',  # Focus mots-clés
            'semantic_enriched': ''  # Enrichi sémantiquement
        }

        # Variante avec contexte
        augmenter = ContextualAugmenter()
        augmented = augmenter.augment_chunk(chunk)
        variants['with_context'] = augmented['content']

        # Variante focus mots-clés
        keywords = metadata.get('keywords_ai') or metadata.get('keywords', [])
        if keywords:
            kw_text = "Mots-clés: " + ", ".join(keywords) + "\n\n" + content
            variants['keywords_focused'] = kw_text
        else:
            variants['keywords_focused'] = content

        # Variante enrichie sémantiquement
        semantic_parts = []

        # Ajouter résumé si disponible
        if metadata.get('summary'):
            semantic_parts.append(f"Résumé: {metadata['summary']}")

        # Ajouter section
        if metadata.get('section_hierarchy_string'):
            semantic_parts.append(f"Section: {metadata['section_hierarchy_string']}")

        # Ajouter sujets
        if metadata.get('topics'):
            semantic_parts.append(f"Sujets: {', '.join(metadata['topics'])}")

        if semantic_parts:
            variants['semantic_enriched'] = '\n'.join(semantic_parts) + '\n\n' + content
        else:
            variants['semantic_enriched'] = content

        return variants


def augment_all_chunks(
    chunking_results: List[Dict],
    strategy: str = "with_context",
    preserve_original: bool = True,
    verbose: bool = True
) -> List[Dict]:
    """
    Augmente tous les chunks avec contexte.

    Args:
        chunking_results: Résultats du chunking
        strategy: Stratégie d'augmentation ("with_context", "embedding_optimized", "hybrid")
        preserve_original: Garder le contenu original
        verbose: Afficher progression

    Returns:
        Résultats avec chunks augmentés
    """
    if verbose:
        print("\n✨ Augmentation contextuelle des chunks...")
        print(f"   Stratégie: {strategy}")

    augmenter = ContextualAugmenter()
    augmented_results = []

    for result in chunking_results:
        if verbose:
            print(f"\n  📄 {result['file_name']}")

        if strategy == "with_context":
            # Ajouter header contextuel
            augmented_chunks = augmenter.augment_batch(
                chunks=result['chunks'],
                preserve_original=preserve_original
            )

        elif strategy == "embedding_optimized":
            # Créer version optimisée pour embeddings
            augmented_chunks = []
            for chunk in result['chunks']:
                optimized_text = augmenter.create_embedding_optimized_text(chunk)
                metadata = chunk['metadata'].copy()
                if preserve_original:
                    metadata['original_content'] = chunk['content']

                augmented_chunks.append({
                    'content': optimized_text,
                    'metadata': metadata
                })

        elif strategy == "hybrid":
            # Générer toutes les variantes
            augmented_chunks = []
            for chunk in result['chunks']:
                variants = HybridTextGenerator.generate_variants(chunk)
                metadata = chunk['metadata'].copy()
                metadata['text_variants'] = variants
                if preserve_original:
                    metadata['original_content'] = chunk['content']

                # Utiliser la variante enrichie par défaut
                augmented_chunks.append({
                    'content': variants['semantic_enriched'],
                    'metadata': metadata
                })

        else:
            # Pas d'augmentation
            augmented_chunks = result['chunks']

        if verbose:
            print(f"    ✅ {len(augmented_chunks)} chunks augmentés")

        augmented_results.append({
            'file_path': result['file_path'],
            'file_name': result['file_name'],
            'num_chunks': result['num_chunks'],
            'total_chars': result['total_chars'],
            'chunks': augmented_chunks
        })

    if verbose:
        total_chunks = sum(len(r['chunks']) for r in augmented_results)
        print(f"\n✅ Augmentation terminée: {total_chunks} chunks traités")

    return augmented_results


if __name__ == "__main__":
    # Test
    test_chunk = {
        'content': "Les résultats financiers du Q4 2024 montrent une croissance significative du chiffre d'affaires.",
        'metadata': {
            'file_name': 'rapport_financier_2024.md',
            'section_hierarchy': ['Résultats', 'Analyse financière', 'Q4 2024'],
            'section_hierarchy_string': 'Résultats > Analyse financière > Q4 2024',
            'keywords': ['résultats', 'financiers', 'croissance', 'chiffre affaires'],
            'topics': ['finance', 'performance', 'entreprise'],
            'document_type': 'rapport'
        }
    }

    augmenter = ContextualAugmenter()

    print("=" * 60)
    print("CHUNK ORIGINAL")
    print("=" * 60)
    print(test_chunk['content'])

    print("\n" + "=" * 60)
    print("CHUNK AUGMENTÉ")
    print("=" * 60)
    augmented = augmenter.augment_chunk(test_chunk)
    print(augmented['content'])

    print("\n" + "=" * 60)
    print("CHUNK OPTIMISÉ POUR EMBEDDING")
    print("=" * 60)
    optimized = augmenter.create_embedding_optimized_text(test_chunk)
    print(optimized)

    print("\n" + "=" * 60)
    print("VARIANTES HYBRIDES")
    print("=" * 60)
    variants = HybridTextGenerator.generate_variants(test_chunk)
    for name, text in variants.items():
        print(f"\n--- {name.upper()} ---")
        print(text[:200] + "..." if len(text) > 200 else text)