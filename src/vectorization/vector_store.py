"""
Module pour stocker et gérer les embeddings dans Pinecone.
"""

import os
import time
import unicodedata
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec

from src.core import settings, get_logger, ConfigurationError, ProgressBar

logger = get_logger(__name__)


def build_pinecone_metadata(chunk: Dict, embedding_version: Optional[str] = None) -> Dict:
    """
    Construit les métadonnées Pinecone à partir d'un chunk enrichi.
    Pinecone accepte strings, numbers, booleans et listes de strings.
    """
    meta = chunk.get('metadata', {})
    content = chunk.get('content', '')
    display = meta.get('display_content') or content

    metadata = {
        'source': meta.get('source', ''),
        'file_name': meta.get('file_name', ''),
        'chunk_index': meta.get('chunk_index', 0),
        'total_chunks': meta.get('total_chunks', 0),
        'chunk_size': meta.get('chunk_size', len(content)),
        'text': content[:8000],
        'display_text': display[:8000],
        'embedding_dimension': len(chunk.get('embedding', [])),
    }

    if meta.get('section_hierarchy_string'):
        metadata['section_hierarchy'] = meta['section_hierarchy_string'][:500]

    if meta.get('section_title'):
        metadata['section_title'] = str(meta['section_title'])[:200]

    if meta.get('summary'):
        metadata['summary'] = str(meta['summary'])[:500]

    if meta.get('document_type') and meta['document_type'] != 'unknown':
        metadata['document_type'] = str(meta['document_type'])[:50]

    topics = meta.get('topics') or []
    if topics:
        metadata['topics'] = [str(t)[:80] for t in topics[:5]]

    keywords = meta.get('keywords_ai') or meta.get('keywords') or []
    if keywords:
        metadata['keywords'] = [str(k)[:60] for k in keywords[:10]]

    domain_tags = meta.get('domain_tags') or []
    if domain_tags:
        metadata['domain_tags'] = [str(tag)[:60] for tag in domain_tags[:8]]

    if meta.get('rag_label'):
        metadata['rag_label'] = str(meta['rag_label'])[:80]

    if meta.get('rag_label_confidence') is not None:
        metadata['rag_label_confidence'] = float(meta['rag_label_confidence'])

    if meta.get('quality_score') is not None:
        metadata['quality_score'] = float(meta['quality_score'])

    if meta.get('chunk_quality_score') is not None:
        metadata['chunk_quality_score'] = float(meta['chunk_quality_score'])

    if embedding_version:
        metadata['embedding_version'] = embedding_version

    return metadata


def sanitize_vector_id(text: str) -> str:
    """
    Nettoie un texte pour l'utiliser comme ID Pinecone (ASCII uniquement).

    Args:
        text: Texte à nettoyer

    Returns:
        Texte nettoyé en ASCII
    """
    # Normaliser les caractères unicode (décomposer les accents)
    text = unicodedata.normalize('NFKD', text)
    # Encoder en ASCII en ignorant les caractères non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remplacer les espaces et caractères spéciaux par des underscores
    text = re.sub(r'[^a-zA-Z0-9_\-.]', '_', text)
    # Éviter les underscores multiples
    text = re.sub(r'_+', '_', text)
    # Enlever les underscores au début et à la fin
    text = text.strip('_')

    return text


def _compute_namespace(chunk: Dict, strategy: str, prefix: str = "") -> str:
    """
    Calcule le namespace Pinecone pour un chunk selon la stratégie.

    Args:
        chunk: Dict avec 'metadata' (file_name, source)
        strategy: "by_file", "by_folder" ou "none"
        prefix: Préfixe optionnel

    Returns:
        Namespace sanitisé (str vide = namespace default Pinecone)
    """
    if strategy == "by_file":
        file_name = chunk.get('metadata', {}).get('file_name', 'unknown')
        ns = sanitize_vector_id(Path(file_name).stem)
    elif strategy == "by_folder":
        source = chunk.get('metadata', {}).get('source', '')
        folder = Path(source).parent.name if source else 'unknown'
        ns = sanitize_vector_id(folder) if folder else 'unknown'
    else:
        # NONE : namespace unique = le préfixe (ou default si vide)
        return prefix

    return f"{prefix}_{ns}" if prefix else ns


class VectorStore:
    """Gestionnaire de base de données vectorielle avec Pinecone."""

    def __init__(
        self,
        index_name: str = "pdf-documents",
        dimension: int = 1536,  # text-embedding-3-small
        metric: str = "cosine"
    ):
        """
        Initialise le store vectoriel Pinecone.

        Args:
            index_name: Nom de l'index Pinecone
            dimension: Dimension des vecteurs (1536 pour text-embedding-3-small)
            metric: Métrique de similarité (cosine, euclidean, dotproduct)
        """
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

        # Initialiser le client Pinecone
        api_key = settings.pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "PINECONE_API_KEY non définie. "
                "Définissez-la dans .env ou via la variable d'environnement."
            )

        self.pc = Pinecone(api_key=api_key)
        logger.debug("Client Pinecone initialisé")

        # Créer ou récupérer l'index
        self._initialize_index()

    def _initialize_index(self):
        """Crée l'index s'il n'existe pas."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Création de l'index '{self.index_name}'...")

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region
                )
            )

            # Attendre que l'index soit prêt
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

            print(f"✓ Index '{self.index_name}' créé")
        else:
            print(f"✓ Index '{self.index_name}' existant trouvé")

        # Connecter à l'index
        self.index = self.pc.Index(self.index_name)

    def add_chunks(
        self, 
        chunks: List[Dict], 
        batch_size: int = 100, 
        namespace: str = "", 
        verbose: bool = True,
        embedding_version: Optional[str] = None
    ) -> None:
        """
        Ajoute des chunks avec leurs embeddings à Pinecone.
        Valide les dimensions avant l'upload.

        Args:
            chunks: Liste de dicts avec 'content', 'metadata', 'embedding'
            batch_size: Taille des batchs pour l'upload
            namespace: Namespace Pinecone (optionnel, "" = default)
            verbose: Afficher la progression
            embedding_version: Version des embeddings (optionnel)
        """
        # Préparer les données pour Pinecone
        vectors = []
        validation_errors = []

        for i, chunk in enumerate(chunks):
            embedding = chunk.get('embedding')
            
            # Validation de l'embedding
            if not embedding:
                validation_errors.append(f"Chunk {i}: embedding manquant")
                continue
                
            if not isinstance(embedding, list):
                validation_errors.append(f"Chunk {i}: embedding n'est pas une liste")
                continue
                
            if len(embedding) != self.dimension:
                validation_errors.append(
                    f"Chunk {i}: dimension {len(embedding)} != {self.dimension}"
                )
                continue

            # ID unique basé sur le fichier et l'index
            # Nettoyer le nom de fichier pour ne garder que des caractères ASCII
            clean_filename = sanitize_vector_id(chunk['metadata']['file_name'])
            vector_id = f"{clean_filename}_chunk_{chunk['metadata']['chunk_index']}"

            # Métadonnées enrichies pour le retrieval RAG
            metadata = build_pinecone_metadata(chunk, embedding_version=embedding_version)

            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })

        # Afficher les erreurs de validation
        if validation_errors:
            logger.error(f"{len(validation_errors)} erreur(s) de validation:")
            for error in validation_errors[:10]:  # Limiter l'affichage
                logger.error(f"  - {error}")
            if len(validation_errors) > 10:
                logger.error(f"  ... et {len(validation_errors) - 10} autres")
            
            if len(validation_errors) == len(chunks):
                from src.core import PipelineError, ErrorType
                raise PipelineError(
                    ErrorType.EMBEDDING,
                    "Tous les embeddings sont invalides"
                )

        if verbose:
            print(f"\n=== Upload Pinecone ===")
            print(f"Index: {self.index_name} | Namespace: {namespace or '(default)'} | Vecteurs: {len(vectors)}")

        upload_progress = ProgressBar(len(vectors), prefix="Pinecone", enabled=verbose)
        uploaded = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            uploaded += len(batch)
            upload_progress.update(uploaded)
        upload_progress.finish("✓")

        if verbose:
            # Attendre que les stats se mettent à jour
            time.sleep(2)
            stats = self.index.describe_index_stats()
            print(f"\n✓ {len(chunks)} chunks ajoutés à Pinecone")
            if namespace:
                ns_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
                print(f"  Total dans le namespace '{namespace}': {ns_count}")
            print(f"  Total dans l'index: {stats['total_vector_count']}")

    def add_chunks_distributed(
        self,
        chunks: List[Dict],
        strategy: str = "by_file",
        namespace_prefix: str = "",
        batch_size: int = 100,
        verbose: bool = True,
        embedding_version: Optional[str] = None,
        reset: bool = False
    ) -> Dict[str, int]:
        """
        Distribue les chunks dans des namespaces distincts selon la stratégie choisie.

        Stratégies disponibles :
        - "by_file"   : un namespace par fichier source (nom du fichier sans extension)
        - "by_folder" : un namespace par dossier parent du fichier source
        - "by_ai"     : classification OpenAI → Dépannage / Dimensionnement / Général
        - "none"      : tous les chunks dans namespace_prefix (ou default si vide)

        Args:
            chunks: Liste de dicts avec 'content', 'metadata', 'embedding'
            strategy: Stratégie de répartition
            namespace_prefix: Préfixe ajouté devant chaque namespace calculé
            batch_size: Taille des batchs pour l'upload Pinecone
            verbose: Afficher la progression
            embedding_version: Version des embeddings (optionnel)
            reset: Vider les namespaces calculés avant l'upload

        Returns:
            Dict {namespace: nombre_de_chunks_uploadés}
        """
        groups: Dict[str, List[Dict]] = defaultdict(list)

        if strategy == "by_ai":
            # Classification IA : chaque chunk reçoit son namespace individuellement
            from src.vectorization.namespace_classifier import classify_chunks
            namespaces = classify_chunks(chunks, verbose=verbose)
            for chunk, ns in zip(chunks, namespaces):
                if namespace_prefix:
                    ns = f"{namespace_prefix}_{ns}"
                groups[ns].append(chunk)
        else:
            # Stratégies mécaniques : by_file, by_folder, none
            for chunk in chunks:
                ns = _compute_namespace(chunk, strategy, namespace_prefix)
                groups[ns].append(chunk)

        if verbose:
            print(f"\n=== Distribution par namespaces ===")
            print(f"Stratégie : {strategy}")
            print(f"Namespaces : {len(groups)}")
            for ns, ns_chunks in sorted(groups.items()):
                ns_display = ns if ns else "(default)"
                print(f"  - {ns_display}: {len(ns_chunks)} chunks")
            print()

        # Upload chaque groupe dans son namespace
        counts: Dict[str, int] = {}
        ns_progress = ProgressBar(len(groups), prefix="Namespaces", enabled=verbose)
        for ns_idx, (ns, ns_chunks) in enumerate(sorted(groups.items()), 1):
            ns_display = ns if ns else "(default)"
            if reset:
                if verbose:
                    print(f"→ Reset '{ns_display}'")
                self.delete_all(namespace=ns)
            if verbose:
                print(f"→ Upload '{ns_display}' ({len(ns_chunks)} chunks)")
            self.add_chunks(
                ns_chunks,
                batch_size=batch_size,
                namespace=ns,
                verbose=verbose,
                embedding_version=embedding_version
            )
            counts[ns_display] = len(ns_chunks)
            ns_progress.update(ns_idx)
        ns_progress.finish("✓")

        if verbose:
            print(f"\n✓ {len(chunks)} chunks distribués dans {len(groups)} namespace(s)")

        return counts

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter_metadata: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> Dict:
        """
        Recherche les chunks les plus similaires.

        Args:
            query_embedding: Vecteur de la requête
            top_k: Nombre de résultats à retourner
            namespace: Namespace à interroger (optionnel, "" = default)
            filter_metadata: Filtres optionnels sur les métadonnées
            include_metadata: Inclure les métadonnées dans les résultats

        Returns:
            Résultats de la recherche
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_metadata,
            include_metadata=include_metadata
        )

        return results

    def get_stats(self) -> Dict:
        """
        Récupère les statistiques de l'index.

        Returns:
            Dictionnaire avec les stats
        """
        stats = self.index.describe_index_stats()

        return {
            'index_name': self.index_name,
            'total_vectors': stats.get('total_vector_count', 0),
            'dimension': self.dimension,
            'metric': self.metric,
            'namespaces': stats.get('namespaces', {})
        }

    def delete_all(self, namespace: str = "") -> None:
        """
        Supprime tous les vecteurs d'un namespace (utiliser avec précaution).

        Args:
            namespace: Namespace à vider (optionnel, "" = default, None = tous les namespaces)
        """
        if namespace is None:
            # Supprimer TOUS les namespaces
            self.index.delete(delete_all=True)
            print(f"✓ Tous les vecteurs de l'index '{self.index_name}' ont été supprimés")
        else:
            # Supprimer un namespace spécifique
            self.index.delete(delete_all=True, namespace=namespace)
            ns_name = namespace if namespace else "(default)"
            print(f"✓ Tous les vecteurs du namespace '{ns_name}' ont été supprimés")

    def list_namespaces(self) -> List[str]:
        """
        Liste tous les namespaces de l'index.

        Returns:
            Liste des namespaces
        """
        stats = self.index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())

        # Ajouter "(default)" s'il y a des vecteurs sans namespace
        if '' in namespaces:
            namespaces.remove('')
            namespaces.insert(0, '(default)')

        return namespaces

    def delete_index(self) -> None:
        """Supprime complètement l'index (utiliser avec précaution)."""
        self.pc.delete_index(self.index_name)
        print(f"✓ Index '{self.index_name}' supprimé")


def store_embeddings(
    enriched_results: List[Dict],
    index_name: str = "pdf-documents",
    dimension: int = 1536,
    namespace: str = "",
    reset: bool = False,
    batch_size: int = 100,
    embedding_version: Optional[str] = None
) -> VectorStore:
    """
    Stocke tous les embeddings dans Pinecone.

    Args:
        enriched_results: Résultats avec embeddings
        index_name: Nom de l'index Pinecone
        dimension: Dimension des vecteurs
        namespace: Namespace Pinecone (optionnel, "" = default)
        reset: Supprimer tous les vecteurs du namespace avant l'ajout
        batch_size: Taille des batchs pour l'upload
        embedding_version: Version des embeddings (optionnel)

    Returns:
        Instance du VectorStore
    """
    # Initialiser le store
    vector_store = VectorStore(
        index_name=index_name,
        dimension=dimension
    )

    # Reset si demandé
    if reset:
        ns_name = namespace if namespace else "(default)"
        print(f"⚠️  Reset demandé: suppression de tous les vecteurs du namespace '{ns_name}'...")
        vector_store.delete_all(namespace=namespace)

    # Collecter tous les chunks
    all_chunks = []
    for result in enriched_results:
        all_chunks.extend(result['chunks'])

    # Ajouter à Pinecone avec versioning
    vector_store.add_chunks(
        all_chunks, 
        batch_size=batch_size, 
        namespace=namespace,
        embedding_version=embedding_version
    )

    # Afficher les stats
    stats = vector_store.get_stats()
    print(f"\n=== Stats Pinecone ===")
    print(f"Index: {stats['index_name']}")
    print(f"Total vecteurs: {stats['total_vectors']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Métrique: {stats['metric']}")

    # Afficher les namespaces
    if stats['namespaces']:
        print(f"\nNamespaces:")
        for ns, ns_stats in stats['namespaces'].items():
            ns_display = ns if ns else "(default)"
            print(f"  - {ns_display}: {ns_stats.get('vector_count', 0)} vecteurs")

    return vector_store
