"""
Module pour stocker et gérer les embeddings dans Pinecone.
"""

import os
import time
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec


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
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY non définie dans .env")

        self.pc = Pinecone(api_key=api_key)

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
                    cloud="aws",
                    region="us-east-1"  # Région gratuite
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

    def add_chunks(self, chunks: List[Dict], batch_size: int = 100, namespace: str = "", verbose: bool = True) -> None:
        """
        Ajoute des chunks avec leurs embeddings à Pinecone.

        Args:
            chunks: Liste de dicts avec 'content', 'metadata', 'embedding'
            batch_size: Taille des batchs pour l'upload
            namespace: Namespace Pinecone (optionnel, "" = default)
            verbose: Afficher la progression
        """
        if verbose:
            print(f"\n=== Ajout dans Pinecone ===")
            print(f"Index: {self.index_name}")
            print(f"Namespace: {namespace if namespace else '(default)'}")
            print(f"Nombre de chunks: {len(chunks)}\n")

        # Préparer les données pour Pinecone
        vectors = []

        for i, chunk in enumerate(chunks):
            # ID unique basé sur le fichier et l'index
            vector_id = f"{chunk['metadata']['file_name']}_chunk_{chunk['metadata']['chunk_index']}"

            # Métadonnées (Pinecone accepte strings, numbers, booleans, lists)
            metadata = {
                'source': chunk['metadata']['source'],
                'file_name': chunk['metadata']['file_name'],
                'chunk_index': chunk['metadata']['chunk_index'],
                'total_chunks': chunk['metadata']['total_chunks'],
                'chunk_size': chunk['metadata']['chunk_size'],
                'text': chunk['content'][:1000]  # Limiter à 1000 chars pour les métadonnées
            }

            vectors.append({
                'id': vector_id,
                'values': chunk['embedding'],
                'metadata': metadata
            })

        # Upload par batchs
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)

            if verbose:
                print(f"  Batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} uploadé")

        if verbose:
            # Attendre que les stats se mettent à jour
            time.sleep(2)
            stats = self.index.describe_index_stats()
            print(f"\n✓ {len(chunks)} chunks ajoutés à Pinecone")
            if namespace:
                ns_count = stats.get('namespaces', {}).get(namespace, {}).get('vector_count', 0)
                print(f"  Total dans le namespace '{namespace}': {ns_count}")
            print(f"  Total dans l'index: {stats['total_vector_count']}")

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
    batch_size: int = 100
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

    # Ajouter à Pinecone
    vector_store.add_chunks(all_chunks, batch_size=batch_size, namespace=namespace)

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
