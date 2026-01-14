from src.core import (
    settings,
    setup_logging,
    get_logger,
    OpenAIClient,
    PipelineError,
    ErrorType
)
from src.vectorization.vector_store import VectorStore

# Configuration du logging
setup_logging(level=settings.log_level, log_file=settings.log_file)
logger = get_logger(__name__)


def query_vector_db(
    question: str,
    vector_store: VectorStore,
    namespace: str = "",
    top_k: int = 5,
    verbose: bool = True
) -> dict:
    """
    Interroge la base de données vectorielle avec une question.

    Args:
        question: Question en langage naturel
        vector_store: Instance du VectorStore
        namespace: Namespace à interroger (optionnel)
        top_k: Nombre de résultats à retourner
        verbose: Afficher les détails

    Returns:
        Dictionnaire avec les résultats
    """
    if verbose:
        logger.info("Recherche en cours...")
        logger.info(f"Question: \"{question}\"")
        logger.info(f"Namespace: {namespace if namespace else '(default)'}")
        logger.info(f"Top-K: {top_k}")

    try:
        client = OpenAIClient().client
        response = client.embeddings.create(
            input=[question],
            model=settings.embedding_model
        )
        query_embedding = response.data[0].embedding

        if verbose:
            logger.info(f"Embedding créé (dimension: {len(query_embedding)})")

        # Interroger Pinecone
        results = vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )

        return results
    except Exception as e:
        raise PipelineError(
            ErrorType.EMBEDDING,
            f"Erreur lors de la création de l'embedding de la question: {e}",
            original_error=e
        )


def display_results(results: dict, verbose: bool = True):
    """
    Affiche les résultats de la recherche de manière formatée.

    Args:
        results: Résultats de Pinecone
        verbose: Afficher tous les détails
    """
    matches = results.get('matches', [])

    if not matches:
        print("❌ Aucun résultat trouvé.\n")
        return

    print("=" * 80)
    print(f"📊 {len(matches)} RÉSULTAT(S) TROUVÉ(S)")
    print("=" * 80)

    for i, match in enumerate(matches, 1):
        score = match.get('score', 0)
        metadata = match.get('metadata', {})

        print(f"\n[{i}] Score de similarité: {score:.4f}")
        print(f"    Source: {metadata.get('file_name', 'N/A')}")
        print(f"    Chunk: {metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")

        if verbose:
            print(f"\n    Contenu:")
            print("    " + "-" * 76)
            text = metadata.get('text', 'N/A')
            # Afficher le texte avec indentation
            for line in text.split('\n'):
                print(f"    {line}")
            print("    " + "-" * 76)

    print("\n" + "=" * 80 + "\n")


def display_menu():
    """Affiche le menu principal."""
    print("\n" + "=" * 80)
    print("🤔 ASK - INTERROGER LA BASE DE DONNÉES VECTORIELLE")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Poser une question")
    print("  2. Changer de namespace")
    print("  3. Afficher les namespaces disponibles")
    print("  4. Afficher les statistiques de la DB")
    print("  0. Quitter")
    print("=" * 80)


def list_namespaces(vector_store: VectorStore):
    """Affiche la liste des namespaces disponibles."""
    print("\n" + "=" * 80)
    print("📁 NAMESPACES DISPONIBLES")
    print("=" * 80)

    stats = vector_store.get_stats()
    namespaces = stats.get('namespaces', {})

    if not namespaces:
        print("\n⚠️  Aucun namespace trouvé dans l'index.")
        print("   Vérifiez que des données ont été ajoutées à la base de données.\n")
        return

    print(f"\nIndex: {stats['index_name']}")
    print(f"Total de vecteurs: {stats['total_vectors']}\n")

    for ns, ns_stats in namespaces.items():
        ns_display = ns if ns else "(default)"
        vector_count = ns_stats.get('vector_count', 0)
        print(f"  • {ns_display}: {vector_count} vecteur(s)")

    print("\n" + "=" * 80)


def display_stats(vector_store: VectorStore):
    """Affiche les statistiques détaillées de la base de données."""
    print("\n" + "=" * 80)
    print("📊 STATISTIQUES DE LA BASE DE DONNÉES")
    print("=" * 80)

    stats = vector_store.get_stats()

    print(f"\nIndex: {stats['index_name']}")
    print(f"Total de vecteurs: {stats['total_vectors']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Métrique de similarité: {stats['metric']}")

    namespaces = stats.get('namespaces', {})
    if namespaces:
        print(f"\nNamespaces ({len(namespaces)}):")
        for ns, ns_stats in namespaces.items():
            ns_display = ns if ns else "(default)"
            vector_count = ns_stats.get('vector_count', 0)
            print(f"  • {ns_display}: {vector_count} vecteur(s)")
    else:
        print("\n⚠️  Aucun namespace trouvé.")

    print(f"\n💡 Dashboard Pinecone: https://app.pinecone.io/")
    print("=" * 80)


def main():
    """Fonction principale interactive."""
    # Vérifier les clés API
    is_valid, missing_keys = settings.validate_api_keys()
    if not is_valid:
        logger.error(f"Clés API manquantes: {', '.join(missing_keys)}")
        print(f"\n❌ ERREUR: Clés API manquantes: {', '.join(missing_keys)}")
        print("   Créez un fichier .env avec les clés nécessaires\n")
        return

    # Initialiser le VectorStore
    logger.info("Connexion à Pinecone...")
    print("\n🔌 Connexion à Pinecone...")
    try:
        vector_store = VectorStore(
            index_name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension
        )
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Pinecone: {e}")
        print(f"\n❌ Erreur lors de la connexion: {e}\n")
        return

    # Namespace par défaut
    current_namespace = ""

    # Afficher les stats au démarrage
    stats = vector_store.get_stats()
    if stats['total_vectors'] == 0:
        logger.warning("La base de données est vide")
        print("\n⚠️  ATTENTION: La base de données est vide!")
        print("   Lancez d'abord 'generate.py' pour ajouter des documents.\n")
        return

    logger.info(f"Connecté à l'index '{settings.pinecone_index_name}' ({stats['total_vectors']} vecteurs)")
    print(f"\n✅ Connecté à l'index '{settings.pinecone_index_name}'")
    print(f"   Total de vecteurs: {stats['total_vectors']}")

    # Menu interactif
    while True:
        display_menu()

        try:
            choice = input("\nVotre choix: ").strip()

            if choice == "0":
                print("\n👋 Au revoir!\n")
                break

            elif choice == "1":
                # Poser une question
                print("\n" + "-" * 80)
                question = input("💬 Votre question: ").strip()

                if not question:
                    print("⚠️  Question vide. Veuillez réessayer.")
                    continue

                # Demander le nombre de résultats
                top_k_input = input("📊 Nombre de résultats (défaut=5): ").strip()
                top_k = int(top_k_input) if top_k_input.isdigit() else 5

                # Rechercher
                results = query_vector_db(
                    question=question,
                    vector_store=vector_store,
                    namespace=current_namespace,
                    top_k=top_k
                )

                # Afficher les résultats
                display_results(results)

            elif choice == "2":
                # Changer de namespace
                list_namespaces(vector_store)
                new_namespace = input("\n🔄 Nouveau namespace (vide pour default): ").strip()
                current_namespace = new_namespace
                ns_display = current_namespace if current_namespace else "(default)"
                print(f"\n✅ Namespace actuel: {ns_display}")

            elif choice == "3":
                # Afficher les namespaces
                list_namespaces(vector_store)

            elif choice == "4":
                # Afficher les stats
                display_stats(vector_store)

            else:
                print("\n⚠️  Choix invalide. Veuillez entrer un nombre entre 0 et 4.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Au revoir!\n")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("Retour au menu principal...\n")


if __name__ == "__main__":
    main()