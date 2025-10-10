"""
Point d'entrée principal du pipeline OCR-VECTOR-DOC.
Extrait, nettoie, chunke et vectorise des PDFs.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from src.pdf_analyzer import analyze_pdfs
from src.extractors import extract_text_from_pdf, extract_text_from_scan
from src.extractors.text_extractor_v2 import extract_structured_text_from_pdf, extract_with_pymupdf4llm
from src.processors import chunk_all_markdown_files, StateManager
from src.vectorization.embeddings import embed_all_files
from src.vectorization.vector_store import store_embeddings

# Charger la configuration
load_dotenv()

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "./DATA")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./OUTPUT")
CACHE_DIR = os.getenv("CACHE_DIR", "./.cache")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-documents")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# Gestionnaire d'états
state_manager = StateManager(CACHE_DIR)


def run_extraction(data_dir: str, output_dir: str, extraction_mode: str = "basic") -> None:
    """
    Étape 1: Extraction des PDFs (natifs et scans).

    Args:
        data_dir: Répertoire contenant les PDFs
        output_dir: Répertoire de sortie pour les markdown
        extraction_mode: Mode d'extraction ("basic", "structured", "pymupdf4llm")
    """
    print("=" * 60)
    print("ÉTAPE 1: EXTRACTION DES PDFS")
    print("=" * 60)

    # Analyser les PDFs
    print("\n=== Analyse des PDFs ===\n")
    pdf_infos = analyze_pdfs(data_dir)

    if not pdf_infos:
        print(f"Aucun PDF trouvé dans {data_dir}")
        return

    print(f"Nombre total de PDFs analysés: {len(pdf_infos)}\n")

    # Séparer scans et textes natifs
    text_pdfs = [p for p in pdf_infos if p['page_type'] == 'text']
    scan_pdfs = [p for p in pdf_infos if p['page_type'] == 'scan']

    # Afficher les infos
    for i, pdf in enumerate(pdf_infos, start=1):
        print(f"[{i}/{len(pdf_infos)}] {pdf['path']}")
        print(f"        Taille: {pdf['size_mb']} MB | Pages: {pdf['num_pages']} | Type: {pdf['page_type']}\n")

    # Extraire les PDFs natifs
    if text_pdfs:
        mode_display = {
            "basic": "Basique (rapide)",
            "structured": "Structurée avec détection de titres",
            "pymupdf4llm": "PyMuPDF4LLM (optimal pour LLM)"
        }

        print(f"\n=== Extraction des PDFs natifs ===")
        print(f"📑 {len(text_pdfs)} PDF(s) natif(s) à traiter")
        print(f"📋 Mode: {mode_display.get(extraction_mode, extraction_mode)}\n")

        for i, pdf in enumerate(text_pdfs, start=1):
            print(f"[{i}/{len(text_pdfs)}] {pdf['path']}")

            try:
                if extraction_mode == "structured":
                    output_path = extract_structured_text_from_pdf(pdf['path'], output_dir, verbose=True)
                elif extraction_mode == "pymupdf4llm":
                    output_path = extract_with_pymupdf4llm(pdf['path'], output_dir, verbose=True)
                else:  # basic
                    output_path = extract_text_from_pdf(pdf['path'], output_dir, verbose=True)

                print(f"  → Fichier créé: {output_path}\n")
            except Exception as e:
                print(f"  ❌ Erreur: {e}\n")

        print(f"✅ Extraction terminée: {len(text_pdfs)} fichier(s) créé(s)\n")

    # Extraire les scans
    if scan_pdfs:
        print(f"\n=== Extraction des PDFs scannés (OCR) ===")
        print(f"📸 {len(scan_pdfs)} PDF(s) scanné(s) à traiter\n")

        for i, pdf in enumerate(scan_pdfs, start=1):
            print(f"[{i}/{len(scan_pdfs)}] {pdf['path']}")
            output_path = extract_text_from_scan(pdf['path'], output_dir, verbose=True)
            print(f"  → Fichier créé: {output_path}\n")

        print(f"✅ Extraction OCR terminée: {len(scan_pdfs)} fichier(s) créé(s)\n")


def run_chunking(output_dir: str, chunk_size: int, chunk_overlap: int):
    """
    Étape 2: Chunking des fichiers markdown.

    Args:
        output_dir: Répertoire contenant les markdowns
        chunk_size: Taille des chunks
        chunk_overlap: Overlap entre chunks

    Returns:
        Résultats du chunking
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 2: CHUNKING DES DOCUMENTS")
    print("=" * 60)

    results = chunk_all_markdown_files(
        directory=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=True
    )

    return results


def run_vectorization(results, model: str, batch_size: int):
    """
    Étape 3: Création des embeddings.

    Args:
        results: Résultats du chunking
        model: Modèle d'embedding OpenAI
        batch_size: Taille des batchs

    Returns:
        Résultats enrichis avec embeddings
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 3: VECTORISATION (EMBEDDINGS)")
    print("=" * 60)

    # Vérifier la clé API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  ERREUR: OPENAI_API_KEY non définie dans .env")
        print("   Créez un fichier .env basé sur .env.example")
        return None

    enriched_results = embed_all_files(
        results=results,
        model=model,
        batch_size=batch_size,
        verbose=True
    )

    return enriched_results


def run_storage(enriched_results, index_name: str, dimension: int, namespace: str = "", reset: bool = False):
    """
    Étape 4: Stockage dans Pinecone.

    Args:
        enriched_results: Résultats avec embeddings
        index_name: Nom de l'index Pinecone
        dimension: Dimension des vecteurs
        namespace: Namespace Pinecone (optionnel)
        reset: Réinitialiser le namespace avant l'ajout

    Returns:
        Instance du VectorStore
    """
    print("\n" + "=" * 60)
    print("ÉTAPE 4: STOCKAGE DANS PINECONE")
    print("=" * 60)

    vector_store = store_embeddings(
        enriched_results=enriched_results,
        index_name=index_name,
        dimension=dimension,
        namespace=namespace,
        reset=reset
    )

    return vector_store


def display_menu():
    """Affiche le menu principal."""
    print("\n" + "=" * 60)
    print("🚀 PIPELINE OCR-VECTOR-DOC")
    print("=" * 60)
    print("\nChoisissez une option:")
    print("  1. Traitement PDF to MD (Extraction uniquement)")
    print("  2. Vectorisation (Chunking + Embeddings)")
    print("  3. Go to DB (Stockage dans Pinecone)")
    print("  4. Pipeline complet (1 → 2 → 3)")
    print("  5. Afficher l'état du cache")
    print("  6. Nettoyer le cache")
    print("  0. Quitter")
    print("=" * 60)


def process_pdf_to_md():
    """Option 1: Extraction PDF vers Markdown."""
    print("\n" + "=" * 60)
    print("📄 TRAITEMENT PDF TO MD")
    print("=" * 60)

    # Demander le mode d'extraction
    print("\n💡 Choisissez le mode d'extraction:")
    print("  1. Basique (rapide, sans structure)")
    print("  2. Structurée (détection automatique des titres)")
    print("  3. PyMuPDF4LLM (optimal pour LLM, nécessite pymupdf4llm)")

    choice = input("\nVotre choix (1/2/3): ").strip()

    extraction_modes = {
        "1": "basic",
        "2": "structured",
        "3": "pymupdf4llm"
    }

    extraction_mode = extraction_modes.get(choice, "structured")  # Par défaut: structured

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    run_extraction(DATA_DIR, OUTPUT_DIR, extraction_mode)

    print("\n✅ Extraction terminée!")


def process_vectorization():
    """Option 2: Vectorisation (Chunking + Embeddings)."""
    print("\n" + "=" * 60)
    print("🔢 VECTORISATION")
    print("=" * 60)

    # Demander le namespace
    namespace = input("\nNamespace Pinecone (laisser vide pour default): ").strip()

    # Vérifier s'il y a des embeddings en cache pour ce namespace
    if state_manager.has_embeddings(namespace):
        ns_display = f" '{namespace}'" if namespace else " (default)"
        print(f"\n💡 Des embeddings sont déjà en cache pour le namespace{ns_display}.")
        choice = input("Voulez-vous les utiliser? (o/n): ").strip().lower()
        if choice == 'o':
            enriched_results = state_manager.load_embeddings(namespace)
            if enriched_results:
                return enriched_results, namespace

    # Vérifier s'il y a des chunks en cache
    results = None
    if state_manager.has_chunks():
        print("\n💡 Des chunks sont déjà en cache.")
        choice = input("Voulez-vous les utiliser? (o/n): ").strip().lower()
        if choice == 'o':
            results = state_manager.load_chunks()

    # Sinon, faire le chunking
    if not results:
        results = run_chunking(OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

        if not results:
            print("\n⚠️  Aucun document markdown trouvé. Lancez d'abord l'option 1.")
            return None

        # Sauvegarder les chunks
        state_manager.save_chunks(results)

    # Étape 3: Vectorisation
    enriched_results = run_vectorization(
        results=results,
        model=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE
    )

    if enriched_results:
        # Sauvegarder les embeddings avec le namespace
        state_manager.save_embeddings(enriched_results, namespace)
        print("\n✅ Vectorisation terminée!")
        return enriched_results, namespace
    else:
        print("\n⚠️  Échec de la vectorisation.")
        return None, None


def process_go_to_db(enriched_results=None, namespace=None):
    """Option 3: Stockage dans Pinecone."""
    print("\n" + "=" * 60)
    print("💾 GO TO DB")
    print("=" * 60)

    # Si pas de résultats fournis, essayer de charger depuis le cache
    if enriched_results is None:
        # Afficher les namespaces disponibles
        available_ns = state_manager.list_available_namespaces()

        if available_ns:
            print("\n💡 Namespaces disponibles en cache:")
            for i, ns in enumerate(available_ns, 1):
                print(f"  {i}. {ns}")
            print(f"  {len(available_ns) + 1}. Nouveau namespace / Lancer vectorisation")

            choice = input(f"\nChoisir un namespace (1-{len(available_ns) + 1}): ").strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_ns):
                    # Charger le namespace sélectionné
                    selected_ns = available_ns[idx]
                    namespace = "" if selected_ns == "(default)" else selected_ns
                    print(f"\n✓ Chargement des embeddings du namespace '{selected_ns}'...")
                    enriched_results = state_manager.load_embeddings(namespace)
                else:
                    # Nouveau namespace ou vectorisation
                    namespace = input("\nNamespace Pinecone (laisser vide pour default): ").strip()
                    print("\nLancement de la vectorisation...")
                    enriched_results, namespace = process_vectorization()
            except (ValueError, IndexError):
                print("Choix invalide.")
                return
        else:
            print("\nAucun embedding en cache. Lancement de la vectorisation...")
            enriched_results, namespace = process_vectorization()

        if not enriched_results:
            print("\n⚠️  Impossible de continuer sans embeddings.")
            return

    # Afficher le namespace cible
    ns_display = f"'{namespace}'" if namespace else "(default)"
    print(f"\n📌 Namespace cible: {ns_display}")

    # Demander si on veut réinitialiser le namespace
    print("\n💡 Options de stockage:")
    print(f"  1. Ajouter aux vecteurs existants du namespace {ns_display}")
    print(f"  2. Réinitialiser le namespace {ns_display} avant l'ajout")
    choice = input("Votre choix (1/2): ").strip()
    reset = (choice == "2")

    if reset:
        print(f"\n⚠️  Le namespace {ns_display} sera réinitialisé avant l'ajout!")
        confirm = input("Êtes-vous sûr? (oui/non): ").strip().lower()
        if confirm != "oui":
            print("Opération annulée.")
            return

    # Étape 4: Stockage
    vector_store = run_storage(
        enriched_results=enriched_results,
        index_name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        namespace=namespace,
        reset=reset
    )

    # Résumé
    print("\n✅ Stockage terminé!")
    stats = vector_store.get_stats()
    print(f"\nBase de données vectorielle Pinecone:")
    print(f"  - Index: {stats['index_name']}")
    print(f"  - Total vecteurs: {stats['total_vectors']}")
    print(f"  - Dimension: {stats['dimension']}")
    print(f"  - Métrique: {stats['metric']}")

    # Afficher les stats par namespace
    if stats['namespaces']:
        print(f"\n  Vecteurs par namespace:")
        for ns, ns_stats in stats['namespaces'].items():
            ns_name = ns if ns else "(default)"
            print(f"    - {ns_name}: {ns_stats.get('vector_count', 0)} vecteurs")

    print(f"\n💡 Dashboard Pinecone: https://app.pinecone.io/")


def run_full_pipeline():
    """Option 4: Pipeline complet."""
    print("\n" + "=" * 60)
    print("🔄 PIPELINE COMPLET")
    print("=" * 60)

    # Demander le namespace
    namespace = input("\nNamespace Pinecone (laisser vide pour default): ").strip()

    # Demander le mode d'extraction
    print("\n💡 Choisissez le mode d'extraction:")
    print("  1. Basique")
    print("  2. Structurée (recommandé)")
    print("  3. PyMuPDF4LLM")

    choice = input("\nVotre choix (1/2/3, défaut=2): ").strip()
    extraction_modes = {"1": "basic", "2": "structured", "3": "pymupdf4llm"}
    extraction_mode = extraction_modes.get(choice, "structured")

    # Créer les répertoires
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Étape 1: Extraction
    run_extraction(DATA_DIR, OUTPUT_DIR, extraction_mode)

    # Étape 2: Chunking
    results = run_chunking(OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    if not results:
        print("\n⚠️  Aucun document à traiter. Arrêt du pipeline.")
        return

    # Étape 3: Vectorisation
    enriched_results = run_vectorization(
        results=results,
        model=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE
    )

    if not enriched_results:
        print("\n⚠️  Échec de la vectorisation. Arrêt du pipeline.")
        return

    # Sauvegarder les embeddings
    state_manager.save_embeddings(enriched_results, namespace)

    # Étape 4: Stockage
    vector_store = run_storage(
        enriched_results=enriched_results,
        index_name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        namespace=namespace,
        reset=False
    )

    # Résumé final
    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)

    stats = vector_store.get_stats()
    print(f"\nBase de données vectorielle Pinecone:")
    print(f"  - Index: {stats['index_name']}")
    print(f"  - Total vecteurs: {stats['total_vectors']}")
    print(f"  - Dimension: {stats['dimension']}")
    print(f"  - Métrique: {stats['metric']}")

    # Afficher les stats par namespace
    if stats['namespaces']:
        print(f"\n  Vecteurs par namespace:")
        for ns, ns_stats in stats['namespaces'].items():
            ns_name = ns if ns else "(default)"
            print(f"    - {ns_name}: {ns_stats.get('vector_count', 0)} vecteurs")

    print("\n💡 Prochaines étapes:")
    print("  - Utilisez le VectorStore pour des recherches sémantiques")
    print("  - Intégrez avec un LLM pour du RAG (Retrieval-Augmented Generation)")
    print(f"  - Dashboard Pinecone: https://app.pinecone.io/")


def show_cache_status():
    """Option 5: Afficher l'état du cache."""
    state_manager.print_status()


def clear_cache():
    """Option 6: Nettoyer le cache."""
    print("\n" + "=" * 60)
    print("🗑️  NETTOYAGE DU CACHE")
    print("=" * 60)

    print("\nOptions:")
    print("  1. Supprimer uniquement les chunks")
    print("  2. Supprimer uniquement les embeddings")
    print("  3. Supprimer tout le cache")
    print("  0. Annuler")

    choice = input("\nVotre choix: ").strip()

    if choice == "1":
        confirm = input("Confirmer la suppression des chunks? (oui/non): ").strip().lower()
        if confirm == "oui":
            state_manager.clear_chunks()
    elif choice == "2":
        confirm = input("Confirmer la suppression des embeddings? (oui/non): ").strip().lower()
        if confirm == "oui":
            state_manager.clear_embeddings()
    elif choice == "3":
        confirm = input("⚠️  Confirmer la suppression de TOUT le cache? (oui/non): ").strip().lower()
        if confirm == "oui":
            state_manager.clear_all()
    elif choice == "0":
        print("Annulé.")
    else:
        print("Choix invalide.")


def main():
    """Menu principal interactif."""
    while True:
        display_menu()

        try:
            choice = input("\nVotre choix: ").strip()

            if choice == "0":
                print("\n👋 Au revoir!")
                break
            elif choice == "1":
                process_pdf_to_md()
            elif choice == "2":
                result = process_vectorization()
                # Ignorer le résultat (juste pour le cache)
            elif choice == "3":
                process_go_to_db()
            elif choice == "4":
                run_full_pipeline()
            elif choice == "5":
                show_cache_status()
            elif choice == "6":
                clear_cache()
            else:
                print("\n⚠️  Choix invalide. Veuillez entrer un nombre entre 0 et 6.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("Retour au menu principal...")


if __name__ == "__main__":
    main()
