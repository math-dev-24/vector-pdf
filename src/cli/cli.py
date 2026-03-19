"""
Interface CLI pour le pipeline.
Séparation claire entre logique métier et interface utilisateur.
"""

from typing import Optional

from src.core import settings, get_logger, PipelineError
from src.processors import StateManager
from src.pipeline import Pipeline, PipelineConfig
from src.pipeline.models import (
    ExtractionMode,
    PDFFilter,
    ChunkingMode,
    NamespaceStrategy
)

logger = get_logger(__name__)


class CLIApplication:
    """Application CLI pour le pipeline."""
    
    def __init__(self):
        """Initialise l'application CLI."""
        self.state_manager = StateManager(str(settings.cache_dir))
        self.pipeline: Optional[Pipeline] = None
    
    def display_menu(self) -> None:
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
    
    def _get_extraction_config(self) -> tuple[ExtractionMode, PDFFilter]:
        """Demande la configuration d'extraction à l'utilisateur."""
        print("\n💡 Type de PDFs à traiter:")
        print("  1. Tous (texte natif + scans)")
        print("  2. Uniquement texte natif")
        print("  3. Uniquement scans")
        
        choice = input("\nVotre choix (1/2/3, défaut=1): ").strip()
        pdf_filters = {"1": PDFFilter.ALL, "2": PDFFilter.TEXT, "3": PDFFilter.SCAN}
        pdf_filter = pdf_filters.get(choice, PDFFilter.ALL)
        
        print("\n💡 Choisissez le mode d'extraction:")
        print("  1. Basique (rapide, sans structure)")
        print("  2. Structurée (détection automatique des titres)")
        print("  3. PyMuPDF4LLM (optimal pour LLM)")
        
        choice = input("\nVotre choix (1/2/3, défaut=2): ").strip()
        extraction_modes = {
            "1": ExtractionMode.BASIC,
            "2": ExtractionMode.STRUCTURED,
            "3": ExtractionMode.PYMUPDF4LLM
        }
        extraction_mode = extraction_modes.get(choice, ExtractionMode.STRUCTURED)
        
        return extraction_mode, pdf_filter
    
    def _get_chunking_config(self) -> ChunkingMode:
        """Demande la configuration de chunking à l'utilisateur."""
        print("\n💡 Mode de chunking:")
        print("  1. Standard (rapide, pas d'IA)")
        print("  2. Avancé (enrichissement IA + contexte)")
        
        choice = input("\nVotre choix (1/2, défaut=2): ").strip()
        return ChunkingMode.STANDARD if choice == "1" else ChunkingMode.ADVANCED
    
    def _get_namespace(self) -> str:
        """Demande le namespace à l'utilisateur."""
        return input("\nNamespace Pinecone (laisser vide pour default): ").strip()

    def _get_namespace_strategy(self) -> tuple[NamespaceStrategy, str, str]:
        """
        Demande la stratégie de namespace à l'utilisateur.

        Returns:
            Tuple (strategy, namespace, namespace_prefix)
        """
        print("\n💡 Stratégie de namespace Pinecone:")
        print("  1. IA           (Dépannage / Dimensionnement / Général) [recommandé]")
        print("  2. Par fichier  (1 namespace / document PDF)")
        print("  3. Par dossier  (1 namespace / dossier source)")
        print("  4. Unique       (tout dans un seul namespace)")

        choice = input("\nVotre choix (1/2/3/4, défaut=1): ").strip()

        if choice == "2":
            strategy = NamespaceStrategy.BY_FILE
            prefix = input("Préfixe optionnel (ex: 'v2', laisser vide pour aucun): ").strip()
            return strategy, "", prefix
        elif choice == "3":
            strategy = NamespaceStrategy.BY_FOLDER
            prefix = input("Préfixe optionnel (ex: 'v2', laisser vide pour aucun): ").strip()
            return strategy, "", prefix
        elif choice == "4":
            strategy = NamespaceStrategy.NONE
            namespace = input("Namespace (laisser vide pour default): ").strip()
            return strategy, namespace, ""
        else:
            # Défaut : BY_AI
            return NamespaceStrategy.BY_AI, "", ""

    def _get_reset_confirmation(self, namespace: str) -> bool:
        """Demande confirmation pour réinitialiser le namespace."""
        ns_display = f"'{namespace}'" if namespace else "(default)"
        print(f"\n⚠️  Le namespace {ns_display} sera réinitialisé avant l'ajout!")
        confirm = input("Êtes-vous sûr? (oui/non): ").strip().lower()
        return confirm == "oui"
    
    def process_pdf_to_md(self) -> None:
        """Option 1: Extraction PDF vers Markdown."""
        print("\n" + "=" * 60)
        print("📄 TRAITEMENT PDF TO MD")
        print("=" * 60)
        
        extraction_mode, pdf_filter = self._get_extraction_config()
        
        config = PipelineConfig(
            data_dir=settings.data_dir,
            output_dir=settings.output_dir,
            extraction_mode=extraction_mode,
            pdf_filter=pdf_filter
        )
        
        pipeline = Pipeline(config=config, state_manager=self.state_manager)
        result = pipeline.extract()
        
        if result.output_files:
            print(f"\n✅ Extraction terminée: {len(result.output_files)} fichier(s) créé(s)")
            print(f"   - PDFs texte: {result.text_pdfs_count}")
            print(f"   - PDFs scannés: {result.scan_pdfs_count}")
            print(f"   - Total pages: {result.total_pages}")
        else:
            print("\n⚠️  Aucun fichier extrait")
    
    def process_vectorization(self) -> Optional[tuple]:
        """Option 2: Vectorisation (Chunking + Embeddings)."""
        print("\n" + "=" * 60)
        print("🔢 VECTORISATION")
        print("=" * 60)
        
        namespace = self._get_namespace()
        
        # Vérifier le cache
        if self.state_manager.has_embeddings(namespace):
            ns_display = f" '{namespace}'" if namespace else " (default)"
            print(f"\n💡 Des embeddings sont déjà en cache pour le namespace{ns_display}.")
            choice = input("Voulez-vous les utiliser? (o/n): ").strip().lower()
            if choice == 'o':
                enriched_results = self.state_manager.load_embeddings(namespace)
                if enriched_results:
                    return enriched_results, namespace
        
        # Vérifier les chunks
        results = None
        if self.state_manager.has_chunks():
            print("\n💡 Des chunks sont déjà en cache.")
            choice = input("Voulez-vous les utiliser? (o/n): ").strip().lower()
            if choice == 'o':
                results = self.state_manager.load_chunks()
        
        # Chunking si nécessaire
        if not results:
            chunking_mode = self._get_chunking_config()
            
            config = PipelineConfig(
                data_dir=settings.data_dir,
                output_dir=settings.output_dir,
                chunking_mode=chunking_mode,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            pipeline = Pipeline(config=config, state_manager=self.state_manager)
            chunking_result = pipeline.chunk()
            
            if not chunking_result.chunks:
                print("\n⚠️  Aucun document markdown trouvé. Lancez d'abord l'option 1.")
                return None
            
            # Convertir en format attendu
            results = [{
                'file_path': 'combined',
                'file_name': 'combined',
                'num_chunks': chunking_result.total_chunks,
                'total_chars': chunking_result.total_chars,
                'chunks': chunking_result.chunks
            }]
            
            self.state_manager.save_chunks(results)
        
        # Vectorisation
        config = PipelineConfig(
            data_dir=settings.data_dir,
            output_dir=settings.output_dir,
            embedding_model=settings.embedding_model,
            embedding_batch_size=settings.embedding_batch_size
        )
        
        pipeline = Pipeline(config=config, state_manager=self.state_manager)
        embedding_result = pipeline.embed(results[0]['chunks'])
        
        if embedding_result.enriched_chunks:
            # Convertir en format attendu
            enriched_results = [{
                'file_path': 'combined',
                'file_name': 'combined',
                'num_chunks': embedding_result.total_embeddings,
                'total_chars': results[0]['total_chars'],
                'chunks': embedding_result.enriched_chunks
            }]
            
            self.state_manager.save_embeddings(enriched_results, namespace)
            print("\n✅ Vectorisation terminée!")
            return enriched_results, namespace
        else:
            print("\n⚠️  Échec de la vectorisation.")
            return None
    
    def process_go_to_db(self, enriched_results=None, namespace=None) -> None:
        """Option 3: Stockage dans Pinecone."""
        print("\n" + "=" * 60)
        print("💾 GO TO DB")
        print("=" * 60)
        
        # Charger depuis le cache si nécessaire
        if enriched_results is None:
            available_ns = self.state_manager.list_available_namespaces()
            
            if available_ns:
                print("\n💡 Namespaces disponibles en cache:")
                for i, ns in enumerate(available_ns, 1):
                    print(f"  {i}. {ns}")
                print(f"  {len(available_ns) + 1}. Nouveau namespace / Lancer vectorisation")
                
                choice = input(f"\nChoisir un namespace (1-{len(available_ns) + 1}): ").strip()
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_ns):
                        selected_ns = available_ns[idx]
                        namespace = "" if selected_ns == "(default)" else selected_ns
                        print(f"\n✓ Chargement des embeddings du namespace '{selected_ns}'...")
                        enriched_results = self.state_manager.load_embeddings(namespace)
                    else:
                        namespace = self._get_namespace()
                        print("\nLancement de la vectorisation...")
                        result = self.process_vectorization()
                        if result:
                            enriched_results, namespace = result
                except (ValueError, IndexError):
                    print("Choix invalide.")
                    return
            else:
                print("\nAucun embedding en cache. Lancement de la vectorisation...")
                result = self.process_vectorization()
                if result:
                    enriched_results, namespace = result
        
        if not enriched_results:
            print("\n⚠️  Impossible de continuer sans embeddings.")
            return
        
        # Stratégie de namespace
        ns_strategy, namespace, ns_prefix = self._get_namespace_strategy()

        if ns_strategy == NamespaceStrategy.NONE:
            ns_display = f"'{namespace}'" if namespace else "(default)"
        else:
            ns_display = f"automatique ({ns_strategy.value})"
            if ns_prefix:
                ns_display += f" avec préfixe '{ns_prefix}'"
        print(f"\n📌 Namespace cible: {ns_display}")

        print("\n💡 Options de stockage:")
        print(f"  1. Ajouter aux vecteurs existants")
        print(f"  2. Réinitialiser le(s) namespace(s) avant l'ajout")

        choice = input("Votre choix (1/2): ").strip()
        reset = (choice == "2")

        if reset and ns_strategy == NamespaceStrategy.NONE:
            if not self._get_reset_confirmation(namespace):
                print("Opération annulée.")
                return

        # Stockage
        config = PipelineConfig(
            data_dir=settings.data_dir,
            output_dir=settings.output_dir,
            namespace=namespace,
            namespace_strategy=ns_strategy,
            namespace_prefix=ns_prefix,
            reset_namespace=reset
        )

        pipeline = Pipeline(config=config, state_manager=self.state_manager)
        chunks = enriched_results[0]['chunks'] if enriched_results else []
        storage_result = pipeline.store(chunks, reset=reset)

        # Afficher les stats
        print("\n✅ Stockage terminé!")
        if storage_result.vector_store:
            stats = storage_result.vector_store.get_stats()
            print(f"\nBase de données vectorielle Pinecone:")
            print(f"  - Index: {stats['index_name']}")
            print(f"  - Total vecteurs: {stats['total_vectors']}")
            print(f"  - Dimension: {stats['dimension']}")
            print(f"  - Métrique: {stats['metric']}")

            if storage_result.namespaces:
                print(f"\n  Vecteurs uploadés par namespace:")
                for ns_name, count in sorted(storage_result.namespaces.items()):
                    print(f"    - {ns_name}: {count} chunks")
            elif stats['namespaces']:
                print(f"\n  Vecteurs par namespace (Pinecone):")
                for ns, ns_stats in stats['namespaces'].items():
                    ns_name = ns if ns else "(default)"
                    print(f"    - {ns_name}: {ns_stats.get('vector_count', 0)} vecteurs")

        print(f"\n💡 Dashboard Pinecone: https://app.pinecone.io/")
    
    def run_full_pipeline(self) -> None:
        """Option 4: Pipeline complet."""
        print("\n" + "=" * 60)
        print("🔄 PIPELINE COMPLET")
        print("=" * 60)

        extraction_mode, pdf_filter = self._get_extraction_config()
        chunking_mode = self._get_chunking_config()
        ns_strategy, namespace, ns_prefix = self._get_namespace_strategy()

        print("\n💡 Options de la base de données vectorielle:")
        print("  1. Ajouter aux vecteurs existants")
        print("  2. Réinitialiser le(s) namespace(s) avant l'ajout")

        choice = input("\nVotre choix (1/2, défaut=1): ").strip()
        reset = (choice == "2")

        if reset and ns_strategy == NamespaceStrategy.NONE:
            if not self._get_reset_confirmation(namespace):
                print("Opération annulée.")
                return

        # Configuration complète
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
            namespace_strategy=ns_strategy,
            namespace_prefix=ns_prefix,
            reset_namespace=reset
        )
        
        pipeline = Pipeline(config=config, state_manager=self.state_manager)
        result = pipeline.run_full()
        
        if result.success:
            print("\n" + "=" * 60)
            print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
            print("=" * 60)
            
            if result.storage and result.storage.vector_store:
                stats = result.storage.vector_store.get_stats()
                print(f"\nBase de données vectorielle Pinecone:")
                print(f"  - Index: {stats['index_name']}")
                print(f"  - Total vecteurs: {stats['total_vectors']}")
                print(f"  - Dimension: {stats['dimension']}")
                print(f"  - Métrique: {stats['metric']}")

                if result.storage.namespaces:
                    print(f"\n  Chunks uploadés par namespace:")
                    for ns_name, count in sorted(result.storage.namespaces.items()):
                        print(f"    - {ns_name}: {count} chunks")
                elif stats['namespaces']:
                    print(f"\n  Vecteurs par namespace (Pinecone):")
                    for ns, ns_stats in stats['namespaces'].items():
                        ns_name = ns if ns else "(default)"
                        print(f"    - {ns_name}: {ns_stats.get('vector_count', 0)} vecteurs")
            
            print("\n💡 Prochaines étapes:")
            print("  - Utilisez le VectorStore pour des recherches sémantiques")
            print("  - Intégrez avec un LLM pour du RAG (Retrieval-Augmented Generation)")
            print(f"  - Dashboard Pinecone: https://app.pinecone.io/")
        else:
            print(f"\n❌ Erreur: {result.error}")
    
    def show_cache_status(self) -> None:
        """Option 5: Afficher l'état du cache."""
        self.state_manager.print_status()
    
    def clear_cache(self) -> None:
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
                self.state_manager.clear_chunks()
        elif choice == "2":
            confirm = input("Confirmer la suppression des embeddings? (oui/non): ").strip().lower()
            if confirm == "oui":
                self.state_manager.clear_embeddings()
        elif choice == "3":
            confirm = input("⚠️  Confirmer la suppression de TOUT le cache? (oui/non): ").strip().lower()
            if confirm == "oui":
                self.state_manager.clear_all()
        elif choice == "0":
            print("Annulé.")
        else:
            print("Choix invalide.")
    
    def run(self) -> None:
        """Lance l'application CLI."""
        while True:
            self.display_menu()
            
            try:
                choice = input("\nVotre choix: ").strip()
                
                if choice == "0":
                    print("\n👋 Au revoir!")
                    break
                elif choice == "1":
                    self.process_pdf_to_md()
                elif choice == "2":
                    self.process_vectorization()
                elif choice == "3":
                    self.process_go_to_db()
                elif choice == "4":
                    self.run_full_pipeline()
                elif choice == "5":
                    self.show_cache_status()
                elif choice == "6":
                    self.clear_cache()
                else:
                    print("\n⚠️  Choix invalide. Veuillez entrer un nombre entre 0 et 6.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Au revoir!")
                break
            except Exception as e:
                logger.error(f"Erreur dans l'application CLI: {e}", exc_info=True)
                print(f"\n❌ Erreur: {e}")
                print("Retour au menu principal...")
