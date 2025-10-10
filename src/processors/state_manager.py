"""
Module pour gérer la sauvegarde/chargement des états intermédiaires du pipeline.
Permet d'exécuter les étapes indépendamment.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional


class StateManager:
    """Gestionnaire d'états pour le pipeline OCR-VECTOR-DOC."""

    def __init__(self, cache_dir: str = "./.cache"):
        """
        Initialise le gestionnaire d'états.

        Args:
            cache_dir: Répertoire pour stocker les états
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.chunks_file = self.cache_dir / "chunks_results.pkl"
        self.embeddings_file = self.cache_dir / "embeddings_results.pkl"
        self.metadata_file = self.cache_dir / "metadata.json"

    def save_chunks(self, results: List[Dict]) -> None:
        """
        Sauvegarde les résultats du chunking.

        Args:
            results: Résultats du chunking
        """
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(results, f)

        # Sauvegarder les métadonnées
        metadata = {
            'num_files': len(results),
            'total_chunks': sum(r['num_chunks'] for r in results),
            'files': [r['file_name'] for r in results]
        }
        self._save_metadata('chunks', metadata)
        print(f"✓ État du chunking sauvegardé dans {self.chunks_file}")

    def load_chunks(self) -> Optional[List[Dict]]:
        """
        Charge les résultats du chunking.

        Returns:
            Résultats du chunking ou None si absent
        """
        if not self.chunks_file.exists():
            return None

        with open(self.chunks_file, 'rb') as f:
            results = pickle.load(f)

        print(f"✓ État du chunking chargé depuis {self.chunks_file}")
        return results

    def save_embeddings(self, enriched_results: List[Dict], namespace: str = "") -> None:
        """
        Sauvegarde les résultats avec embeddings.

        Args:
            enriched_results: Résultats enrichis avec embeddings
            namespace: Namespace associé (optionnel)
        """
        # Si namespace spécifié, utiliser un fichier différent
        if namespace:
            embeddings_file = self.cache_dir / f"embeddings_{namespace}.pkl"
        else:
            embeddings_file = self.embeddings_file

        with open(embeddings_file, 'wb') as f:
            pickle.dump(enriched_results, f)

        # Sauvegarder les métadonnées
        total_chunks = sum(len(r['chunks']) for r in enriched_results)
        metadata = {
            'num_files': len(enriched_results),
            'total_chunks': total_chunks,
            'embedding_dimension': len(enriched_results[0]['chunks'][0]['embedding']) if enriched_results else 0,
            'files': [r['file_name'] for r in enriched_results],
            'namespace': namespace
        }

        # Clé unique pour les métadonnées
        meta_key = f'embeddings_{namespace}' if namespace else 'embeddings'
        self._save_metadata(meta_key, metadata)

        ns_display = f" (namespace: {namespace})" if namespace else ""
        print(f"✓ État des embeddings sauvegardé{ns_display} dans {embeddings_file}")

    def load_embeddings(self, namespace: str = "") -> Optional[List[Dict]]:
        """
        Charge les résultats avec embeddings.

        Args:
            namespace: Namespace à charger (optionnel)

        Returns:
            Résultats enrichis ou None si absent
        """
        # Choisir le bon fichier
        if namespace:
            embeddings_file = self.cache_dir / f"embeddings_{namespace}.pkl"
        else:
            embeddings_file = self.embeddings_file

        if not embeddings_file.exists():
            return None

        with open(embeddings_file, 'rb') as f:
            enriched_results = pickle.load(f)

        ns_display = f" (namespace: {namespace})" if namespace else ""
        print(f"✓ État des embeddings chargé{ns_display} depuis {embeddings_file}")
        return enriched_results

    def _save_metadata(self, stage: str, metadata: Dict) -> None:
        """Sauvegarde les métadonnées d'une étape."""
        all_metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)

        all_metadata[stage] = metadata

        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

    def get_metadata(self, stage: Optional[str] = None) -> Dict:
        """
        Récupère les métadonnées d'une étape ou toutes.

        Args:
            stage: Nom de l'étape ('chunks' ou 'embeddings') ou None pour tout

        Returns:
            Métadonnées demandées
        """
        if not self.metadata_file.exists():
            return {}

        with open(self.metadata_file, 'r') as f:
            all_metadata = json.load(f)

        if stage:
            return all_metadata.get(stage, {})
        return all_metadata

    def has_chunks(self) -> bool:
        """Vérifie si des chunks sont disponibles."""
        return self.chunks_file.exists()

    def has_embeddings(self, namespace: str = "") -> bool:
        """
        Vérifie si des embeddings sont disponibles.

        Args:
            namespace: Namespace à vérifier (optionnel)

        Returns:
            True si les embeddings existent
        """
        if namespace:
            embeddings_file = self.cache_dir / f"embeddings_{namespace}.pkl"
        else:
            embeddings_file = self.embeddings_file
        return embeddings_file.exists()

    def list_available_namespaces(self) -> List[str]:
        """
        Liste tous les namespaces disponibles en cache.

        Returns:
            Liste des namespaces
        """
        namespaces = []

        # Chercher tous les fichiers embeddings_*.pkl
        for file in self.cache_dir.glob("embeddings_*.pkl"):
            # Extraire le namespace du nom de fichier
            namespace = file.stem.replace("embeddings_", "")
            if namespace != "results":  # Ignorer embeddings_results.pkl
                namespaces.append(namespace)

        # Vérifier si le default existe
        if self.embeddings_file.exists():
            namespaces.insert(0, "(default)")

        return namespaces

    def clear_all(self) -> None:
        """Supprime tous les états sauvegardés."""
        if self.chunks_file.exists():
            self.chunks_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("✓ Tous les états ont été supprimés")

    def clear_chunks(self) -> None:
        """Supprime l'état des chunks."""
        if self.chunks_file.exists():
            self.chunks_file.unlink()
            print("✓ État des chunks supprimé")

    def clear_embeddings(self, namespace: str = "") -> None:
        """
        Supprime l'état des embeddings.

        Args:
            namespace: Namespace à supprimer (optionnel, "" = default)
        """
        if namespace:
            embeddings_file = self.cache_dir / f"embeddings_{namespace}.pkl"
        else:
            embeddings_file = self.embeddings_file

        if embeddings_file.exists():
            embeddings_file.unlink()
            ns_display = f" (namespace: {namespace})" if namespace else ""
            print(f"✓ État des embeddings{ns_display} supprimé")

    def print_status(self) -> None:
        """Affiche l'état actuel du cache."""
        print("\n=== État du cache ===")
        print(f"Répertoire: {self.cache_dir}")

        if self.has_chunks():
            chunks_meta = self.get_metadata('chunks')
            print(f"\n✓ Chunks disponibles:")
            print(f"  - Fichiers: {chunks_meta.get('num_files', 0)}")
            print(f"  - Total chunks: {chunks_meta.get('total_chunks', 0)}")
        else:
            print("\n✗ Pas de chunks disponibles")

        # Afficher les embeddings par namespace
        namespaces = self.list_available_namespaces()
        if namespaces:
            print(f"\n✓ Embeddings disponibles ({len(namespaces)} namespace(s)):")
            all_metadata = self.get_metadata()

            for ns in namespaces:
                # Trouver les métadonnées correspondantes
                if ns == "(default)":
                    meta_key = 'embeddings'
                else:
                    meta_key = f'embeddings_{ns}'

                emb_meta = all_metadata.get(meta_key, {})
                print(f"\n  [{ns}]")
                print(f"    - Fichiers: {emb_meta.get('num_files', 0)}")
                print(f"    - Total chunks: {emb_meta.get('total_chunks', 0)}")
                print(f"    - Dimension: {emb_meta.get('embedding_dimension', 0)}")
        else:
            print("\n✗ Pas d'embeddings disponibles")
