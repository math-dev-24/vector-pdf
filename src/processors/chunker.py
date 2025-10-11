"""
Module pour découper les fichiers markdown en chunks optimisés pour la vectorisation.
Utilise LangChain pour un chunking intelligent qui respecte la structure du document.
Supporte le multithreading pour améliorer les performances.
"""

import os
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_smart_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Crée un splitter intelligent qui respecte la structure markdown.

    Args:
        chunk_size: Taille maximale d'un chunk en caractères
        chunk_overlap: Nombre de caractères de chevauchement entre chunks

    Returns:
        RecursiveCharacterTextSplitter configuré
    """
    # Séparateurs hiérarchiques pour markdown (du plus important au moins important)
    separators = [
        "\n### ",  # Titres niveau 3 (sections)
        "\n## ",   # Titres niveau 2
        "\n# ",    # Titres niveau 1
        "\n\n",    # Paragraphes
        "\n",      # Lignes
        ". ",      # Phrases
        " ",       # Mots
        ""         # Caractères
    ]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )


def chunk_markdown_file(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict:
    """
    Découpe un fichier markdown en chunks avec métadonnées.

    Args:
        file_path: Chemin du fichier markdown
        chunk_size: Taille maximale d'un chunk
        chunk_overlap: Overlap entre chunks

    Returns:
        Dictionnaire avec informations sur le fichier et ses chunks
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Créer le splitter
    splitter = create_smart_splitter(chunk_size, chunk_overlap)

    # Découper le texte
    chunks = splitter.split_text(text)

    # Créer les métadonnées pour chaque chunk
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            'content': chunk,
            'metadata': {
                'source': file_path,
                'file_name': os.path.basename(file_path),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            }
        })

    return {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'num_chunks': len(chunks),
        'chunks': chunks_with_metadata,
        'total_chars': len(text)
    }


def chunk_all_markdown_files(
    directory: str = "./OUTPUT",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    verbose: bool = True,
    max_workers: int = None
) -> List[Dict]:
    """
    Découpe tous les fichiers markdown d'un répertoire en chunks.
    Utilise le multithreading pour améliorer les performances.

    Args:
        directory: Répertoire contenant les fichiers markdown
        chunk_size: Taille maximale d'un chunk
        chunk_overlap: Overlap entre chunks
        verbose: Afficher les logs de progression
        max_workers: Nombre maximum de threads (None = auto)

    Returns:
        Liste de dictionnaires avec les informations de chunking
    """
    output_dir = Path(directory)

    if not output_dir.exists():
        print(f"Le répertoire {directory} n'existe pas.")
        return []

    # Trouver tous les fichiers .md
    md_files = list(output_dir.glob("*.md"))

    if not md_files:
        print(f"Aucun fichier .md trouvé dans {directory}")
        return []

    if max_workers is None:
        # Optimal pour I/O bound tasks
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    if verbose:
        print(f"\n📝 Chunking de {len(md_files)} fichier(s) markdown...")
        print(f"   Paramètres: chunk_size={chunk_size}, overlap={chunk_overlap}")
        if len(md_files) > 1:
            print(f"   Threads: {max_workers}")
        print()

    results = []

    # Traitement parallèle si plusieurs fichiers
    if len(md_files) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre tous les jobs
            future_to_file = {
                executor.submit(chunk_markdown_file, str(md_file), chunk_size, chunk_overlap): md_file
                for md_file in md_files
            }

            # Traiter les résultats au fur et à mesure
            for i, future in enumerate(as_completed(future_to_file), 1):
                md_file = future_to_file[future]

                try:
                    result = future.result()
                    results.append(result)

                    if verbose:
                        print(f"  [{i}/{len(md_files)}] ✅ {md_file.name}: {result['num_chunks']} chunks")
                except Exception as e:
                    if verbose:
                        print(f"  [{i}/{len(md_files)}] ❌ {md_file.name}: {e}")
    else:
        # Un seul fichier : traitement séquentiel
        md_file = md_files[0]
        if verbose:
            print(f"  [1/1] Traitement de {md_file.name}...")

        result = chunk_markdown_file(str(md_file), chunk_size, chunk_overlap)
        results.append(result)

        if verbose:
            print(f"       ✅ {result['num_chunks']} chunks créés")

    if verbose:
        total_chunks = sum(r['num_chunks'] for r in results)
        print(f"\n✅ Chunking terminé: {total_chunks} chunks au total")

    return results


if __name__ == "__main__":
    print("=== Chunking intelligent des fichiers markdown (LangChain) ===\n")

    # Paramètres de chunking
    # Note: OpenAI recommande ~500-1000 tokens pour les embeddings
    # 1 token ≈ 4 caractères en français
    # Donc 1000 caractères ≈ 250 tokens (taille optimale)
    CHUNK_SIZE = 1000  # Caractères
    CHUNK_OVERLAP = 200  # Overlap pour préserver le contexte

    results = chunk_all_markdown_files(
        directory="./OUTPUT",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if not results:
        print("Aucun fichier traité.")
    else:
        print(f"Nombre de fichiers traités: {len(results)}\n")

        total_chunks = 0

        for result in results:
            print(f"Fichier: {result['file_name']}")
            print(f"  Taille: {result['total_chars']:,} caractères")
            print(f"  Nombre de chunks: {result['num_chunks']}")

            # Afficher les tailles min/max des chunks
            chunk_sizes = [c['metadata']['chunk_size'] for c in result['chunks']]
            if chunk_sizes:
                print(f"  Taille des chunks: min={min(chunk_sizes)}, max={max(chunk_sizes)}, moy={sum(chunk_sizes)//len(chunk_sizes)}")

            print()
            total_chunks += result['num_chunks']

        print(f"Total de chunks générés: {total_chunks}")
        print(f"\nParamètres de chunking:")
        print(f"  - Taille de chunk: {CHUNK_SIZE} caractères (~{CHUNK_SIZE//4} tokens)")
        print(f"  - Overlap: {CHUNK_OVERLAP} caractères")
        print(f"  - Splitter: RecursiveCharacterTextSplitter (respecte la structure markdown)")
