"""
Extracteur PDF amélioré avec détection de structure et titres.
Utilise PyMuPDF pour analyser les tailles de police et détecter la hiérarchie.
Supporte le multithreading pour améliorer les performances.
"""

import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
import os
from src.processors.text_cleaner import clean_text


def analyze_font_sizes(doc: fitz.Document) -> dict:
    """
    Analyse les tailles de police dans le document pour identifier les niveaux de titres.

    Args:
        doc: Document PyMuPDF

    Returns:
        Dictionnaire avec statistiques des tailles de police
    """
    font_sizes = defaultdict(int)

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span.get("size", 0), 1)
                        if size > 0:
                            font_sizes[size] += len(span.get("text", ""))

    return dict(sorted(font_sizes.items(), reverse=True))


def determine_heading_levels(font_stats: dict, body_size: float = None) -> dict:
    """
    Détermine les niveaux de titres en fonction des tailles de police.
    Améliore la détection en utilisant plusieurs critères.

    Args:
        font_stats: Statistiques des tailles de police
        body_size: Taille de police du corps de texte (auto-détectée si None)

    Returns:
        Mapping taille -> niveau markdown (# ## ### ####)
    """
    if not font_stats:
        return {}

    # Trier par fréquence d'utilisation
    sizes_by_freq = sorted(font_stats.items(), key=lambda x: x[1], reverse=True)

    # La taille la plus fréquente est généralement le corps de texte
    # Mais vérifier si c'est raisonnable (pas trop petit)
    if body_size is None:
        body_size = sizes_by_freq[0][0]

        # Si la taille la plus fréquente est très petite, chercher une meilleure candidate
        if body_size < 8:
            for size, freq in sizes_by_freq:
                if size >= 10:
                    body_size = size
                    break

    # Obtenir toutes les tailles triées par ordre décroissant
    all_sizes = sorted(font_stats.keys(), reverse=True)

    # Séparer les tailles en catégories plus finement
    level_map = {}

    for size in all_sizes:
        ratio = size / body_size

        if ratio >= 2.0:  # Titre principal (H1)
            level_map[size] = "#"
        elif ratio >= 1.6:  # Sous-titre important (H2)
            level_map[size] = "##"
        elif ratio >= 1.3:  # Sous-titre moyen (H3)
            level_map[size] = "###"
        elif ratio >= 1.15:  # Sous-titre petit (H4)
            level_map[size] = "####"
        elif ratio >= 0.9:  # Corps de texte
            level_map[size] = ""
        else:  # Petit texte (notes de bas de page, etc.)
            level_map[size] = ""

    return level_map


def extract_structured_text_from_pdf(
    pdf_path: str,
    output_dir: str = "./OUTPUT",
    verbose: bool = True,
    auto_detect_headings: bool = True
) -> str:
    """
    Extrait le texte d'un PDF avec détection de structure et titres.

    Args:
        pdf_path: Chemin vers le PDF source
        output_dir: Répertoire de sortie
        verbose: Afficher les logs
        auto_detect_headings: Détecter automatiquement les titres

    Returns:
        Chemin vers le fichier markdown créé
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{pdf_path.stem}.md"

    # Ouvrir le PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if verbose:
        print(f"  📄 Analyse de {total_pages} pages...")

    # Analyser les tailles de police
    heading_map = {}
    if auto_detect_headings:
        if verbose:
            print(f"  🔍 Détection automatique des titres...")

        font_stats = analyze_font_sizes(doc)
        heading_map = determine_heading_levels(font_stats)

        if verbose and heading_map:
            print(f"  ✓ {len([v for v in heading_map.values() if v])} niveau(x) de titre(s) détecté(s)")

    markdown_lines = []

    # Extraire avec structure
    for page_num in range(total_pages):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        if verbose and (page_num + 1) % 5 == 0:
            print(f"    Page {page_num + 1}/{total_pages} traitée")

        for block in blocks:
            if block.get("type") == 0:  # Text block
                block_text = []
                current_heading = ""

                for line in block.get("lines", []):
                    line_text = ""
                    line_size = None

                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = round(span.get("size", 0), 1)

                        if text:
                            line_text += text + " "
                            if line_size is None:
                                line_size = size

                    line_text = line_text.strip()

                    if line_text:
                        # Déterminer si c'est un titre
                        if line_size and line_size in heading_map:
                            heading_prefix = heading_map[line_size]

                            if heading_prefix:  # C'est un titre
                                # Sauvegarder le bloc précédent
                                if block_text:
                                    markdown_lines.append(" ".join(block_text))
                                    markdown_lines.append("")
                                    block_text = []

                                # Ajouter le titre
                                markdown_lines.append(f"{heading_prefix} {line_text}")
                                markdown_lines.append("")
                            else:
                                block_text.append(line_text)
                        else:
                            block_text.append(line_text)

                # Ajouter le reste du bloc
                if block_text:
                    markdown_lines.append(" ".join(block_text))
                    markdown_lines.append("")

    doc.close()

    if verbose:
        print(f"  🧹 Nettoyage du texte...")

    # Assembler et nettoyer
    full_content = "\n".join(markdown_lines)
    cleaned_content = clean_text(full_content, is_ocr=False)

    # Sauvegarder
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    if verbose:
        # Compter les titres
        title_count = cleaned_content.count("\n#")
        print(f"  ✅ Sauvegardé: {len(cleaned_content)} caractères, {title_count} titre(s)")

    return str(output_file)


def extract_with_pymupdf4llm(
    pdf_path: str,
    output_dir: str = "./OUTPUT",
    verbose: bool = True
) -> str:
    """
    Extrait avec pymupdf4llm (bibliothèque spécialisée markdown).
    Nécessite: pip install pymupdf4llm

    Args:
        pdf_path: Chemin vers le PDF
        output_dir: Répertoire de sortie
        verbose: Afficher les logs

    Returns:
        Chemin vers le fichier markdown
    """
    try:
        import pymupdf4llm
    except ImportError:
        raise ImportError(
            "pymupdf4llm n'est pas installé. "
            "Installez-le avec: pip install pymupdf4llm"
        )

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{pdf_path.stem}.md"

    if verbose:
        print(f"  📄 Extraction avec pymupdf4llm...")

    # Extraire avec pymupdf4llm
    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    if verbose:
        print(f"  🧹 Nettoyage du texte...")

    # Nettoyer
    cleaned_content = clean_text(md_text, is_ocr=False)

    # Sauvegarder
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    if verbose:
        title_count = cleaned_content.count("\n#")
        print(f"  ✅ Sauvegardé: {len(cleaned_content)} caractères, {title_count} titre(s)")

    return str(output_file)


def process_multiple_pdfs(
    pdf_paths: List[str],
    output_dir: str = "./OUTPUT",
    extraction_func: Callable = extract_structured_text_from_pdf,
    verbose: bool = True,
    max_workers: int = 1,
    use_multithreading: bool = False
) -> List[str]:
    """
    Traite plusieurs PDFs en parallèle avec multithreading (optionnel).

    Args:
        pdf_paths: Liste des chemins vers les PDFs
        output_dir: Répertoire de sortie
        extraction_func: Fonction d'extraction à utiliser
        verbose: Afficher les logs
        max_workers: Nombre maximum de threads (ignoré si use_multithreading=False)
        use_multithreading: Activer le traitement parallèle (False par défaut pour stabilité)
                            ATTENTION: peut causer des crashs avec PDFs contenant des images/scans

    Returns:
        Liste des chemins des fichiers markdown créés
    """
    # Désactiver le multithreading par défaut pour éviter les crashs avec PDFs images
    if not use_multithreading:
        max_workers = 1
    elif max_workers is None or max_workers < 1:
        # Utiliser le nombre optimal de threads (CPU count + 4 pour I/O bound tasks)
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    if verbose:
        print(f"\n🚀 Traitement parallèle de {len(pdf_paths)} PDF(s)...")
        print(f"   Threads: {max_workers}")

    output_files = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les jobs
        future_to_pdf = {
            executor.submit(extraction_func, pdf_path, output_dir, False): pdf_path
            for pdf_path in pdf_paths
        }

        # Traiter les résultats au fur et à mesure
        for i, future in enumerate(as_completed(future_to_pdf), 1):
            pdf_path = future_to_pdf[future]

            try:
                output_file = future.result()
                output_files.append(output_file)

                if verbose:
                    print(f"  [{i}/{len(pdf_paths)}] ✅ {Path(pdf_path).name}")
            except Exception as e:
                errors.append((pdf_path, str(e)))
                if verbose:
                    print(f"  [{i}/{len(pdf_paths)}] ❌ {Path(pdf_path).name}: {e}")

    if verbose:
        print(f"\n✅ Traitement terminé: {len(output_files)}/{len(pdf_paths)} réussi(s)")
        if errors:
            print(f"   ⚠️  {len(errors)} erreur(s) rencontrée(s)")

    return output_files


if __name__ == "__main__":
    """Test de l'extracteur amélioré."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_extractor_v2.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=" * 60)
    print("TEST EXTRACTEUR V2 (Structure améliorée)")
    print("=" * 60)

    output = extract_structured_text_from_pdf(
        pdf_path=pdf_path,
        output_dir="./OUTPUT_TEST",
        verbose=True,
        auto_detect_headings=True
    )

    print(f"\n✓ Fichier créé: {output}")

    # Afficher un aperçu
    with open(output, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')[:50]  # 50 premières lignes

        print("\n" + "=" * 60)
        print("APERÇU (50 premières lignes)")
        print("=" * 60)
        print('\n'.join(lines))

        if len(content.split('\n')) > 50:
            print("\n[...]\n")
