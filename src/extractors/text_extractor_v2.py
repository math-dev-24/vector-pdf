"""
Extracteur PDF amélioré avec détection de structure et titres.
Utilise PyMuPDF pour analyser les tailles de police et détecter la hiérarchie.
"""

import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict
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

    Args:
        font_stats: Statistiques des tailles de police
        body_size: Taille de police du corps de texte (auto-détectée si None)

    Returns:
        Mapping taille -> niveau markdown (# ## ###)
    """
    if not font_stats:
        return {}

    # Trier par fréquence d'utilisation
    sizes_by_freq = sorted(font_stats.items(), key=lambda x: x[1], reverse=True)

    # La taille la plus fréquente est généralement le corps de texte
    if body_size is None:
        body_size = sizes_by_freq[0][0]

    # Créer le mapping
    level_map = {}

    for size, _ in font_stats.items():
        if size > body_size * 1.8:  # Très grand
            level_map[size] = "#"
        elif size > body_size * 1.4:  # Grand
            level_map[size] = "##"
        elif size > body_size * 1.1:  # Moyen
            level_map[size] = "###"
        elif size >= body_size * 0.95:  # Corps de texte
            level_map[size] = ""
        else:  # Petit (notes de bas de page, etc.)
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
