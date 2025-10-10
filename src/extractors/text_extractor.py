import fitz  # PyMuPDF
from pathlib import Path
from src.processors.text_cleaner import clean_text


def extract_text_from_pdf(pdf_path: str, output_dir: str = "./OUTPUT", verbose: bool = True) -> str:
    """
    Extrait le texte d'un PDF natif (non scanné) et le sauvegarde en markdown.
    Préserve la structure avec les titres et paragraphes.

    Args:
        pdf_path: Chemin vers le PDF source
        output_dir: Répertoire de sortie pour les fichiers markdown
        verbose: Afficher les logs de progression

    Returns:
        Chemin vers le fichier markdown créé
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier de sortie
    output_file = output_dir / f"{pdf_path.stem}.md"

    # Ouvrir le PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if verbose:
        print(f"  📄 Extraction de {total_pages} pages...")

    markdown_content = []

    # Extraire le texte page par page (sans titres de pages)
    for page_num in range(total_pages):
        page = doc[page_num]

        # Extraire le texte avec structure
        text = page.get_text("text")

        if verbose and (page_num + 1) % 5 == 0:
            print(f"    Page {page_num + 1}/{total_pages} traitée")

        if text.strip():
            markdown_content.append(text)
            markdown_content.append("\n\n")

    doc.close()

    if verbose:
        print(f"  🧹 Nettoyage du texte...")

    # Assembler le contenu
    full_content = ''.join(markdown_content)

    # Nettoyer le texte
    cleaned_content = clean_text(full_content, is_ocr=False)

    # Sauvegarder en markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    if verbose:
        print(f"  ✅ Sauvegardé: {len(cleaned_content)} caractères")

    return str(output_file)


def process_native_pdfs(pdf_list: list[dict], output_dir: str = "./OUTPUT") -> list[str]:
    """
    Traite tous les PDFs natifs (type='text') d'une liste.

    Args:
        pdf_list: Liste de dicts avec infos PDF (depuis analyze_pdfs)
        output_dir: Répertoire de sortie

    Returns:
        Liste des chemins des fichiers markdown créés
    """
    output_files = []

    # Filtrer uniquement les PDFs natifs
    native_pdfs = [pdf for pdf in pdf_list if pdf['page_type'] == 'text']

    print(f"Traitement de {len(native_pdfs)} PDF(s) natif(s)...\n")

    for pdf_info in native_pdfs:
        try:
            print(f"Extraction: {pdf_info['path']}")
            output_file = extract_text_from_pdf(pdf_info['path'], output_dir)
            output_files.append(output_file)
            print(f"  → Créé: {output_file}\n")
        except Exception as e:
            print(f"  ✗ Erreur: {e}\n")

    print(f"Terminé: {len(output_files)} fichier(s) créé(s)")
    return output_files
