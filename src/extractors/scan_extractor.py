"""
Module pour l'extraction de texte depuis des PDFs scannés via OCR.
Utilise pytesseract + pdf2image pour l'OCR.
"""

from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from src.processors.text_cleaner import clean_text


def extract_text_from_scan(pdf_path: str, output_dir: str = "./OUTPUT", lang: str = "fra", verbose: bool = True) -> str:
    """
    Extrait le texte d'un PDF scanné via OCR et le sauvegarde en markdown.

    Args:
        pdf_path: Chemin vers le PDF scanné
        output_dir: Répertoire de sortie
        lang: Langue pour l'OCR (fra=français, eng=anglais)
        verbose: Afficher les logs de progression

    Returns:
        Chemin vers le fichier markdown créé
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier de sortie
    output_file = output_dir / f"{pdf_path.stem}.md"

    # Convertir le PDF en images
    if verbose:
        print(f"  🖼️  Conversion en images (DPI: 300)...")
    images = convert_from_path(str(pdf_path), dpi=300)

    if verbose:
        print(f"  🔍 OCR en cours sur {len(images)} pages...")

    markdown_content = []

    # OCR sur chaque page (sans titres de pages)
    for page_num, image in enumerate(images, start=1):
        if verbose:
            print(f"    Page {page_num}/{len(images)} - OCR en cours...")

        # Extraire le texte avec Tesseract
        text = pytesseract.image_to_string(image, lang=lang)

        if text.strip():
            markdown_content.append(text)
            markdown_content.append("\n\n")

    if verbose:
        print(f"  🧹 Nettoyage du texte OCR...")

    # Assembler le contenu
    full_content = ''.join(markdown_content)

    # Nettoyer le texte (nettoyage plus agressif pour OCR)
    cleaned_content = clean_text(full_content, is_ocr=True)

    # Sauvegarder en markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    if verbose:
        print(f"  ✅ Sauvegardé: {len(cleaned_content)} caractères")

    return str(output_file)


