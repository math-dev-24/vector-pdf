"""
Module pour analyser les PDFs (type, taille, pages).
Détection multi-critères scan vs texte natif.
"""

import os
from pathlib import Path
from typing import Dict, List
import fitz  # PyMuPDF

# Seuils pour la détection scan vs texte
MIN_CHARS_PER_PAGE_TEXT = 80  # Moins = probable scan
MIN_IMAGE_RATIO_SCAN = 0.5  # Si >50% des pages ont des images full-page = scan
PAGES_TO_SAMPLE = 5  # Échantillonner plus de pages pour fiabilité


def analyze_pdf(pdf_path: str) -> Dict:
    """
    Analyse un PDF et détermine son type (scan ou texte natif).
    Utilise plusieurs critères pour une détection plus fiable :
    - Densité de texte (caractères par page)
    - Présence d'images full-page (indicateur de scan)

    Args:
        pdf_path: Chemin vers le PDF

    Returns:
        Dictionnaire avec les infos du PDF
    """
    doc = fitz.open(pdf_path)

    # Taille du fichier
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

    # Nombre de pages
    num_pages = len(doc)
    pages_to_check = min(PAGES_TO_SAMPLE, num_pages)

    total_text_length = 0
    pages_with_fullpage_images = 0

    for page_num in range(pages_to_check):
        page = doc[page_num]
        text = page.get_text()
        total_text_length += len(text.strip())

        # Détecter les images qui couvrent une grande partie de la page (scan)
        images = page.get_images()
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        for img_index in images:
            xref = img_index[0]
            try:
                img_rect = page.get_image_rects(xref)
                if img_rect:
                    img_area = sum(r.width * r.height for r in img_rect)
                    # Image couvre >40% de la page = probablement un scan
                    if img_area > 0.4 * page_area:
                        pages_with_fullpage_images += 1
                        break
            except (ValueError, KeyError):
                pass

    doc.close()

    # Critère 1 : densité de texte
    avg_chars_per_page = total_text_length / pages_to_check
    low_text_density = avg_chars_per_page < MIN_CHARS_PER_PAGE_TEXT

    # Critère 2 : ratio de pages avec images full-page
    image_ratio = pages_with_fullpage_images / pages_to_check
    high_image_ratio = image_ratio >= MIN_IMAGE_RATIO_SCAN

    # Décision : scan si (peu de texte) OU (beaucoup d'images full-page)
    page_type = "scan" if (low_text_density or high_image_ratio) else "text"

    return {
        "path": pdf_path,
        "size_mb": round(file_size_mb, 2),
        "num_pages": num_pages,
        "page_type": page_type,
        "avg_chars_per_page": round(avg_chars_per_page, 1),
        "image_page_ratio": round(image_ratio, 2),
    }


def analyze_pdfs(directory: str) -> List[Dict]:
    """
    Analyse tous les PDFs d'un répertoire (récursif).

    Args:
        directory: Répertoire contenant les PDFs

    Returns:
        Liste de dictionnaires avec les infos des PDFs
    """
    pdf_files = list(Path(directory).rglob("*.pdf"))

    results = []
    for pdf_file in pdf_files:
        try:
            info = analyze_pdf(str(pdf_file))
            results.append(info)
        except Exception as e:
            print(f"Erreur lors de l'analyse de {pdf_file}: {e}")

    return results
