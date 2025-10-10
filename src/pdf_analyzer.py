"""
Module pour analyser les PDFs (type, taille, pages).
"""

import os
from pathlib import Path
from typing import Dict, List
import fitz  # PyMuPDF


def analyze_pdf(pdf_path: str) -> Dict:
    """
    Analyse un PDF et détermine son type (scan ou texte natif).

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

    # Déterminer si c'est un scan ou un PDF texte
    # On vérifie le nombre de caractères sur les premières pages
    total_text_length = 0
    pages_to_check = min(3, num_pages)  # Vérifier les 3 premières pages

    for page_num in range(pages_to_check):
        page = doc[page_num]
        text = page.get_text()
        total_text_length += len(text.strip())

    doc.close()

    # Si moins de 100 caractères par page en moyenne -> scan
    avg_chars_per_page = total_text_length / pages_to_check
    page_type = "scan" if avg_chars_per_page < 100 else "text"

    return {
        "path": pdf_path,
        "size_mb": round(file_size_mb, 2),
        "num_pages": num_pages,
        "page_type": page_type
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
