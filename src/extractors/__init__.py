"""
Modules d'extraction de texte depuis les PDFs (natifs et scannés).
"""

from .text_extractor import extract_text_from_pdf
from .scan_extractor import extract_text_from_scan

__all__ = ['extract_text_from_pdf', 'extract_text_from_scan']
