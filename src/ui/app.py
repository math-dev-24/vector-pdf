"""
Point d'entrée de l'application graphique PySide6.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from src.core import setup_logging
from .main_window import MainWindow


def run_ui() -> int:
    """Lance l'interface graphique. Retourne le code de sortie."""
    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("OCR-VECTOR-DOC")
    app.setOrganizationName("vector-pdf")

    # Style par défaut
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    return app.exec()
