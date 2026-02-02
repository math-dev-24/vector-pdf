#!/usr/bin/env python3
"""
Point d'entrée pour l'interface graphique PySide6.
Lance l'application avec: python ui.py ou uv run python ui.py
"""

import sys

from src.ui import run_ui

if __name__ == "__main__":
    sys.exit(run_ui())
