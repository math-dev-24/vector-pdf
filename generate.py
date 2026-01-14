"""
Point d'entrée principal pour le pipeline OCR-VECTOR-DOC.
Utilise la nouvelle architecture modulaire avec séparation des responsabilités.
"""

from src.core import setup_logging, settings
from src.cli import CLIApplication

setup_logging(level=settings.log_level, log_file=settings.log_file)


def main():
    """Point d'entrée principal."""
    app = CLIApplication()
    app.run()


if __name__ == "__main__":
    main()
