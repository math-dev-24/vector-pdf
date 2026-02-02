"""
Dialogs de configuration pour l'interface graphique.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QComboBox,
    QLineEdit,
    QCheckBox,
    QDialogButtonBox,
    QLabel,
    QGroupBox,
)

from src.pipeline.models import ExtractionMode, PDFFilter, ChunkingMode


class ExtractionConfigDialog(QDialog):
    """Dialog pour configurer l'extraction PDF."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration extraction PDF")
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Type de PDFs
        pdf_group = QGroupBox("Type de PDFs")
        pdf_layout = QVBoxLayout()
        self.pdf_filter_combo = QComboBox()
        self.pdf_filter_combo.addItems([
            "Tous (texte natif + scans)",
            "Uniquement texte natif",
            "Uniquement scans",
        ])
        pdf_layout.addWidget(self.pdf_filter_combo)
        pdf_group.setLayout(pdf_layout)
        layout.addWidget(pdf_group)

        # Mode d'extraction
        mode_group = QGroupBox("Mode d'extraction")
        mode_layout = QVBoxLayout()
        self.extraction_mode_combo = QComboBox()
        self.extraction_mode_combo.addItems([
            "Basique (rapide, sans structure)",
            "Structurée (détection automatique des titres)",
            "PyMuPDF4LLM (optimal pour LLM)",
        ])
        self.extraction_mode_combo.setCurrentIndex(1)  # Structurée par défaut
        mode_layout.addWidget(self.extraction_mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> tuple[ExtractionMode, PDFFilter]:
        """Retourne (extraction_mode, pdf_filter)."""
        pdf_map = {
            0: PDFFilter.ALL,
            1: PDFFilter.TEXT,
            2: PDFFilter.SCAN,
        }
        mode_map = {
            0: ExtractionMode.BASIC,
            1: ExtractionMode.STRUCTURED,
            2: ExtractionMode.PYMUPDF4LLM,
        }
        return (
            mode_map[self.extraction_mode_combo.currentIndex()],
            pdf_map[self.pdf_filter_combo.currentIndex()],
        )


class ChunkingConfigDialog(QDialog):
    """Dialog pour configurer le chunking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration chunking")
        self.setMinimumWidth(350)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.chunking_combo = QComboBox()
        self.chunking_combo.addItems([
            "Standard (rapide, pas d'IA)",
            "Avancé (enrichissement IA + contexte)",
        ])
        self.chunking_combo.setCurrentIndex(1)  # Avancé par défaut
        layout.addWidget(QLabel("Mode de chunking:"))
        layout.addWidget(self.chunking_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_chunking_mode(self) -> ChunkingMode:
        return (
            ChunkingMode.STANDARD
            if self.chunking_combo.currentIndex() == 0
            else ChunkingMode.ADVANCED
        )


class NamespaceDialog(QDialog):
    """Dialog pour saisir le namespace Pinecone."""

    def __init__(self, parent=None, title: str = "Namespace Pinecone"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.namespace_edit = QLineEdit()
        self.namespace_edit.setPlaceholderText("Laisser vide pour default")
        layout.addWidget(QLabel("Namespace:"))
        layout.addWidget(self.namespace_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_namespace(self) -> str:
        return self.namespace_edit.text().strip()


class VectorizationConfigDialog(QDialog):
    """Dialog pour configurer la vectorisation."""

    def __init__(self, parent=None, has_chunks: bool = False, has_embeddings: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Configuration vectorisation")
        self.setMinimumWidth(400)
        self._setup_ui(has_chunks, has_embeddings)

    def _setup_ui(self, has_chunks: bool, has_embeddings: bool) -> None:
        layout = QVBoxLayout(self)

        # Namespace
        self.namespace_edit = QLineEdit()
        self.namespace_edit.setPlaceholderText("Laisser vide pour default")
        layout.addWidget(QLabel("Namespace:"))
        layout.addWidget(self.namespace_edit)

        # Options cache
        self.use_chunks_check = QCheckBox("Utiliser les chunks en cache")
        self.use_chunks_check.setChecked(has_chunks)
        self.use_chunks_check.setEnabled(has_chunks)
        layout.addWidget(self.use_chunks_check)

        self.use_embeddings_check = QCheckBox("Utiliser les embeddings en cache")
        self.use_embeddings_check.setChecked(has_embeddings)
        self.use_embeddings_check.setEnabled(has_embeddings)
        layout.addWidget(self.use_embeddings_check)

        # Chunking
        layout.addWidget(QLabel("Mode de chunking (si nouveau chunking):"))
        self.chunking_combo = QComboBox()
        self.chunking_combo.addItems([
            "Standard (rapide, pas d'IA)",
            "Avancé (enrichissement IA + contexte)",
        ])
        self.chunking_combo.setCurrentIndex(1)
        layout.addWidget(self.chunking_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> tuple[str, ChunkingMode, bool, bool]:
        """Retourne (namespace, chunking_mode, use_chunks, use_embeddings)."""
        chunking = (
            ChunkingMode.STANDARD
            if self.chunking_combo.currentIndex() == 0
            else ChunkingMode.ADVANCED
        )
        return (
            self.namespace_edit.text().strip(),
            chunking,
            self.use_chunks_check.isChecked(),
            self.use_embeddings_check.isChecked(),
        )


class GoToDbConfigDialog(QDialog):
    """Dialog pour configurer le stockage Pinecone."""

    def __init__(
        self,
        parent=None,
        namespaces: list[str] | None = None,
        selected_namespace: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Configuration stockage Pinecone")
        self.setMinimumWidth(450)
        self.namespaces = namespaces or []
        self.selected_namespace = selected_namespace
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Choix du namespace
        layout.addWidget(QLabel("Namespace:"))
        self.namespace_combo = QComboBox()
        if self.namespaces:
            self.namespace_combo.addItems(self.namespaces)
            if self.selected_namespace:
                idx = self.namespace_combo.findText(
                    self.selected_namespace if self.selected_namespace else "(default)"
                )
                if idx >= 0:
                    self.namespace_combo.setCurrentIndex(idx)
        else:
            self.namespace_combo.addItem("(aucun en cache - lancer vectorisation)")
        layout.addWidget(self.namespace_combo)

        self.reset_check = QCheckBox("Réinitialiser le namespace avant l'ajout")
        self.reset_check.setToolTip("Supprime tous les vecteurs existants avant d'ajouter les nouveaux")
        layout.addWidget(self.reset_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> tuple[str, bool]:
        """Retourne (namespace, reset)."""
        ns = self.namespace_combo.currentText()
        if ns == "(default)" or ns == "(aucun en cache - lancer vectorisation)":
            ns = ""
        return ns, self.reset_check.isChecked()


class FullPipelineConfigDialog(QDialog):
    """Dialog pour configurer le pipeline complet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration pipeline complet")
        self.setMinimumWidth(450)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.namespace_edit = QLineEdit()
        self.namespace_edit.setPlaceholderText("Laisser vide pour default")
        layout.addWidget(QLabel("Namespace Pinecone:"))
        layout.addWidget(self.namespace_edit)

        # Extraction
        layout.addWidget(QLabel("Type de PDFs:"))
        self.pdf_filter_combo = QComboBox()
        self.pdf_filter_combo.addItems([
            "Tous (texte natif + scans)",
            "Uniquement texte natif",
            "Uniquement scans",
        ])
        layout.addWidget(self.pdf_filter_combo)

        layout.addWidget(QLabel("Mode d'extraction:"))
        self.extraction_mode_combo = QComboBox()
        self.extraction_mode_combo.addItems([
            "Basique",
            "Structurée",
            "PyMuPDF4LLM",
        ])
        self.extraction_mode_combo.setCurrentIndex(1)
        layout.addWidget(self.extraction_mode_combo)

        # Chunking
        layout.addWidget(QLabel("Mode de chunking:"))
        self.chunking_combo = QComboBox()
        self.chunking_combo.addItems(["Standard", "Avancé"])
        self.chunking_combo.setCurrentIndex(1)
        layout.addWidget(self.chunking_combo)

        self.reset_check = QCheckBox("Réinitialiser le namespace avant l'ajout")
        layout.addWidget(self.reset_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> tuple[str, ExtractionMode, PDFFilter, ChunkingMode, bool]:
        """Retourne (namespace, extraction_mode, pdf_filter, chunking_mode, reset)."""
        pdf_map = {0: PDFFilter.ALL, 1: PDFFilter.TEXT, 2: PDFFilter.SCAN}
        mode_map = {
            0: ExtractionMode.BASIC,
            1: ExtractionMode.STRUCTURED,
            2: ExtractionMode.PYMUPDF4LLM,
        }
        chunk_map = {
            0: ChunkingMode.STANDARD,
            1: ChunkingMode.ADVANCED,
        }
        return (
            self.namespace_edit.text().strip(),
            mode_map[self.extraction_mode_combo.currentIndex()],
            pdf_map[self.pdf_filter_combo.currentIndex()],
            chunk_map[self.chunking_combo.currentIndex()],
            self.reset_check.isChecked(),
        )


class ClearCacheDialog(QDialog):
    """Dialog pour choisir quoi nettoyer dans le cache."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nettoyage du cache")
        self.setMinimumWidth(350)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.choice_combo = QComboBox()
        self.choice_combo.addItems([
            "Supprimer uniquement les chunks",
            "Supprimer uniquement les embeddings",
            "Supprimer tout le cache",
        ])
        layout.addWidget(QLabel("Options:"))
        layout.addWidget(self.choice_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_choice(self) -> str:
        """Retourne 'chunks', 'embeddings' ou 'all'."""
        choices = ["chunks", "embeddings", "all"]
        return choices[self.choice_combo.currentIndex()]
