"""
Fenêtre principale de l'interface graphique.
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QProgressBar,
)
from PySide6.QtCore import QThread
from PySide6.QtGui import QFont

from src.core import settings
from src.processors import StateManager

from .workers import PipelineWorker
from .dialogs import (
    ExtractionConfigDialog,
    ChunkingConfigDialog,
    NamespaceDialog,
    VectorizationConfigDialog,
    GoToDbConfigDialog,
    FullPipelineConfigDialog,
    ClearCacheDialog,
)
from src.pipeline.models import ExtractionMode, PDFFilter, ChunkingMode


class MainWindow(QMainWindow):
    """Fenêtre principale de l'application."""

    def __init__(self):
        super().__init__()
        self.state_manager = StateManager(str(settings.cache_dir))
        self.worker: PipelineWorker | None = None
        self.thread: QThread | None = None
        self._enriched_results = None
        self._current_namespace = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("OCR-VECTOR-DOC - Pipeline")
        self.setMinimumSize(700, 600)
        self.resize(800, 650)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Titre
        title = QLabel("🚀 Pipeline OCR-VECTOR-DOC")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Boutons d'action
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()

        btn_layout1 = QHBoxLayout()
        self.btn_pdf_md = QPushButton("1. PDF → MD (Extraction)")
        self.btn_pdf_md.setMinimumHeight(40)
        self.btn_pdf_md.clicked.connect(self._on_pdf_to_md)
        btn_layout1.addWidget(self.btn_pdf_md)

        self.btn_vectorization = QPushButton("2. Vectorisation")
        self.btn_vectorization.setMinimumHeight(40)
        self.btn_vectorization.clicked.connect(self._on_vectorization)
        btn_layout1.addWidget(self.btn_vectorization)

        self.btn_go_db = QPushButton("3. Go to DB (Pinecone)")
        self.btn_go_db.setMinimumHeight(40)
        self.btn_go_db.clicked.connect(self._on_go_to_db)
        btn_layout1.addWidget(self.btn_go_db)
        actions_layout.addLayout(btn_layout1)

        btn_layout2 = QHBoxLayout()
        self.btn_full = QPushButton("4. Pipeline complet")
        self.btn_full.setMinimumHeight(40)
        self.btn_full.clicked.connect(self._on_full_pipeline)
        btn_layout2.addWidget(self.btn_full)

        self.btn_cache_status = QPushButton("5. État du cache")
        self.btn_cache_status.setMinimumHeight(40)
        self.btn_cache_status.clicked.connect(self._on_cache_status)
        btn_layout2.addWidget(self.btn_cache_status)

        self.btn_clear_cache = QPushButton("6. Nettoyer le cache")
        self.btn_clear_cache.setMinimumHeight(40)
        self.btn_clear_cache.clicked.connect(self._on_clear_cache)
        btn_layout2.addWidget(self.btn_clear_cache)
        actions_layout.addLayout(btn_layout2)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Mode indéterminé
        layout.addWidget(self.progress_bar)

        # Zone de log / résultats
        log_group = QGroupBox("Journal")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self._log("Application démarrée. Prêt.")

    def _log(self, msg: str) -> None:
        """Ajoute un message au journal."""
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _set_busy(self, busy: bool) -> None:
        """Active/désactive l'état occupé."""
        for btn in [
            self.btn_pdf_md,
            self.btn_vectorization,
            self.btn_go_db,
            self.btn_full,
            self.btn_cache_status,
            self.btn_clear_cache,
        ]:
            btn.setEnabled(not busy)
        self.progress_bar.setVisible(busy)

    def _run_worker(self, worker: PipelineWorker, thread: QThread) -> None:
        """Configure et lance un worker dans un thread."""
        self.worker = worker
        self.thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_worker_finished)
        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(thread.quit)
        self._set_busy(True)
        self._log("⏳ Opération en cours...")
        thread.start()

    def _on_worker_progress(self, msg: str) -> None:
        self._log(f"  → {msg}")

    def _on_worker_finished(self, success: bool, message: str) -> None:
        self._set_busy(False)
        if success:
            self._log(f"✅ {message}")
        else:
            self._log(f"❌ {message}")
        self.worker = None
        self.thread = None

    def _on_pdf_to_md(self) -> None:
        dialog = ExtractionConfigDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        extraction_mode, pdf_filter = dialog.get_config()
        worker = PipelineWorker(self.state_manager)
        worker.operation = "extraction"
        worker.extraction_mode = extraction_mode
        worker.pdf_filter = pdf_filter

        thread = QThread()
        worker.run = lambda: worker.run_extraction(
            worker.extraction_mode, worker.pdf_filter
        )
        self._run_worker(worker, thread)

    def _on_vectorization(self) -> None:
        has_chunks = self.state_manager.has_chunks()
        has_embeddings = self.state_manager.has_embeddings("")
        dialog = VectorizationConfigDialog(
            self, has_chunks=has_chunks, has_embeddings=has_embeddings
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        namespace, chunking_mode, use_chunks, use_embeddings = dialog.get_config()

        worker = PipelineWorker(self.state_manager)
        worker.operation = "vectorization"
        worker.namespace = namespace
        worker.chunking_mode = chunking_mode
        worker.use_cached_chunks = use_chunks
        worker.use_cached_embeddings = use_embeddings

        thread = QThread()
        worker.run = lambda: worker.run_vectorization(
            worker.namespace,
            worker.chunking_mode,
            worker.use_cached_chunks,
            worker.use_cached_embeddings,
        )
        self._run_worker(worker, thread)

    def _on_go_to_db(self) -> None:
        available_ns = self.state_manager.list_available_namespaces()

        if not available_ns:
            reply = QMessageBox.question(
                self,
                "Aucun embedding",
                "Aucun embedding en cache. Voulez-vous lancer la vectorisation d'abord?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_vectorization()
            return

        dialog = GoToDbConfigDialog(
            self, namespaces=available_ns, selected_namespace=self._current_namespace
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        namespace, reset = dialog.get_config()

        if reset:
            ns_display = f"'{namespace}'" if namespace else "(default)"
            reply = QMessageBox.warning(
                self,
                "Confirmation",
                f"Le namespace {ns_display} sera réinitialisé avant l'ajout.\n"
                "Êtes-vous sûr?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Charger les embeddings
        ns_key = "" if namespace == "(default)" else namespace
        enriched_results = self.state_manager.load_embeddings(ns_key)

        if not enriched_results:
            self._log("❌ Impossible de charger les embeddings.")
            return

        worker = PipelineWorker(self.state_manager)
        worker.operation = "store"
        worker.enriched_results = enriched_results
        worker.namespace = ns_key
        worker.reset = reset

        thread = QThread()
        worker.run = lambda: worker.run_store(
            worker.enriched_results, worker.namespace, worker.reset
        )
        self._run_worker(worker, thread)

    def _on_full_pipeline(self) -> None:
        dialog = FullPipelineConfigDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        namespace, extraction_mode, pdf_filter, chunking_mode, reset = dialog.get_config()

        if reset:
            ns_display = f"'{namespace}'" if namespace else "(default)"
            reply = QMessageBox.warning(
                self,
                "Confirmation",
                f"Le namespace {ns_display} sera réinitialisé avant l'ajout.\n"
                "Êtes-vous sûr?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        worker = PipelineWorker(self.state_manager)
        worker.operation = "full"
        worker.namespace = namespace
        worker.extraction_mode = extraction_mode
        worker.pdf_filter = pdf_filter
        worker.chunking_mode = chunking_mode
        worker.reset = reset

        thread = QThread()
        worker.run = lambda: worker.run_full_pipeline(
            worker.namespace,
            worker.extraction_mode,
            worker.pdf_filter,
            worker.chunking_mode,
            worker.reset,
        )
        self._run_worker(worker, thread)

    def _on_cache_status(self) -> None:
        lines = ["=== État du cache ===", f"Répertoire: {self.state_manager.cache_dir}"]

        if self.state_manager.has_chunks():
            meta = self.state_manager.get_metadata("chunks")
            lines.append("\n✓ Chunks disponibles:")
            lines.append(f"  - Fichiers: {meta.get('num_files', 0)}")
            lines.append(f"  - Total chunks: {meta.get('total_chunks', 0)}")
        else:
            lines.append("\n✗ Pas de chunks disponibles")

        namespaces = self.state_manager.list_available_namespaces()
        if namespaces:
            lines.append(f"\n✓ Embeddings ({len(namespaces)} namespace(s)):")
            all_meta = self.state_manager.get_metadata()
            for ns in namespaces:
                meta_key = "embeddings" if ns == "(default)" else f"embeddings_{ns}"
                emb = all_meta.get(meta_key, {})
                lines.append(f"  [{ns}] chunks: {emb.get('total_chunks', 0)}")
        else:
            lines.append("\n✗ Pas d'embeddings disponibles")

        self._log("\n".join(lines))

    def _on_clear_cache(self) -> None:
        dialog = ClearCacheDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        choice = dialog.get_choice()
        if choice == "chunks":
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "Confirmer la suppression des chunks?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.state_manager.clear_chunks()
                self._log("✓ Chunks supprimés")
        elif choice == "embeddings":
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "Confirmer la suppression des embeddings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                for ns in self.state_manager.list_available_namespaces():
                    ns_key = "" if ns == "(default)" else ns
                    self.state_manager.clear_embeddings(ns_key)
                self._log("✓ Embeddings supprimés")
        elif choice == "all":
            reply = QMessageBox.warning(
                self,
                "Confirmation",
                "Confirmer la suppression de TOUT le cache?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.state_manager.clear_all()
                self._log("✓ Cache entièrement supprimé")
