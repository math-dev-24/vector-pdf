"""
Exceptions personnalisées pour le pipeline.
"""

from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types d'erreurs du pipeline."""
    PDF_EXTRACTION = "pdf_extraction"
    PDF_ANALYSIS = "pdf_analysis"
    EMBEDDING = "embedding"
    PINECONE = "pinecone"
    CHUNKING = "chunking"
    CACHE = "cache"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class PipelineError(Exception):
    """Exception de base pour les erreurs du pipeline."""
    
    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        original_error: Optional[Exception] = None,
        details: Optional[dict] = None
    ):
        """
        Initialise l'erreur du pipeline.
        
        Args:
            error_type: Type d'erreur
            message: Message d'erreur
            original_error: Exception originale (optionnel)
            details: Détails supplémentaires (optionnel)
        """
        self.error_type = error_type
        self.message = message
        self.original_error = original_error
        self.details = details or {}
        super().__init__(self.message)
        
        # Logger l'erreur
        logger.error(
            f"{error_type.value}: {message}",
            exc_info=original_error,
            extra={"error_type": error_type.value, "details": self.details}
        )
    
    def __str__(self) -> str:
        """Représentation string de l'erreur."""
        base = f"[{self.error_type.value}] {self.message}"
        if self.original_error:
            base += f" (Original: {type(self.original_error).__name__})"
        return base


class ConfigurationError(PipelineError):
    """Erreur de configuration."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(ErrorType.CONFIGURATION, message, original_error)


class ValidationError(PipelineError):
    """Erreur de validation."""
    
    def __init__(self, message: str, field: Optional[str] = None, original_error: Optional[Exception] = None):
        details = {"field": field} if field else {}
        super().__init__(ErrorType.VALIDATION, message, original_error, details)
