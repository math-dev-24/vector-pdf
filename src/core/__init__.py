"""
Module core contenant les éléments fondamentaux de l'application.
"""

from .config import Settings, settings
from .logging_config import setup_logging, get_logger
from .exceptions import PipelineError, ErrorType, ConfigurationError, ValidationError
from .singletons import OpenAIClient, PineconeClient, MistralClient
from .retry import retry_with_backoff
from .metrics import Metrics, MetricsCollector

__all__ = [
    "Settings",
    "settings",
    "setup_logging",
    "get_logger",
    "PipelineError",
    "ErrorType",
    "ConfigurationError",
    "ValidationError",
    "OpenAIClient",
    "PineconeClient",
    "MistralClient",
    "retry_with_backoff",
    "Metrics",
    "MetricsCollector",
]
