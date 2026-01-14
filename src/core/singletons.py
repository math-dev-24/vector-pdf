"""
Singletons pour les clients API (OpenAI, Pinecone).
"""

import os
from typing import Optional
from openai import OpenAI, AsyncOpenAI
from pinecone import Pinecone
import logging

from .config import settings
from .exceptions import ConfigurationError, PipelineError, ErrorType

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Singleton pour le client OpenAI."""
    
    _instance: Optional['OpenAIClient'] = None
    _client: Optional[OpenAI] = None
    _async_client: Optional[AsyncOpenAI] = None
    
    def __new__(cls):
        """Crée une seule instance du client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le client OpenAI."""
        if self._client is None:
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "OPENAI_API_KEY non définie. "
                    "Définissez-la dans .env ou via la variable d'environnement."
                )
            
            self._client = OpenAI(api_key=api_key)
            logger.info("Client OpenAI initialisé")
    
    @property
    def client(self) -> OpenAI:
        """Récupère le client OpenAI synchrone."""
        if self._client is None:
            self.__init__()
        return self._client
    
    @property
    def async_client(self) -> AsyncOpenAI:
        """Récupère le client OpenAI asynchrone."""
        if self._async_client is None:
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError("OPENAI_API_KEY non définie")
            
            self._async_client = AsyncOpenAI(api_key=api_key)
            logger.info("Client OpenAI async initialisé")
        
        return self._async_client
    
    def reset(self):
        """Réinitialise les clients (utile pour les tests)."""
        self._client = None
        self._async_client = None


class PineconeClient:
    """Singleton pour le client Pinecone."""
    
    _instance: Optional['PineconeClient'] = None
    _client: Optional[Pinecone] = None
    
    def __new__(cls):
        """Crée une seule instance du client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le client Pinecone."""
        if self._client is None:
            api_key = settings.pinecone_api_key or os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "PINECONE_API_KEY non définie. "
                    "Définissez-la dans .env ou via la variable d'environnement."
                )
            
            try:
                self._client = Pinecone(api_key=api_key)
                logger.info("Client Pinecone initialisé")
            except Exception as e:
                raise PipelineError(
                    ErrorType.PINECONE,
                    f"Erreur lors de l'initialisation du client Pinecone: {e}",
                    original_error=e
                )
    
    @property
    def client(self) -> Pinecone:
        """Récupère le client Pinecone."""
        if self._client is None:
            self.__init__()
        return self._client
    
    def reset(self):
        """Réinitialise le client (utile pour les tests)."""
        self._client = None


class MistralClient:
    """Singleton pour le client Mistral AI."""
    
    _instance: Optional['MistralClient'] = None
    _api_key: Optional[str] = None
    
    def __new__(cls):
        """Crée une seule instance du client."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le client Mistral."""
        if self._api_key is None:
            api_key = settings.mistral_api_key or os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "MISTRAL_API_KEY non définie. "
                    "Définissez-la dans .env ou via la variable d'environnement."
                )
            
            self._api_key = api_key
            logger.info("Client Mistral initialisé")
    
    @property
    def api_key(self) -> str:
        """Récupère la clé API Mistral."""
        if self._api_key is None:
            self.__init__()
        return self._api_key
    
    def reset(self):
        """Réinitialise le client (utile pour les tests)."""
        self._api_key = None