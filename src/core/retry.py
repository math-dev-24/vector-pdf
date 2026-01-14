"""
Utilitaires de retry pour les appels API avec exponential backoff.
"""

import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
from src.core import get_logger

logger = get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Décorateur pour retry avec exponential backoff.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        initial_delay: Délai initial en secondes
        max_delay: Délai maximum en secondes
        exponential_base: Base pour l'exponentielle
        exceptions: Tuple d'exceptions à capturer
    
    Returns:
        Fonction décorée avec retry
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"Échec après {max_attempts} tentatives pour {func.__name__}: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Tentative {attempt}/{max_attempts} échouée pour {func.__name__}: {e}. "
                        f"Retry dans {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
            
            # Ne devrait jamais arriver ici, mais pour mypy
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic error")
        
        return wrapper
    return decorator
