"""
Extracteur PDF utilisant Mistral AI Document OCR API.
Alternative moderne à Tesseract avec meilleure précision et extraction structurée.
"""

from pathlib import Path
from typing import Optional, Dict, List
import base64
import requests
from src.core import (
    settings, 
    get_logger, 
    PipelineError, 
    ErrorType, 
    ConfigurationError,
    retry_with_backoff
)

logger = get_logger(__name__)


class MistralOCRClient:
    """Client pour l'API Mistral OCR."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client Mistral OCR.
        
        Args:
            api_key: Clé API Mistral (défaut: settings.mistral_api_key)
        """
        self.api_key = api_key or settings.mistral_api_key
        if not self.api_key:
            raise ConfigurationError(
                "MISTRAL_API_KEY non définie. "
                "Définissez-la dans .env ou via la variable d'environnement."
            )
        
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.debug("Client Mistral OCR initialisé")
    
    @retry_with_backoff(max_attempts=3, initial_delay=2.0)
    def extract_from_pdf(
        self,
        pdf_path: Path,
        language: str = "fr",
        extract_structure: bool = True
    ) -> Dict:
        """
        Extrait le texte d'un PDF via Mistral OCR.
        
        Args:
            pdf_path: Chemin vers le PDF
            language: Langue du document (fr, en, etc.)
            extract_structure: Extraire la structure (titres, tableaux)
            
        Returns:
            Dictionnaire avec texte et métadonnées extraites
        """
        # Lire le PDF et encoder en base64
        try:
            with open(pdf_path, 'rb') as f:
                pdf_content = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            raise PipelineError(
                ErrorType.PDF_EXTRACTION,
                f"Erreur lors de la lecture du PDF {pdf_path}: {e}",
                original_error=e
            )
        
        # Préparer le prompt selon la langue
        language_prompts = {
            "fr": "Extrait tout le texte de ce document PDF en français. "
                  "Préserve la structure avec les titres, paragraphes et tableaux si présents. "
                  "Retourne le résultat en format Markdown.",
            "en": "Extract all text from this PDF document in English. "
                  "Preserve the structure with headings, paragraphs and tables if present. "
                  "Return the result in Markdown format.",
        }
        
        prompt = language_prompts.get(language, language_prompts["fr"])
        
        # Préparer la requête
        payload = {
            "model": "pixtral-12b-2409",  # Modèle avec vision pour documents
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{pdf_content}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,  # Basse température pour plus de précision
            "max_tokens": 8000  # Limite pour gros documents
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=180  # Timeout pour gros PDFs
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extraire le texte de la réponse
            text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not text:
                raise PipelineError(
                    ErrorType.PDF_EXTRACTION,
                    "Aucun texte extrait par Mistral OCR"
                )
            
            return {
                'text': text,
                'raw_response': result,
                'model_used': result.get('model', 'unknown'),
                'tokens_used': result.get('usage', {}).get('total_tokens', 0)
            }
            
        except requests.exceptions.Timeout:
            raise PipelineError(
                ErrorType.PDF_EXTRACTION,
                f"Timeout lors de l'extraction Mistral OCR pour {pdf_path.name}",
            )
        except requests.exceptions.RequestException as e:
            raise PipelineError(
                ErrorType.PDF_EXTRACTION,
                f"Erreur API Mistral OCR: {e}",
                original_error=e
            )


def extract_text_with_mistral_ocr(
    pdf_path: str,
    output_dir: str = "./OUTPUT",
    language: str = "fr",
    verbose: bool = True
) -> str:
    """
    Extrait le texte d'un PDF via Mistral OCR.
    
    Args:
        pdf_path: Chemin vers le PDF
        output_dir: Répertoire de sortie
        language: Langue du document (fr, en, etc.)
        verbose: Afficher les logs
        
    Returns:
        Chemin vers le fichier markdown créé
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{pdf_path.stem}.md"
    
    if verbose:
        logger.info(f"Extraction avec Mistral OCR: {pdf_path.name}")
    
    try:
        # Initialiser le client
        client = MistralOCRClient()
        
        # Extraire
        result = client.extract_from_pdf(
            pdf_path=pdf_path,
            language=language,
            extract_structure=True
        )
        
        # Le texte est déjà en Markdown depuis Mistral
        markdown_content = result['text']
        
        # Nettoyer le texte (moins agressif que pour OCR classique)
        from src.processors.text_cleaner import clean_text
        cleaned_content = clean_text(markdown_content, is_ocr=False)
        
        # Sauvegarder
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
        
        if verbose:
            tokens = result.get('tokens_used', 0)
            logger.info(f"✅ Sauvegardé: {len(cleaned_content)} caractères")
            logger.info(f"   Tokens utilisés: {tokens}")
            logger.info(f"   Modèle: {result.get('model_used', 'unknown')}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Erreur Mistral OCR pour {pdf_path}: {e}")
        raise PipelineError(
            ErrorType.PDF_EXTRACTION,
            f"Erreur lors de l'extraction Mistral OCR: {e}",
            original_error=e
        )
