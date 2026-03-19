"""
Module pour enrichir les métadonnées des chunks avec de l'information sémantique.
Utilise OpenAI pour extraire entités, mots-clés, résumés, etc.
"""

import os
import re
import json
from typing import Dict, List
from openai import OpenAI

from src.core import get_logger

logger = get_logger(__name__)


class MetadataEnricher:
    """Enrichit les métadonnées des chunks avec extraction IA."""

    def __init__(self, use_ai: bool = True, model: str = "gpt-4o-mini"):
        """
        Initialise l'enrichisseur de métadonnées.

        Args:
            use_ai: Utiliser l'IA pour enrichissement (sinon extraction basique)
            model: Modèle OpenAI à utiliser (gpt-4o-mini recommandé pour coût/performance)
        """
        self.use_ai = use_ai
        self.model = model

        if use_ai:
            from src.core import settings
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY non trouvée, enrichissement IA désactivé")
                self.use_ai = False
            else:
                self.client = OpenAI(api_key=api_key)

    def extract_keywords_basic(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extraction basique de mots-clés (sans IA) basée sur fréquence.

        Args:
            text: Texte à analyser
            max_keywords: Nombre maximum de mots-clés

        Returns:
            Liste de mots-clés
        """
        # Nettoyer et tokenizer
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())

        # Mots vides français
        stop_words = {
            'être', 'avoir', 'faire', 'dire', 'pouvoir', 'aller', 'voir', 'savoir',
            'vouloir', 'venir', 'falloir', 'devoir', 'croire', 'trouver', 'donner',
            'prendre', 'parler', 'aimer', 'passer', 'mettre', 'cette', 'celui',
            'celle', 'ceux', 'celles', 'quel', 'quelle', 'quels', 'quelles',
            'dans', 'pour', 'avec', 'sans', 'sous', 'vers', 'chez', 'plus',
            'très', 'bien', 'tout', 'tous', 'toute', 'toutes', 'ainsi', 'aussi',
            'donc', 'mais', 'puis', 'alors', 'encore', 'enfin', 'comme', 'depuis'
        }

        # Compter fréquences
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Trier par fréquence
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return [word for word, freq in sorted_words[:max_keywords]]

    def extract_entities_basic(self, text: str) -> Dict[str, List[str]]:
        """
        Extraction basique d'entités (sans IA) avec regex.

        Args:
            text: Texte à analyser

        Returns:
            Dictionnaire {type: [entités]}
        """
        entities = {
            'dates': [],
            'numbers': [],
            'emails': [],
            'organizations': []
        }

        # Dates (formats: DD/MM/YYYY, YYYY-MM-DD, etc.)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b'
        ]
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))

        # Nombres importants (montants, pourcentages)
        entities['numbers'].extend(re.findall(r'\b\d+[,.]?\d*\s*(?:€|EUR|USD|\$|%)\b', text))

        # Emails
        entities['emails'].extend(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))

        # Organisations (mots en majuscules de 2+ lettres)
        entities['organizations'].extend(re.findall(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', text))

        # Dédupliquer
        for key in entities:
            entities[key] = list(set(entities[key]))[:5]  # Max 5 par type

        return entities

    def extract_with_ai(self, text: str, max_length: int = 2000) -> Dict:
        """
        Extraction avancée avec IA (OpenAI).

        Args:
            text: Texte à analyser
            max_length: Longueur max du texte (pour éviter tokens excessifs)

        Returns:
            Dictionnaire avec métadonnées enrichies
        """
        if not self.use_ai:
            return {}

        # Tronquer si trop long
        text_sample = text[:max_length] if len(text) > max_length else text

        prompt = f"""Analyse ce texte et extrais les informations suivantes au format JSON:

{text_sample}

Retourne UNIQUEMENT un JSON valide avec cette structure (pas de markdown, pas d'explication):
{{
  "keywords": ["mot-clé1", "mot-clé2", "mot-clé3"],
  "entities": {{
    "persons": ["personne1", "personne2"],
    "organizations": ["org1", "org2"],
    "locations": ["lieu1", "lieu2"],
    "dates": ["date1", "date2"]
  }},
  "topics": ["sujet1", "sujet2"],
  "document_type": "type de document (rapport, facture, contrat, article, etc.)",
  "language": "code langue (fr, en, etc.)",
  "summary": "résumé en 1 phrase courte"
}}

Limites: max 5 par catégorie. Si aucune info, retourne liste vide []."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en extraction d'information. Réponds UNIQUEMENT avec du JSON valide, sans markdown ni texte explicatif. Ne mets JAMAIS de virgule en fin de ligne avant } ou ]."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}  # Force JSON valide
            )

            content = response.choices[0].message.content.strip()

            # Nettoyer le markdown si présent
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = re.sub(r'^```\s*', '', content)  # Aussi pour ``` sans json

            # Nettoyer les virgules en fin de ligne (trailing commas)
            # Remplacer les virgules suivies d'un saut de ligne et d'un } ou ]
            content = re.sub(r',\s*\n\s*([}\]])', r'\1', content)
            # Remplacer les virgules juste avant } ou ]
            content = re.sub(r',\s*([}\]])', r'\1', content)

            try:
                metadata = json.loads(content)
            except json.JSONDecodeError as json_err:
                # Tentative de réparation supplémentaire
                # Supprimer les commentaires JSON (non standard mais parfois présents)
                content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
                content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
                
                try:
                    metadata = json.loads(content)
                except json.JSONDecodeError:
                    # Si ça échoue encore, essayer d'extraire juste les parties valides
                    logger.warning(f"Erreur parsing JSON après nettoyage: {json_err}")
                    logger.debug(f"Contenu problématique: {content[:200]}...")
                    return {}

            return metadata

        except Exception as e:
            logger.warning(f"Erreur extraction IA: {e}")
            return {}

    def detect_content_features(self, text: str) -> Dict[str, bool]:
        """
        Détecte les features du contenu (tableaux, listes, code, etc.).

        Args:
            text: Texte à analyser

        Returns:
            Dictionnaire de booléens pour chaque feature
        """
        features = {
            'has_table': False,
            'has_list': False,
            'has_code': False,
            'has_math': False,
            'has_citations': False,
            'has_footnotes': False
        }

        # Tableaux markdown ou détection de colonnes
        if re.search(r'\|.*\|.*\|', text) or re.search(r'\t.*\t.*\t', text):
            features['has_table'] = True

        # Listes (markdown ou numérotées)
        if re.search(r'^\s*[-*+]\s+', text, re.MULTILINE) or re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            features['has_list'] = True

        # Code (blocs ou inline)
        if '```' in text or re.search(r'`[^`]+`', text):
            features['has_code'] = True

        # Formules mathématiques
        if re.search(r'\$.*\$|\\frac|\\sum|\\int|≈|≤|≥|∑|∫', text):
            features['has_math'] = True

        # Citations (guillemets ou format académique)
        if re.search(r'«.*»|".*"|\'.*\'|\[?\d+\]', text):
            features['has_citations'] = True

        # Notes de bas de page
        if re.search(r'\[\^\d+\]|^\d+\.\s*(?:[A-Z]|Note)', text, re.MULTILINE):
            features['has_footnotes'] = True

        return features

    def calculate_quality_score(self, text: str, metadata: Dict) -> float:
        """
        Calcule un score de qualité pour le chunk (0-1).

        Args:
            text: Texte du chunk
            metadata: Métadonnées du chunk

        Returns:
            Score de qualité (0-1)
        """
        score = 1.0

        # Pénalités
        text_length = len(text.strip())

        # Trop court ou trop long
        if text_length < 100:
            score -= 0.3
        elif text_length > 3000:
            score -= 0.2

        # Ratio caractères alphabétiques (détecter artefacts)
        alpha_count = sum(c.isalpha() for c in text)
        if text_length > 0:
            alpha_ratio = alpha_count / text_length
            if alpha_ratio < 0.5:  # Trop de caractères spéciaux
                score -= 0.3

        # Phrases incomplètes (pas de ponctuation finale)
        if text.strip() and text.strip()[-1] not in '.!?':
            score -= 0.1

        # Bonus si métadonnées riches
        if metadata.get('keywords'):
            score += 0.1
        if metadata.get('entities'):
            score += 0.1

        return max(0.0, min(1.0, score))

    def enrich_chunk_metadata(
        self,
        text: str,
        base_metadata: Dict,
        use_ai_extraction: bool = True
    ) -> Dict:
        """
        Enrichit les métadonnées d'un chunk.

        Args:
            text: Contenu du chunk
            base_metadata: Métadonnées de base existantes
            use_ai_extraction: Utiliser l'IA pour extraction avancée

        Returns:
            Métadonnées enrichies
        """
        enriched = base_metadata.copy()

        # Extraction basique (toujours)
        enriched['keywords'] = self.extract_keywords_basic(text)
        enriched['entities_basic'] = self.extract_entities_basic(text)

        # Features du contenu
        enriched.update(self.detect_content_features(text))

        # Extraction IA (optionnel)
        if use_ai_extraction and self.use_ai:
            ai_metadata = self.extract_with_ai(text)
            if ai_metadata:
                enriched['keywords_ai'] = ai_metadata.get('keywords', [])
                enriched['entities_ai'] = ai_metadata.get('entities', {})
                enriched['topics'] = ai_metadata.get('topics', [])
                enriched['document_type'] = ai_metadata.get('document_type', 'unknown')
                enriched['language'] = ai_metadata.get('language', 'unknown')
                enriched['summary'] = ai_metadata.get('summary', '')

        # Score de qualité
        enriched['chunk_quality_score'] = self.calculate_quality_score(text, enriched)

        # Longueur et stats
        enriched['char_count'] = len(text)
        enriched['word_count'] = len(text.split())
        enriched['sentence_count'] = len(re.findall(r'[.!?]+', text))

        return enriched

    def enrich_batch(
        self,
        chunks: List[Dict],
        use_ai: bool = True,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Enrichit un lot de chunks.

        Args:
            chunks: Liste de chunks avec 'content' et 'metadata'
            use_ai: Utiliser l'IA pour enrichissement
            verbose: Afficher progression

        Returns:
            Chunks avec métadonnées enrichies
        """
        enriched_chunks = []

        for i, chunk in enumerate(chunks):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Enrichissement: {i+1}/{len(chunks)} chunks traités")

            enriched_metadata = self.enrich_chunk_metadata(
                text=chunk['content'],
                base_metadata=chunk['metadata'],
                use_ai_extraction=use_ai
            )

            enriched_chunks.append({
                'content': chunk['content'],
                'metadata': enriched_metadata
            })

        if verbose:
            print(f"  ✅ {len(chunks)} chunks enrichis")

        return enriched_chunks




if __name__ == "__main__":
    # Test
    enricher = MetadataEnricher(use_ai=True)

    test_text = """
    ### Rapport financier Q4 2024

    L'entreprise TechCorp a enregistré un chiffre d'affaires de 2.5M€ au quatrième trimestre 2024.
    Cette croissance de 15% par rapport à l'année précédente démontre la solidité du modèle économique.

    Contact: contact@techcorp.fr
    Date: 15/01/2025
    """

    metadata = enricher.enrich_chunk_metadata(
        text=test_text,
        base_metadata={'source': 'test.pdf', 'chunk_index': 0},
        use_ai_extraction=True
    )

    print("Métadonnées enrichies:")
    import json
    print(json.dumps(metadata, indent=2, ensure_ascii=False))