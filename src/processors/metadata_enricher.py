"""
Module pour enrichir les métadonnées des chunks avec de l'information sémantique.
Utilise OpenAI pour extraire entités, mots-clés, résumés, etc.
"""

import os
import re
import json
from typing import Dict, List, Optional
from openai import OpenAI

from src.core import get_logger, ProgressBar

logger = get_logger(__name__)

RAG_LABELS = {
    "norme_reglementation",
    "dimensionnement",
    "depannage",
    "installation_mise_en_service",
    "maintenance",
    "securite",
    "fluide_frigorigene",
    "general",
}

DOMAIN_TAG_PATTERNS = {
    "froid": r"\bfroid|frigorifique|refrigeration|réfrigération|groupe froid|chambre froide",
    "climatisation": r"\bclimatisation|clim\b|split|vrv|drv|cta|traitement d'air",
    "pompe_a_chaleur": r"pompe à chaleur|\bpac\b|aerothermie|aérothermie",
    "fluide_frigorigene": r"\bfluide|frigorig[eè]ne|r32|r410a|r407c|r134a|co2|r744|propane|r290|hfo|hfc",
    "pression": r"\bpression|bar\b|épreuve|epreuve|soupape|détendeur|detendeur",
    "temperature": r"\btemp[ée]rature|surchauffe|sous-refroidissement|évaporation|evaporation|condensation",
    "debit": r"\bd[ée]bit|m3/h|m³/h|l/s|kg/h",
    "puissance": r"\bpuissance|kw\b|watt|capacit[ée]|charge thermique",
    "norme": r"\bnorme|nf en|en 378|atex|erp|r[ée]glement|f-gas|ce\b|directive",
    "maintenance": r"\bmaintenance|entretien|contr[ôo]le|inspection|nettoyage|v[ée]rification",
}

LABEL_PATTERNS = [
    ("norme_reglementation", r"\bnorme|nf en|en 378|r[ée]glement|directive|arr[êe]t[ée]|obligation|f-gas|conformit[ée]"),
    ("dimensionnement", r"\bdimensionnement|calcul|s[ée]lection|puissance|d[ée]bit|perte de charge|charge thermique|abaque|formule"),
    ("depannage", r"\bd[ée]pannage|diagnostic|panne|d[ée]faut|alarme|code erreur|sympt[oô]me|rem[eè]de|cause"),
    ("installation_mise_en_service", r"\binstallation|mise en service|raccordement|montage|implantation|tirage au vide|brasage|charge"),
    ("maintenance", r"\bmaintenance|entretien|inspection|nettoyage|contr[ôo]le p[ée]riodique|v[ée]rification"),
    ("securite", r"\bs[ée]curit[ée]|danger|risque|toxicit[ée]|inflammable|pression maximale|epi|habilitation|consigne"),
    ("fluide_frigorigene", r"\bfluide|frigorig[eè]ne|r32|r410a|r407c|r134a|co2|r744|r290|hfo|hfc|prg|gwp"),
]


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

    def detect_domain_tags(self, text: str) -> List[str]:
        """Retourne des tags métier froid/climatisation détectés par heuristiques."""
        normalized = text.lower()
        tags = [
            tag
            for tag, pattern in DOMAIN_TAG_PATTERNS.items()
            if re.search(pattern, normalized, re.IGNORECASE)
        ]
        return tags[:8]

    def detect_rag_label_basic(self, text: str) -> tuple[str, float]:
        """Classe un chunk dans un label métier stable, sans appel IA."""
        normalized = text.lower()
        best_label = "general"
        best_score = 0

        for label, pattern in LABEL_PATTERNS:
            score = len(re.findall(pattern, normalized, re.IGNORECASE))
            if score > best_score:
                best_label = label
                best_score = score

        if best_score >= 3:
            return best_label, 0.78
        if best_score >= 1:
            return best_label, 0.62
        return "general", 0.45

    def normalize_rag_label(self, label: str, fallback: str = "general") -> str:
        """Normalise un label IA vers l'identifiant attendu."""
        if not label:
            return fallback
        clean = str(label).strip().lower().replace(" ", "_").replace("-", "_")
        return clean if clean in RAG_LABELS else fallback

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

        prompt = f"""Analyse ce texte technique de normes, froid, refrigeration, climatisation ou HVAC et extrais les informations au format JSON.

{text_sample}

Retourne UNIQUEMENT un JSON valide (pas de markdown):
{{
  "keywords": ["mot1", "mot2"],
  "topics": ["sujet1", "sujet2"],
  "domain_tags": ["froid", "climatisation"],
  "rag_label": "norme_reglementation|dimensionnement|depannage|installation_mise_en_service|maintenance|securite|fluide_frigorigene|general",
  "rag_label_confidence": 0.0,
  "document_type": "norme|manuel|procedure|fiche_technique|rapport|autre",
  "language": "fr",
  "summary": "résumé en une phrase courte sans guillemets"
}}

Max 5 keywords, max 3 topics, max 5 domain_tags. Listes vides [] si rien trouvé."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu extrais des métadonnées structurées. "
                            "Réponds UNIQUEMENT avec du JSON valide, compact, sans markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            content = (response.choices[0].message.content or "").strip()
            metadata = self._parse_ai_json(content)
            if metadata:
                return metadata

            # Retry une fois avec un prompt minimal si échec (JSON tronqué ou invalide)
            return self._extract_with_ai_retry(text_sample)

        except Exception as e:
            logger.debug(f"Erreur extraction IA: {e}")
            return {}

    def _parse_ai_json(self, content: str) -> Dict:
        """Parse le JSON renvoyé par l'IA avec réparations légères."""
        if not content:
            return {}

        # JSON mode OpenAI : parser directement sans nettoyage agressif
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        cleaned = content
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        # Extraire le bloc JSON principal
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            fragment = cleaned[start : end + 1]
            fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
            try:
                parsed = json.loads(fragment)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError as err:
                if not fragment.rstrip().endswith("}"):
                    logger.debug(f"JSON IA probablement tronqué ({err})")
                else:
                    logger.debug(f"JSON IA invalide ({err}): {fragment[:120]}...")

        return {}

    def _extract_with_ai_retry(self, text: str) -> Dict:
        """Second essai avec un schéma minimal."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Réponds uniquement en JSON compact valide.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Extrais keywords (max 3) et topics (max 2) de ce texte:\n\n"
                            f"{text[:1200]}\n\n"
                            'Format: {"keywords":[],"topics":[],"summary":""}'
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            return self._parse_ai_json(content)
        except Exception as e:
            logger.debug(f"Retry extraction IA échoué: {e}")
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
        basic_label, basic_confidence = self.detect_rag_label_basic(text)
        enriched['rag_label'] = basic_label
        enriched['rag_label_confidence'] = basic_confidence
        enriched['domain_tags'] = self.detect_domain_tags(text)

        # Features du contenu
        enriched.update(self.detect_content_features(text))

        # Extraction IA (optionnel) — les métadonnées basiques restent disponibles si échec
        if use_ai_extraction and self.use_ai:
            ai_metadata = self.extract_with_ai(text)
            if ai_metadata:
                enriched['keywords_ai'] = ai_metadata.get('keywords', [])
                if ai_metadata.get('entities'):
                    enriched['entities_ai'] = ai_metadata['entities']
                enriched['topics'] = ai_metadata.get('topics', [])
                if ai_metadata.get('domain_tags'):
                    tags = [
                        str(tag).strip().lower().replace(" ", "_")
                        for tag in ai_metadata.get('domain_tags', [])
                        if str(tag).strip()
                    ]
                    enriched['domain_tags'] = list(dict.fromkeys(enriched.get('domain_tags', []) + tags))[:8]
                enriched['rag_label'] = self.normalize_rag_label(
                    ai_metadata.get('rag_label', ''),
                    fallback=enriched.get('rag_label', 'general'),
                )
                if ai_metadata.get('rag_label_confidence') is not None:
                    try:
                        enriched['rag_label_confidence'] = float(ai_metadata['rag_label_confidence'])
                    except (TypeError, ValueError):
                        pass
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
        verbose: bool = False,
        batch_size: Optional[int] = None,
        min_quality_for_ai: float = 0.5,
    ) -> List[Dict]:
        """
        Enrichit un lot de chunks (IA en batch pour réduire les appels API).

        Args:
            chunks: Liste de chunks avec 'content' et 'metadata'
            use_ai: Utiliser l'IA pour enrichissement
            verbose: Afficher progression
            batch_size: Taille des batchs GPT (défaut: settings.ai_enrichment_batch_size)
            min_quality_for_ai: Ne pas appeler GPT sur chunks sous ce score qualité basique

        Returns:
            Chunks avec métadonnées enrichies
        """
        from src.core import settings

        batch_size = batch_size or settings.ai_enrichment_batch_size
        enriched_chunks: List[Dict] = []
        total = len(chunks)
        progress = ProgressBar(total, prefix="Enrichissement", enabled=verbose)

        # Enrichissement basique pour tous
        basic_enriched: List[Dict] = []
        for chunk in chunks:
            meta = self.enrich_chunk_metadata(
                text=chunk['content'],
                base_metadata=chunk['metadata'],
                use_ai_extraction=False,
            )
            basic_enriched.append({'content': chunk['content'], 'metadata': meta})

        if not (use_ai and self.use_ai):
            for i, item in enumerate(basic_enriched):
                enriched_chunks.append(item)
                progress.update(i + 1)
            progress.finish("✓")
            return enriched_chunks

        # Batch IA sur chunks de qualité suffisante
        ai_candidates: List[int] = []
        for i, item in enumerate(basic_enriched):
            score = item['metadata'].get('chunk_quality_score', 1.0)
            if score >= min_quality_for_ai:
                ai_candidates.append(i)

        for batch_start in range(0, len(ai_candidates), batch_size):
            batch_indices = ai_candidates[batch_start:batch_start + batch_size]
            ai_results = self._extract_batch_with_ai(
                [basic_enriched[i]['content'] for i in batch_indices]
            )
            for idx, ai_meta in zip(batch_indices, ai_results):
                if ai_meta:
                    basic_enriched[idx]['metadata'].update({
                        'keywords_ai': ai_meta.get('keywords', []),
                        'topics': ai_meta.get('topics', []),
                        'document_type': ai_meta.get('document_type', 'unknown'),
                        'language': ai_meta.get('language', 'unknown'),
                        'summary': ai_meta.get('summary', ''),
                    })
                    if ai_meta.get('domain_tags'):
                        tags = [
                            str(tag).strip().lower().replace(" ", "_")
                            for tag in ai_meta.get('domain_tags', [])
                            if str(tag).strip()
                        ]
                        existing_tags = basic_enriched[idx]['metadata'].get('domain_tags', [])
                        basic_enriched[idx]['metadata']['domain_tags'] = list(
                            dict.fromkeys(existing_tags + tags)
                        )[:8]
                    if ai_meta.get('rag_label'):
                        current_label = basic_enriched[idx]['metadata'].get('rag_label', 'general')
                        basic_enriched[idx]['metadata']['rag_label'] = self.normalize_rag_label(
                            ai_meta.get('rag_label', ''),
                            fallback=current_label,
                        )
                    if ai_meta.get('rag_label_confidence') is not None:
                        try:
                            basic_enriched[idx]['metadata']['rag_label_confidence'] = float(
                                ai_meta['rag_label_confidence']
                            )
                        except (TypeError, ValueError):
                            pass
                    if ai_meta.get('entities'):
                        basic_enriched[idx]['metadata']['entities_ai'] = ai_meta['entities']

        for i, item in enumerate(basic_enriched):
            enriched_chunks.append(item)
            progress.update(i + 1)

        progress.finish("✓")
        return enriched_chunks

    def _extract_batch_with_ai(self, texts: List[str]) -> List[Dict]:
        """Extrait les métadonnées pour plusieurs chunks en un seul appel GPT."""
        if not texts or not self.use_ai:
            return [{} for _ in texts]

        items = []
        for i, text in enumerate(texts):
            sample = text[:1500] if len(text) > 1500 else text
            items.append({"id": i, "text": sample})

        prompt = f"""Analyse ces extraits techniques de normes, froid, refrigeration, climatisation ou HVAC et retourne un JSON:
{{"chunks": [{{"id": 0, "keywords": [], "topics": [], "domain_tags": [], "rag_label": "norme_reglementation|dimensionnement|depannage|installation_mise_en_service|maintenance|securite|fluide_frigorigene|general", "rag_label_confidence": 0.0, "document_type": "norme|manuel|procedure|fiche_technique|rapport|autre", "language": "fr", "summary": "une phrase"}}]}}

Extraits:
{json.dumps(items, ensure_ascii=False)}

Max 5 keywords, 3 topics et 5 domain_tags par chunk. Utilise rag_label comme label principal du passage."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu extrais des métadonnées structurées. JSON valide uniquement.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = self._parse_ai_json(content)
            chunk_list = parsed.get("chunks", [])
            by_id = {item.get("id", j): item for j, item in enumerate(chunk_list)}
            return [by_id.get(i, {}) for i in range(len(texts))]
        except Exception as e:
            logger.debug(f"Erreur enrichissement batch IA: {e}")
            return [{} for _ in texts]




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
