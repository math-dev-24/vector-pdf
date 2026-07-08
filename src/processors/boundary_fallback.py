"""
Fallback LLM pour proposer des frontières de chunks quand la structure markdown est faible.
Utilisé uniquement si enable_boundary_fallback=True et score structure < seuil.
"""

import json
import re
from typing import List, Tuple

from src.core import get_logger, settings

logger = get_logger(__name__)


def compute_structure_score(text: str, num_sections: int) -> float:
    """
    Score 0-1 de la qualité structurelle du document.
    Basé sur le nombre de sections détectées vs longueur du texte.
    """
    if not text.strip():
        return 0.0

    char_count = len(text)
    expected_sections = max(3, char_count // 2500)
    if expected_sections == 0:
        return 1.0

    ratio = min(num_sections / expected_sections, 1.0)
    heading_lines = len(re.findall(r"^#{1,6}\s+\S", text, re.MULTILINE))
    heading_bonus = min(heading_lines / expected_sections, 1.0) * 0.3
    return min(1.0, ratio * 0.7 + heading_bonus)


def needs_boundary_fallback(text: str, num_sections: int) -> bool:
    """Indique si le fallback LLM est recommandé."""
    if not settings.enable_boundary_fallback:
        return False
    score = compute_structure_score(text, num_sections)
    return score < settings.boundary_fallback_section_threshold


def split_with_llm_fallback(text: str, max_chunk_size: int = 2000) -> List[Tuple[str, str]]:
    """
    Demande à GPT des points de découpage thématiques pour un texte mal structuré.

    Returns:
        Liste de (titre_suggéré, contenu)
    """
    from openai import OpenAI

    api_key = settings.openai_api_key
    if not api_key:
        logger.warning("Boundary fallback ignoré: OPENAI_API_KEY absente")
        return [("Document", text)]

    sample = text[:12000]
    client = OpenAI(api_key=api_key)

    prompt = f"""Ce texte technique manque de structure markdown claire.
Propose 3 à 8 sections cohérentes pour le RAG.

Texte:
{sample}

Réponds UNIQUEMENT en JSON:
{{"sections": [{{"title": "titre court", "start_excerpt": "20 premiers caractères exacts du début de la section"}}]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu découpes des documents techniques. JSON uniquement."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        sections = data.get("sections", [])
    except Exception as exc:
        logger.warning(f"Boundary fallback LLM échoué: {exc}")
        return [("Document", text)]

    if not sections:
        return [("Document", text)]

    chunks: List[Tuple[str, str]] = []
    positions: List[Tuple[str, int]] = []

    for sec in sections:
        title = sec.get("title", "Section")
        excerpt = sec.get("start_excerpt", "")
        if excerpt and excerpt in text:
            positions.append((title, text.index(excerpt)))

    positions.sort(key=lambda x: x[1])

    if not positions:
        return [("Document", text)]

    for i, (title, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        chunk_text = text[start:end].strip()
        if len(chunk_text) < 50:
            continue
        if len(chunk_text) > max_chunk_size:
            mid = len(chunk_text) // 2
            chunks.append((f"{title} (1/2)", chunk_text[:mid]))
            chunks.append((f"{title} (2/2)", chunk_text[mid:]))
        else:
            chunks.append((title, chunk_text))

    return chunks if chunks else [("Document", text)]
