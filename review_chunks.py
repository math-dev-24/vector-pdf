#!/usr/bin/env python3
"""
Génère un échantillon de chunks pour validation humaine ponctuelle.
Export JSON + HTML pour noter la qualité du découpage RAG.
"""

import argparse
import html
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.core import settings, setup_logging, get_logger
from src.pipeline import Pipeline, PipelineConfig
from src.pipeline.models import ChunkingMode
from src.processors.state_manager import StateManager

logger = get_logger(__name__)

SAMPLE_SIZE = 20


def _flatten_chunks(cached_results: List[Dict]) -> List[Dict]:
    """Aplatit les résultats cache (liste de fichiers) en liste de chunks."""
    if not cached_results:
        return []
    if cached_results and "chunks" in cached_results[0]:
        flat: List[Dict] = []
        for file_result in cached_results:
            flat.extend(file_result.get("chunks", []))
        return flat
    return cached_results


def _stratified_sample(chunks: List[Dict], sample_size: int = SAMPLE_SIZE) -> List[Dict]:
    """Échantillon stratifié : début, milieu, fin + sections variées."""
    if len(chunks) <= sample_size:
        return chunks

    n = len(chunks)
    indices = set()

    # Début / milieu / fin
    for pos in (0, n // 4, n // 2, 3 * n // 4, n - 1):
        indices.add(pos)

    # Tableaux et sections distinctes
    sections_seen: set[str] = set()
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        section = meta.get("section_hierarchy_string", "")
        has_table = meta.get("has_table", False)
        if has_table and len(indices) < sample_size:
            indices.add(i)
        elif section and section not in sections_seen and len(indices) < sample_size:
            sections_seen.add(section)
            indices.add(i)

    remaining = sample_size - len(indices)
    pool = [i for i in range(n) if i not in indices]
    if remaining > 0 and pool:
        indices.update(random.sample(pool, min(remaining, len(pool))))

    return [chunks[i] for i in sorted(indices)]


def _chunk_to_review_item(chunk: Dict, index: int) -> Dict[str, Any]:
    meta = chunk.get("metadata", {})
    return {
        "id": index,
        "content": chunk.get("content", ""),
        "content_preview": chunk.get("content", "")[:500],
        "section": meta.get("section_hierarchy_string") or meta.get("section_title", ""),
        "chunk_title": meta.get("chunk_title", ""),
        "quality_score": meta.get("quality_score") or meta.get("chunk_quality_score"),
        "topics": meta.get("topics", []),
        "summary": meta.get("summary", ""),
        "has_table": meta.get("has_table", False),
        "token_count": meta.get("token_count"),
        "file_name": meta.get("file_name", ""),
        "review": {
            "rating": None,
            "note": "",
            "action": None,
        },
    }


def _render_html(items: List[Dict[str, Any]], output_path: Path) -> None:
    rows = []
    for item in items:
        title = html.escape(f"Chunk #{item['id']} - {item.get('file_name', '')}")
        section = html.escape(item.get('section') or '-')
        quality = html.escape(str(item.get('quality_score', '-')))
        topics = html.escape(', '.join(item.get('topics') or []) or '-')
        summary = html.escape(item.get('summary') or '-')
        preview = html.escape(item.get('content_preview', ''))
        rows.append(f"""
        <article class="chunk" data-id="{item['id']}">
          <h2>{title}</h2>
          <p><strong>Section:</strong> {section}</p>
          <p><strong>Qualité:</strong> {quality}</p>
          <p><strong>Sujets:</strong> {topics}</p>
          <p><strong>Résumé:</strong> {summary}</p>
          <pre>{preview}</pre>
          <div class="review">
            <label>Note:
              <select name="rating-{item['id']}">
                <option value="">—</option>
                <option value="good">Bon</option>
                <option value="bad">Mauvais</option>
                <option value="merge">À fusionner</option>
                <option value="split">À splitter</option>
              </select>
            </label>
          </div>
        </article>
        """)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Review chunks RAG</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    .chunk {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }}
    pre {{ background: #f6f6f6; padding: 0.75rem; overflow-x: auto; white-space: pre-wrap; }}
    h1 {{ color: #333; }}
  </style>
</head>
<body>
  <h1>Review chunks RAG ({len(items)} échantillons)</h1>
  <p>Généré le {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
  {''.join(rows)}
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def export_review_sample(
    output_dir: Path,
    review_dir: Path,
    sample_size: int = SAMPLE_SIZE,
    use_cache: bool = True,
    chunking_mode: ChunkingMode = ChunkingMode.ADVANCED,
) -> Path:
    """
    Génère l'échantillon de review à partir des markdowns ou du cache chunks.
    """
    review_dir.mkdir(parents=True, exist_ok=True)
    state = StateManager(str(settings.cache_dir))

    chunks: List[Dict] = []

    if use_cache and state.has_chunks():
        logger.info("Chargement des chunks depuis le cache")
        cached = state.load_chunks()
        chunks = _flatten_chunks(cached or [])
    else:
        logger.info("Génération des chunks via le pipeline")
        config = PipelineConfig(
            data_dir=settings.data_dir,
            output_dir=output_dir,
            chunking_mode=chunking_mode,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        pipeline = Pipeline(config=config)
        result = pipeline.chunk()
        chunks = result.chunks if result else []

    if not chunks:
        raise RuntimeError(
            f"Aucun chunk disponible. Lancez d'abord generate.py ou placez des .md dans {output_dir}"
        )

    sample = _stratified_sample(chunks, sample_size=sample_size)
    items = [_chunk_to_review_item(c, i) for i, c in enumerate(sample)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = review_dir / f"review_chunks_{timestamp}.json"
    html_path = review_dir / f"review_chunks_{timestamp}.html"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "sample_size": len(items),
        "settings": {
            "chunk_min_size": settings.chunk_min_size,
            "chunk_max_size": settings.chunk_max_size,
            "min_chunk_quality": settings.min_chunk_quality,
            "chunk_merge_strategy": settings.chunk_merge_strategy,
            "enable_ai_enrichment": settings.enable_ai_enrichment,
        },
        "items": items,
    }

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_html(items, html_path)

    logger.info(f"JSON: {json_path}")
    logger.info(f"HTML: {html_path}")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export d'un échantillon de chunks pour review humaine")
    parser.add_argument("--output-dir", type=Path, default=settings.output_dir, help="Répertoire OUTPUT/")
    parser.add_argument("--review-dir", type=Path, default=Path("./review"), help="Répertoire de sortie review")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Nombre de chunks à échantillonner")
    parser.add_argument("--no-cache", action="store_true", help="Regénérer les chunks au lieu du cache")
    parser.add_argument("--standard", action="store_true", help="Mode chunking standard (sans advanced)")
    args = parser.parse_args()

    setup_logging(level=settings.log_level)
    mode = ChunkingMode.STANDARD if args.standard else ChunkingMode.ADVANCED

    print("\n=== Review chunks RAG ===\n")
    path = export_review_sample(
        output_dir=args.output_dir,
        review_dir=args.review_dir,
        sample_size=args.sample_size,
        use_cache=not args.no_cache,
        chunking_mode=mode,
    )
    print(f"\n✅ Échantillon exporté: {path}")
    print(f"   Ouvrez le fichier .html associé dans {args.review_dir}")
    print("\nAprès review, ajustez dans .env :")
    print("  MIN_CHUNK_QUALITY, CHUNK_MIN_SIZE, CHUNK_MERGE_STRATEGY, ENABLE_AI_ENRICHMENT\n")


if __name__ == "__main__":
    main()
