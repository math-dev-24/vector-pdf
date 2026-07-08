#!/usr/bin/env python3
"""
Evalue la pertinence retrieval du RAG sur un jeu de questions metier.

Le script ne genere pas de reponse LLM. Il mesure si les bons chunks/sources
remontent dans le top-k, puis exporte un rapport JSON + HTML.
"""

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ask import ALL_NAMESPACES, query_vector_db
from src.core import settings, setup_logging, get_logger
from src.vectorization.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class EvalSummary:
    total: int
    hit_at_k: int
    source_match_at_k: int
    label_match_at_k: int
    tag_match_at_k: int
    weak_results: int
    average_best_score: float

    def to_dict(self) -> Dict[str, Any]:
        def ratio(value: int) -> float:
            return round(value / self.total, 3) if self.total else 0.0

        return {
            "total": self.total,
            "hit_at_k": self.hit_at_k,
            "hit_at_k_rate": ratio(self.hit_at_k),
            "source_match_at_k": self.source_match_at_k,
            "source_match_at_k_rate": ratio(self.source_match_at_k),
            "label_match_at_k": self.label_match_at_k,
            "label_match_at_k_rate": ratio(self.label_match_at_k),
            "tag_match_at_k": self.tag_match_at_k,
            "tag_match_at_k_rate": ratio(self.tag_match_at_k),
            "weak_results": self.weak_results,
            "weak_results_rate": ratio(self.weak_results),
            "average_best_score": round(self.average_best_score, 4),
        }


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        questions = payload.get("questions", [])
    else:
        questions = payload

    if not isinstance(questions, list):
        raise ValueError("Le fichier doit contenir une liste ou une cle 'questions'.")

    cleaned = []
    for i, item in enumerate(questions, 1):
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError(f"Question invalide a l'index {i}: champ 'question' requis.")
        cleaned.append(item)

    return cleaned


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    return [str(v).lower() for v in value if str(v).strip()]


def _metadata_blob(metadata: Dict[str, Any]) -> str:
    parts = [
        metadata.get("file_name", ""),
        metadata.get("source", ""),
        metadata.get("section_hierarchy", ""),
        metadata.get("section_title", ""),
        metadata.get("summary", ""),
        metadata.get("text", ""),
        metadata.get("display_text", ""),
        " ".join(metadata.get("topics", []) or []),
        " ".join(metadata.get("keywords", []) or []),
        " ".join(metadata.get("domain_tags", []) or []),
    ]
    return "\n".join(str(p) for p in parts if p).lower()


def _contains_any(blob: str, expected: List[str]) -> bool:
    return bool(expected) and any(item in blob for item in expected)


def _match_labels(matches: List[Dict[str, Any]], expected_labels: List[str]) -> bool:
    if not expected_labels:
        return False
    expected = set(expected_labels)
    for match in matches:
        label = str(match.get("metadata", {}).get("rag_label", "")).lower()
        if label in expected:
            return True
    return False


def _match_tags(matches: List[Dict[str, Any]], expected_tags: List[str]) -> bool:
    if not expected_tags:
        return False
    expected = set(expected_tags)
    for match in matches:
        tags = {
            str(tag).lower()
            for tag in match.get("metadata", {}).get("domain_tags", []) or []
        }
        if tags.intersection(expected):
            return True
    return False


def _score_question(
    question: Dict[str, Any],
    matches: List[Dict[str, Any]],
    weak_score_threshold: float,
) -> Dict[str, Any]:
    expected_sources = _normalize_list(question.get("expected_sources"))
    expected_keywords = _normalize_list(question.get("expected_keywords"))
    expected_labels = _normalize_list(question.get("expected_labels"))
    expected_tags = _normalize_list(question.get("expected_tags"))

    source_match = False
    keyword_match = False

    for match in matches:
        blob = _metadata_blob(match.get("metadata", {}))
        source_match = source_match or _contains_any(blob, expected_sources)
        keyword_match = keyword_match or _contains_any(blob, expected_keywords)

    label_match = _match_labels(matches, expected_labels)
    tag_match = _match_tags(matches, expected_tags)
    has_expectations = any([expected_sources, expected_keywords, expected_labels, expected_tags])
    hit = source_match or keyword_match or label_match or tag_match
    best_score = matches[0].get("score", 0.0) if matches else 0.0

    return {
        "source_match": source_match,
        "keyword_match": keyword_match,
        "label_match": label_match,
        "tag_match": tag_match,
        "hit": hit if has_expectations else None,
        "best_score": best_score,
        "weak": not matches or best_score < weak_score_threshold,
    }


def _compact_match(match: Dict[str, Any], rank: int) -> Dict[str, Any]:
    metadata = match.get("metadata", {})
    return {
        "rank": rank,
        "score": match.get("score", 0.0),
        "namespace": metadata.get("namespace", ""),
        "source": metadata.get("source", ""),
        "file_name": metadata.get("file_name", ""),
        "chunk_index": metadata.get("chunk_index", ""),
        "section": metadata.get("section_hierarchy") or metadata.get("section_title", ""),
        "rag_label": metadata.get("rag_label", ""),
        "rag_label_confidence": metadata.get("rag_label_confidence"),
        "domain_tags": metadata.get("domain_tags", []),
        "topics": metadata.get("topics", []),
        "summary": metadata.get("summary", ""),
        "preview": (metadata.get("display_text") or metadata.get("text") or "")[:900],
    }


def evaluate_questions(
    questions: List[Dict[str, Any]],
    vector_store: VectorStore,
    namespace: str,
    top_k: int,
    weak_score_threshold: float,
) -> Dict[str, Any]:
    rows = []

    for i, question in enumerate(questions, 1):
        qid = question.get("id") or f"q{i:03d}"
        text = question["question"]
        question_namespace = question.get("namespace", namespace)
        logger.info(f"[{i}/{len(questions)}] {qid}: {text}")

        results = query_vector_db(
            question=text,
            vector_store=vector_store,
            namespace=question_namespace,
            top_k=top_k,
            verbose=False,
        )
        matches = results.get("matches", [])
        scoring = _score_question(question, matches, weak_score_threshold)

        rows.append({
            "id": qid,
            "question": text,
            "namespace": question_namespace,
            "expected_sources": question.get("expected_sources", []),
            "expected_keywords": question.get("expected_keywords", []),
            "expected_labels": question.get("expected_labels", []),
            "expected_tags": question.get("expected_tags", []),
            "scoring": scoring,
            "matches": [
                _compact_match(match, rank)
                for rank, match in enumerate(matches, 1)
            ],
        })

    total = len(rows)
    scored_rows = [row for row in rows if row["scoring"]["hit"] is not None]
    denominator = len(scored_rows) or total
    best_scores = [row["scoring"]["best_score"] for row in rows]

    summary = EvalSummary(
        total=denominator,
        hit_at_k=sum(1 for row in scored_rows if row["scoring"]["hit"]),
        source_match_at_k=sum(1 for row in scored_rows if row["scoring"]["source_match"]),
        label_match_at_k=sum(1 for row in scored_rows if row["scoring"]["label_match"]),
        tag_match_at_k=sum(1 for row in scored_rows if row["scoring"]["tag_match"]),
        weak_results=sum(1 for row in rows if row["scoring"]["weak"]),
        average_best_score=sum(best_scores) / len(best_scores) if best_scores else 0.0,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "index_name": settings.pinecone_index_name,
            "embedding_model": settings.embedding_model,
            "namespace": namespace,
            "top_k": top_k,
            "weak_score_threshold": weak_score_threshold,
        },
        "summary": summary.to_dict(),
        "questions": rows,
    }


def _render_bool(value: Any) -> str:
    if value is None:
        return "-"
    return "OK" if value else "KO"


def _render_html(report: Dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    cards = "".join(
        f"<div class='metric'><strong>{html.escape(str(key))}</strong><span>{html.escape(str(value))}</span></div>"
        for key, value in summary.items()
    )

    sections = []
    for row in report["questions"]:
        scoring = row["scoring"]
        matches_html = []
        for match in row["matches"]:
            tags = ", ".join(match.get("domain_tags") or [])
            topics = ", ".join(match.get("topics") or [])
            matches_html.append(f"""
            <article class="match">
              <h3>#{match['rank']} - score {match['score']:.4f}</h3>
              <p><strong>Source:</strong> {html.escape(match.get('file_name') or '-')}</p>
              <p><strong>Namespace:</strong> {html.escape(match.get('namespace') or '-')}</p>
              <p><strong>Section:</strong> {html.escape(str(match.get('section') or '-'))}</p>
              <p><strong>Label:</strong> {html.escape(str(match.get('rag_label') or '-'))}</p>
              <p><strong>Tags:</strong> {html.escape(tags or '-')}</p>
              <p><strong>Sujets:</strong> {html.escape(topics or '-')}</p>
              <p><strong>Resume:</strong> {html.escape(str(match.get('summary') or '-'))}</p>
              <pre>{html.escape(str(match.get('preview') or ''))}</pre>
            </article>
            """)

        status = _render_bool(scoring["hit"])
        sections.append(f"""
        <section class="question">
          <h2>{html.escape(row['id'])} - {status}</h2>
          <p class="q">{html.escape(row['question'])}</p>
          <p>
            Source: {_render_bool(scoring['source_match'])}
            | Keywords: {_render_bool(scoring['keyword_match'])}
            | Label: {_render_bool(scoring['label_match'])}
            | Tags: {_render_bool(scoring['tag_match'])}
            | Weak: {_render_bool(scoring['weak'])}
            | Best score: {scoring['best_score']:.4f}
          </p>
          {''.join(matches_html)}
        </section>
        """)

    document = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Evaluation RAG retrieval</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .75rem; margin: 1rem 0 2rem; }}
    .metric {{ border: 1px solid #ddd; border-radius: 6px; padding: .75rem; background: #fafafa; }}
    .metric strong {{ display: block; font-size: .85rem; color: #555; }}
    .metric span {{ font-size: 1.35rem; }}
    .question {{ border-top: 2px solid #222; padding-top: 1rem; margin-top: 2rem; }}
    .q {{ font-size: 1.1rem; }}
    .match {{ border: 1px solid #ddd; border-radius: 6px; padding: .75rem; margin: .75rem 0; }}
    .match h3 {{ margin-top: 0; }}
    pre {{ background: #f6f6f6; padding: .75rem; overflow-x: auto; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Evaluation RAG retrieval</h1>
  <p>Genere le {html.escape(report['generated_at'])}</p>
  <div class="metrics">{cards}</div>
  {''.join(sections)}
</body>
</html>"""
    path.write_text(document, encoding="utf-8")


def save_report(report: Dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"rag_eval_{timestamp}.json"
    html_path = output_dir / f"rag_eval_{timestamp}.html"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _render_html(report, html_path)
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalue le retrieval RAG sur un jeu de questions.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--namespace", default=ALL_NAMESPACES, help="'__all__' pour tous les namespaces")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--weak-score-threshold", type=float, default=0.25)
    args = parser.parse_args()

    setup_logging(level=settings.log_level, log_file=settings.log_file)

    if not args.questions.exists():
        example_path = args.questions.with_name("questions.example.json")
        if example_path.exists():
            print(
                f"⚠️  {args.questions} introuvable. "
                f"Utilisation de l'exemple: {example_path}"
            )
            args.questions = example_path
        else:
            raise FileNotFoundError(
                f"Fichier introuvable: {args.questions}. "
                "Copiez eval/questions.example.json vers eval/questions.json puis adaptez-le."
            )

    is_valid, missing_keys = settings.validate_api_keys()
    if not is_valid:
        raise RuntimeError(f"Cles API manquantes: {', '.join(missing_keys)}")

    questions = _load_questions(args.questions)
    vector_store = VectorStore(
        index_name=settings.pinecone_index_name,
        dimension=settings.pinecone_dimension,
    )

    report = evaluate_questions(
        questions=questions,
        vector_store=vector_store,
        namespace=args.namespace,
        top_k=args.top_k,
        weak_score_threshold=args.weak_score_threshold,
    )
    json_path, html_path = save_report(report, args.output_dir)

    print("\n=== Evaluation RAG retrieval ===")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nJSON: {json_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
