"""
Classification automatique des chunks en namespaces via OpenAI.

Les namespaces peuvent être configurés via NAMESPACE_DEFINITIONS dans .env.
Par défaut, 3 namespaces sont utilisés :
- depannage      : diagnostic, résolution de pannes, erreurs, maintenance corrective
- dimensionnement: calculs, spécifications, sélection de matériel, capacité, débits
- general        : tout le reste (intro, contexte, normes générales, administratif)
"""

import json
import re
import unicodedata
from typing import List, Dict, Tuple
from src.core import OpenAIClient, get_logger, settings

logger = get_logger(__name__)

# Namespaces par défaut
_DEFAULT_NS_CONFIG = {
    "Dépannage": {
        "id": "depannage",
        "description": (
            "dépannage, diagnostic de pannes, codes d'erreur, symptômes/causes/remèdes, "
            "procédures de maintenance corrective, résolution de problèmes"
        ),
    },
    "Dimensionnement": {
        "id": "dimensionnement",
        "description": (
            "calculs de dimensionnement, sélection de matériel, spécifications techniques, "
            "débits, pressions, puissances, tableaux de sélection, formules de calcul"
        ),
    },
    "Général": {
        "id": "general",
        "description": (
            "tout ce qui ne relève pas des deux catégories précédentes — présentation du produit, "
            "introduction, normes générales, sécurité générale, glossaire, informations "
            "administratives, tables des matières"
        ),
    },
}


def _sanitize_ns_id(label: str) -> str:
    """Convertit un libellé en identifiant Pinecone valide (ASCII, sans accents)."""
    normalized = unicodedata.normalize("NFKD", label.lower())
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9_]", "_", ascii_str).strip("_")


def _get_active_config() -> Tuple[dict, set, dict]:
    """
    Retourne (label_to_ns, valid_namespaces, ns_config) depuis settings ou les defaults.

    ns_config format: {label: {"id": str, "description": str}}
    """
    if settings.namespace_definitions:
        try:
            custom = json.loads(settings.namespace_definitions)
            label_to_ns: dict = {}
            valid_namespaces: set = set()
            ns_config: dict = {}

            for label, cfg in custom.items():
                if not isinstance(cfg, dict):
                    raise ValueError(f"Entrée invalide pour '{label}': attendu un objet dict")
                ns_id = cfg.get("id") or _sanitize_ns_id(label)
                description = cfg.get("description", label)
                ns_config[label] = {"id": ns_id, "description": description}
                label_to_ns[label] = ns_id
                label_to_ns[label.strip()] = ns_id
                label_to_ns[label.lower()] = ns_id
                label_to_ns[ns_id] = ns_id
                valid_namespaces.add(ns_id)

            logger.info(
                f"Namespaces personnalisés chargés: {list(valid_namespaces)}"
            )
            return label_to_ns, valid_namespaces, ns_config

        except Exception as e:
            logger.warning(
                f"namespace_definitions invalide ({e}). "
                "Utilisation des namespaces par défaut."
            )

    # Defaults
    label_to_ns = {}
    valid_namespaces = set()
    for label, cfg in _DEFAULT_NS_CONFIG.items():
        ns_id = cfg["id"]
        label_to_ns[label] = ns_id
        label_to_ns[label.strip()] = ns_id
        label_to_ns[label.lower()] = ns_id
        label_to_ns[ns_id] = ns_id
    valid_namespaces = {cfg["id"] for cfg in _DEFAULT_NS_CONFIG.values()}
    return label_to_ns, valid_namespaces, _DEFAULT_NS_CONFIG


def _build_system_prompt(ns_config: dict) -> str:
    """Construit le prompt système de classification à partir de la config active."""
    labels = list(ns_config.keys())
    labels_quoted = ", ".join(f'"{l}"' for l in labels)

    ns_lines = []
    for label, cfg in ns_config.items():
        ns_lines.append(f'- "{label}": {cfg["description"]}.')
    ns_block = "\n".join(ns_lines)

    example_vals = [f'"{l}"' for l in labels[:3]]
    example_json = '{"classifications": [' + ", ".join(example_vals) + "]}"

    return f"""\
Tu es un assistant expert en classification de documents techniques.

Tu reçois une liste de passages (chunks) extraits de documents PDF techniques.
Pour chaque chunk, tu dois l'assigner à EXACTEMENT l'un de ces {len(labels)} namespaces :

{ns_block}

Règles STRICTES :
1. Tu dois retourner un objet JSON avec une seule clé "classifications".
2. "classifications" est un tableau de N chaînes, où N est le nombre de chunks fournis.
3. Chaque valeur doit être EXACTEMENT l'un de : {labels_quoted}.
4. Ne génère aucun autre texte en dehors du JSON.

Exemple de réponse pour 3 chunks :
{example_json}
"""


def classify_chunks(
    chunks: List[Dict],
    batch_size: int = 25,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> List[str]:
    """
    Classifie chaque chunk dans l'un des namespaces configurés via OpenAI.

    Les namespaces sont lus depuis settings.namespace_definitions (JSON) ou
    utilisent les 3 namespaces par défaut si non défini.

    Args:
        chunks: Liste de dicts avec une clé 'content'
        batch_size: Nombre de chunks par appel API (défaut : 25)
        model: Modèle OpenAI à utiliser
        verbose: Afficher la progression

    Returns:
        Liste de namespaces dans le même ordre que `chunks`
    """
    label_to_ns, valid_namespaces, ns_config = _get_active_config()
    system_prompt = _build_system_prompt(ns_config)
    fallback_ns = next(iter(valid_namespaces))  # premier namespace comme fallback

    client = OpenAIClient().client
    total = len(chunks)
    results: List[str] = []

    if verbose:
        print(f"\n=== Classification IA des namespaces ===")
        print(f"Modèle : {model} | Chunks : {total} | Batch : {batch_size}")
        print(f"Namespaces : {list(valid_namespaces)}")

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        batch_labels = _classify_batch(
            client, batch, model, system_prompt, label_to_ns, valid_namespaces, fallback_ns
        )
        results.extend(batch_labels)

        if verbose:
            done = min(batch_start + batch_size, total)
            print(f"  [{done}/{total}] classifiés", end="\r")

    if verbose:
        print()
        _print_distribution(results, total, valid_namespaces)

    return results


def _classify_batch(
    client,
    batch: List[Dict],
    model: str,
    system_prompt: str,
    label_to_ns: dict,
    valid_namespaces: set,
    fallback_ns: str,
) -> List[str]:
    """
    Envoie un batch de chunks à l'API et retourne la liste de namespaces.
    Fallback sur le premier namespace valide si la réponse est malformée.
    """
    user_lines = []
    for i, chunk in enumerate(batch):
        content = chunk.get("content", "").strip()
        snippet = content[:600].replace("\n", " ")
        user_lines.append(f"[{i}] {snippet}")

    user_message = "\n".join(user_lines)

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        classifications = data.get("classifications", [])

        if len(classifications) != len(batch):
            logger.warning(
                f"Réponse IA : {len(classifications)} classifications pour {len(batch)} chunks. "
                f"Fallback sur '{fallback_ns}'."
            )
            return [fallback_ns] * len(batch)

        return [_label_to_namespace(label, label_to_ns, valid_namespaces, fallback_ns) for label in classifications]

    except Exception as e:
        logger.error(f"Erreur lors de la classification IA : {e}. Fallback sur '{fallback_ns}'.")
        return [fallback_ns] * len(batch)


def _label_to_namespace(label: str, label_to_ns: dict, valid_namespaces: set, fallback_ns: str) -> str:
    """Convertit le libellé retourné par l'IA en namespace Pinecone valide."""
    ns = label_to_ns.get(label.strip()) or label_to_ns.get(label.strip().lower())
    return ns if ns in valid_namespaces else fallback_ns


def _print_distribution(results: List[str], total: int, valid_namespaces: set) -> None:
    """Affiche la répartition des namespaces après classification."""
    from collections import Counter
    counts = Counter(results)
    print(f"\nRépartition des {total} chunks :")
    for ns in sorted(valid_namespaces):
        count = counts.get(ns, 0)
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct / 5)
        print(f"  {ns:<20} {count:>4} chunks  {pct:5.1f}%  {bar}")
