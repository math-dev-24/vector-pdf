"""
Script de comparaison des méthodes d'extraction PDF.
Compare les 3 modes: basic, structured, pymupdf4llm
"""

import sys
from pathlib import Path
from src.extractors.text_extractor import extract_text_from_pdf
from src.extractors.text_extractor_v2 import extract_structured_text_from_pdf, extract_with_pymupdf4llm


def count_headings(markdown_content: str) -> dict:
    """
    Compte les titres dans un contenu markdown.

    Returns:
        Dict avec le nombre de titres par niveau
    """
    lines = markdown_content.split('\n')

    counts = {
        'h1': 0,  # #
        'h2': 0,  # ##
        'h3': 0,  # ###
        'h4': 0,  # ####
        'total': 0
    }

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('####'):
            counts['h4'] += 1
            counts['total'] += 1
        elif stripped.startswith('###'):
            counts['h3'] += 1
            counts['total'] += 1
        elif stripped.startswith('##'):
            counts['h2'] += 1
            counts['total'] += 1
        elif stripped.startswith('#'):
            counts['h1'] += 1
            counts['total'] += 1

    return counts


def analyze_markdown_quality(file_path: str) -> dict:
    """
    Analyse la qualité d'un fichier markdown.

    Returns:
        Dict avec métriques de qualité
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    metrics = {
        'total_chars': len(content),
        'total_lines': len(lines),
        'non_empty_lines': len([l for l in lines if l.strip()]),
        'headings': count_headings(content),
        'avg_line_length': sum(len(l) for l in lines) / len(lines) if lines else 0,
        'paragraphs': content.count('\n\n'),
    }

    return metrics


def compare_extraction_methods(pdf_path: str, output_base_dir: str = "./COMPARISON"):
    """
    Compare les 3 méthodes d'extraction sur un même PDF.

    Args:
        pdf_path: Chemin du PDF à tester
        output_base_dir: Répertoire de sortie pour les comparaisons
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"❌ Fichier non trouvé: {pdf_path}")
        return

    output_base = Path(output_base_dir)
    output_base.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print(f"📊 COMPARAISON DES MÉTHODES D'EXTRACTION")
    print("=" * 80)
    print(f"\nPDF: {pdf_path.name}")

    results = {}

    # Méthode 1: Basic
    print("\n" + "-" * 80)
    print("1️⃣  MÉTHODE BASIQUE (extraction simple)")
    print("-" * 80)

    try:
        output_dir_basic = output_base / "basic"
        output_dir_basic.mkdir(exist_ok=True)

        output_file = extract_text_from_pdf(
            str(pdf_path),
            str(output_dir_basic),
            verbose=True
        )

        metrics = analyze_markdown_quality(output_file)
        results['basic'] = {
            'file': output_file,
            'metrics': metrics,
            'status': 'success'
        }

        print(f"\n✓ Fichier créé: {output_file}")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        results['basic'] = {'status': 'error', 'error': str(e)}

    # Méthode 2: Structured
    print("\n" + "-" * 80)
    print("2️⃣  MÉTHODE STRUCTURÉE (détection de titres)")
    print("-" * 80)

    try:
        output_dir_struct = output_base / "structured"
        output_dir_struct.mkdir(exist_ok=True)

        output_file = extract_structured_text_from_pdf(
            str(pdf_path),
            str(output_dir_struct),
            verbose=True,
            auto_detect_headings=True
        )

        metrics = analyze_markdown_quality(output_file)
        results['structured'] = {
            'file': output_file,
            'metrics': metrics,
            'status': 'success'
        }

        print(f"\n✓ Fichier créé: {output_file}")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        results['structured'] = {'status': 'error', 'error': str(e)}

    # Méthode 3: PyMuPDF4LLM
    print("\n" + "-" * 80)
    print("3️⃣  MÉTHODE PYMUPDF4LLM (optimisé pour LLM)")
    print("-" * 80)

    try:
        output_dir_llm = output_base / "pymupdf4llm"
        output_dir_llm.mkdir(exist_ok=True)

        output_file = extract_with_pymupdf4llm(
            str(pdf_path),
            str(output_dir_llm),
            verbose=True
        )

        metrics = analyze_markdown_quality(output_file)
        results['pymupdf4llm'] = {
            'file': output_file,
            'metrics': metrics,
            'status': 'success'
        }

        print(f"\n✓ Fichier créé: {output_file}")

    except ImportError:
        print("\n⚠️  PyMuPDF4LLM n'est pas installé")
        print("   Installez avec: pip install pymupdf4llm")
        results['pymupdf4llm'] = {'status': 'not_installed'}
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        results['pymupdf4llm'] = {'status': 'error', 'error': str(e)}

    # Résumé comparatif
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ COMPARATIF")
    print("=" * 80)

    print(f"\n{'Méthode':<20} {'Caractères':<15} {'Lignes':<10} {'Titres':<10} {'Paragraphes':<12}")
    print("-" * 80)

    for method_name, result in results.items():
        if result['status'] == 'success':
            m = result['metrics']
            print(f"{method_name:<20} {m['total_chars']:<15,} {m['total_lines']:<10} "
                  f"{m['headings']['total']:<10} {m['paragraphs']:<12}")
        elif result['status'] == 'not_installed':
            print(f"{method_name:<20} {'(non installé)'}")
        else:
            print(f"{method_name:<20} {'(erreur)'}")

    # Détails des titres
    print("\n" + "-" * 80)
    print("DÉTAIL DES TITRES DÉTECTÉS")
    print("-" * 80)

    for method_name, result in results.items():
        if result['status'] == 'success':
            headings = result['metrics']['headings']
            print(f"\n{method_name.upper()}:")
            print(f"  - H1 (#): {headings['h1']}")
            print(f"  - H2 (##): {headings['h2']}")
            print(f"  - H3 (###): {headings['h3']}")
            print(f"  - H4 (####): {headings['h4']}")
            print(f"  - TOTAL: {headings['total']}")

    # Recommandation
    print("\n" + "=" * 80)
    print("💡 RECOMMANDATION")
    print("=" * 80)

    successful = [k for k, v in results.items() if v['status'] == 'success']

    if len(successful) > 1:
        # Comparer le nombre de titres détectés
        title_counts = {
            method: results[method]['metrics']['headings']['total']
            for method in successful
        }

        best_method = max(title_counts, key=title_counts.get)

        print(f"\n✨ Méthode recommandée: {best_method.upper()}")
        print(f"   {title_counts[best_method]} titres détectés")

        print("\n📝 Analyse:")

        if title_counts['basic'] == 0 and title_counts[best_method] > 0:
            print("   ✓ La méthode structurée détecte bien les titres (absent en mode basique)")
        elif title_counts[best_method] > title_counts['basic']:
            print("   ✓ Meilleure détection de la structure avec la méthode avancée")

        print("\n💡 Pour LangChain/RAG:")
        print("   - La détection de titres améliore le chunking intelligent")
        print("   - Les titres permettent de préserver le contexte dans les chunks")
        print("   - Meilleure performance pour la recherche sémantique")

    print("\n📂 Fichiers générés dans: " + str(output_base))
    print("   Comparez les fichiers pour choisir la méthode adaptée à vos PDFs")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python compare_extraction_methods.py <pdf_path>")
        print("\nExemple:")
        print("  python compare_extraction_methods.py ./DATA/mon_document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    compare_extraction_methods(pdf_path)
