"""
Module pour nettoyer et normaliser le texte extrait des PDFs.
Améliore la qualité des données avant chunking et vectorisation.
"""

import re


def remove_page_numbers(text: str) -> str:
    """
    Supprime les numéros de page (ex: '1 / 7', 'Page 1/23', '- 5 -').
    """
    # Pattern pour différents formats de numéros de page
    patterns = [
        r'^\s*\d+\s*/\s*\d+\s*$',  # 1 / 7
        r'^\s*Page\s+\d+\s*/\s*\d+\s*$',  # Page 1/23
        r'^\s*-\s*\d+\s*-\s*$',  # - 5 -
        r'^\s*\d+\s*$',  # Juste un chiffre seul
    ]

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        is_page_number = False
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_page_number = True
                break

        if not is_page_number:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_figure_references(text: str) -> str:
    """
    Supprime les références aux figures (ex: 'Figure 1:', 'Fig. 2 :', 'Image: pixabay').
    """
    lines = text.split('\n')
    cleaned_lines = []
    skip_next = False

    for line in lines:
        stripped = line.strip()

        # Pattern pour détecter les références aux figures
        figure_patterns = [
            r'^Figure\s+\d+\s*:.*$',  # Figure 1: description
            r'^Fig\.\s+\d+\s*:.*$',   # Fig. 2: description
            r'^Fig\s+\d+\s*:.*$',     # Fig 2: description
            r'^Image\s*:.*$',         # Image: pixabay
            r'^Graphique\s+\d+\s*:.*$',  # Graphique 1: description
        ]

        is_figure = False
        for pattern in figure_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_figure = True
                break

        if not is_figure:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_update_dates(text: str) -> str:
    """
    Supprime les lignes de mise à jour (ex: 'Mise à jour : Avril 2018', 'Dernière mise à jour: septembre 2018').
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Pattern pour détecter les lignes de mise à jour
        update_patterns = [
            r'^Mise\s+à\s+jour\s*:.*$',  # Mise à jour : ...
            r'^Dernière\s+mise\s+à\s+jour\s*:.*$',  # Dernière mise à jour: ...
            r'^Last\s+update\s*:.*$',  # Last update: ...
            r'^Updated?\s*:.*$',  # Update: / Updated: ...
        ]

        is_update = False
        for pattern in update_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_update = True
                break

        if not is_update:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_table_of_contents(text: str) -> str:
    """
    Supprime les tables des matières et sommaires.
    """
    lines = text.split('\n')

    # Patterns pour détecter le début d'une table des matières
    toc_start_patterns = [
        r'^Table\s+des\s+matières\s*$',
        r'^Sommaire\s*$',
        r'^Table\s+of\s+contents\s*$',
        r'^Contents\s*$',
    ]

    in_toc = False
    toc_start_idx = -1
    cleaned_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Détecter le début d'une table des matières
        if not in_toc:
            for pattern in toc_start_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    in_toc = True
                    toc_start_idx = i
                    break

        # Si dans une table des matières, chercher la fin
        if in_toc:
            # La ToC se termine généralement quand on arrive à du contenu avec des paragraphes
            # ou un titre de section (###)

            # Détecter les lignes typiques de ToC (avec points et numéros de page)
            is_toc_line = bool(re.search(r'\.{3,}', stripped))  # Lignes avec ...
            is_page_indicator = bool(re.match(r'^Page\s*$', stripped, re.IGNORECASE))
            is_short_numbered = bool(re.match(r'^\d+\s*$', stripped))  # Juste un numéro

            # Continuer à skip si c'est une ligne de ToC typique ou vide
            if is_toc_line or is_page_indicator or is_short_numbered or not stripped:
                continue

            # Si on trouve un titre markdown (###) ou un paragraphe long, la ToC est finie
            if stripped.startswith('###') or (len(stripped) > 80 and '.' in stripped):
                in_toc = False
                cleaned_lines.append(line)
                continue

            # Si on a avancé de plus de 30 lignes depuis le début, considérer que la ToC est finie
            if i - toc_start_idx > 30:
                in_toc = False
                cleaned_lines.append(line)
                continue

        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_ocr_artifacts(text: str) -> str:
    """
    Supprime les artefacts OCR (caractères aléatoires, lignes courtes sans sens).
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Ignorer lignes très courtes avec peu de lettres
        if len(stripped) < 3:
            continue

        # Ignorer lignes avec trop de caractères spéciaux (artefacts OCR)
        alpha_count = sum(c.isalpha() for c in stripped)
        if alpha_count > 0 and alpha_count / len(stripped) < 0.5:
            continue

        # Ignorer lignes avec patterns d'artefacts
        if re.match(r'^[^a-zA-Z0-9\s]{3,}$', stripped):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces : supprime espaces multiples, lignes vides excessives.
    """
    # Supprimer espaces en début/fin de chaque ligne
    lines = [line.rstrip() for line in text.split('\n')]

    # Réduire espaces multiples en un seul
    lines = [re.sub(r' +', ' ', line) for line in lines]

    # Réduire lignes vides multiples à max 2
    cleaned_lines = []
    empty_count = 0

    for line in lines:
        if not line.strip():
            empty_count += 1
            if empty_count <= 2:
                cleaned_lines.append(line)
        else:
            empty_count = 0
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def fix_broken_urls(text: str) -> str:
    """
    Tente de fusionner les URLs cassées sur plusieurs lignes.
    """
    # Détecter les URLs cassées (ligne se terminant par http:// ou https://)
    text = re.sub(r'(https?://)\s*\n\s*', r'\1', text)

    # Fusionner lignes d'URL
    lines = text.split('\n')
    merged_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Si la ligne contient une URL incomplète
        if 'http' in line and not line.strip().endswith(')'):
            # Vérifier si la ligne suivante complète l'URL
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith('#'):
                merged_lines.append(line.rstrip() + lines[i + 1].lstrip())
                i += 2
                continue

        merged_lines.append(line)
        i += 1

    return '\n'.join(merged_lines)


def improve_markdown_structure(text: str) -> str:
    """
    Améliore la structure markdown : détecte les titres en majuscules, fusionne les lignes cassées.
    """
    lines = text.split('\n')
    improved_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Détecter titres en MAJUSCULES (sauf s'ils sont déjà en markdown)
        if stripped and stripped.isupper() and len(stripped.split()) <= 15:
            if not stripped.startswith('#'):
                # Transformer en titre niveau 3
                improved_lines.append(f"\n### {stripped}\n")
                continue

        improved_lines.append(line)

    # Fusionner les lignes cassées (phrase coupée)
    merged_lines = []
    i = 0

    while i < len(improved_lines):
        line = improved_lines[i]

        # Si ligne se termine par lettre/virgule et suivante commence par minuscule
        if i + 1 < len(improved_lines):
            next_line = improved_lines[i + 1].strip()

            if (line.strip() and
                not line.strip().endswith('.') and
                not line.strip().endswith(':') and
                not line.strip().startswith('#') and
                next_line and
                next_line[0].islower() and
                len(line.strip()) > 20):

                merged_lines.append(line.rstrip() + ' ' + next_line)
                i += 2
                continue

        merged_lines.append(line)
        i += 1

    return '\n'.join(merged_lines)


def clean_text(text: str, is_ocr: bool = False) -> str:
    """
    Applique tous les nettoyages sur le texte.

    Args:
        text: Texte à nettoyer
        is_ocr: True si le texte provient d'OCR (nettoyage plus agressif)

    Returns:
        Texte nettoyé
    """
    # Nettoyages de base
    text = remove_page_numbers(text)
    text = remove_figure_references(text)
    text = remove_update_dates(text)
    text = remove_table_of_contents(text)
    text = normalize_whitespace(text)
    text = fix_broken_urls(text)
    text = improve_markdown_structure(text)

    # Nettoyages spécifiques OCR
    if is_ocr:
        text = remove_ocr_artifacts(text)

    # Nettoyage final des espaces
    text = normalize_whitespace(text)

    return text.strip()


def clean_markdown_file(input_path: str, output_path: str = None, is_ocr: bool = False) -> str:
    """
    Nettoie un fichier markdown et le sauvegarde.

    Args:
        input_path: Chemin du fichier d'entrée
        output_path: Chemin du fichier de sortie (si None, écrase l'original)
        is_ocr: True si le texte provient d'OCR

    Returns:
        Chemin du fichier nettoyé
    """
    if output_path is None:
        output_path = input_path

    # Lire le fichier
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Nettoyer
    cleaned_text = clean_text(text, is_ocr=is_ocr)

    # Sauvegarder
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    return output_path
