"""
Module pour nettoyer et normaliser le texte extrait des PDFs.
Améliore la qualité des données avant chunking et vectorisation.
"""

import re
import unicodedata


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


def remove_unknown_characters(text: str, keep_common_symbols: bool = True) -> str:
    """
    Supprime ou remplace les caractères inconnus, emojis et icônes.

    Args:
        text: Texte à nettoyer
        keep_common_symbols: Garder les symboles courants (©, ®, ™, €, £, etc.)
    """
    # Liste des symboles courants à préserver
    common_symbols = set('©®™€£¥°±×÷≈≠≤≥•◦▪▫→←↑↓⇒⇐⇑⇓')

    cleaned_chars = []
    for char in text:
        # Garder les caractères ASCII standard
        if ord(char) < 128:
            cleaned_chars.append(char)
            continue

        # Garder les symboles courants si demandé
        if keep_common_symbols and char in common_symbols:
            cleaned_chars.append(char)
            continue

        # Catégories Unicode à garder
        category = unicodedata.category(char)

        # Garder : lettres, nombres, ponctuation, espaces
        if category.startswith(('L', 'N', 'P', 'Z', 'S')):
            # Exclure les emojis et symboles graphiques
            if category in ('So', 'Cn'):  # Autres symboles, non assignés
                # Vérifier si c'est un emoji ou icône
                if ord(char) > 0x2000:  # Au-delà des symboles de base
                    cleaned_chars.append('[?]')  # Remplacer par placeholder
                    continue

            cleaned_chars.append(char)
        else:
            # Remplacer par espace pour les autres
            cleaned_chars.append(' ')

    text = ''.join(cleaned_chars)

    # Nettoyer les placeholders consécutifs
    text = re.sub(r'\[\?\]\s*\[\?\]', '[?]', text)

    # Nettoyer les placeholders isolés sur une ligne
    text = re.sub(r'^\s*\[\?\]\s*$', '', text, flags=re.MULTILINE)

    return text


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


def normalize_whitespace(text: str, max_empty_lines: int = 1) -> str:
    """
    Normalise les espaces : supprime espaces multiples, lignes vides excessives.

    Args:
        text: Texte à normaliser
        max_empty_lines: Nombre maximum de lignes vides consécutives (défaut: 1)
    """
    # Supprimer espaces en début/fin de chaque ligne
    lines = [line.rstrip() for line in text.split('\n')]

    # Réduire espaces multiples en un seul
    lines = [re.sub(r' +', ' ', line) for line in lines]

    # Réduire lignes vides multiples
    cleaned_lines = []
    empty_count = 0

    for line in lines:
        if not line.strip():
            empty_count += 1
            if empty_count <= max_empty_lines:
                cleaned_lines.append(line)
        else:
            empty_count = 0
            cleaned_lines.append(line)

    # Supprimer les lignes vides au début et à la fin
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines)


def remove_emails(text: str) -> str:
    """
    Supprime toutes les adresses email du texte.
    """
    # Pattern pour détecter les emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '', text)
    return text


def remove_urls(text: str) -> str:
    """
    Supprime toutes les URLs du texte (http, https, www).
    """
    # Pattern pour URLs complètes
    url_patterns = [
        r'https?://[^\s<>"{}|\\^`\[\]]+',  # http:// ou https://
        r'www\.[^\s<>"{}|\\^`\[\]]+',      # www.
        r'ftp://[^\s<>"{}|\\^`\[\]]+',     # ftp://
    ]

    for pattern in url_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text


def remove_copyright_notices(text: str) -> str:
    """
    Supprime les phrases contenant des mentions de droits réservés, copyright, etc.
    """
    lines = text.split('\n')
    cleaned_lines = []

    # Patterns pour détecter les mentions de droits
    copyright_patterns = [
        r'.*(?:tous\s+)?droits?\s+(?:d[\'e]\s*)?(?:reproduction\s+)?(?:strictement\s+)?r[ée]serv[ée]s?.*',
        r'.*copyright.*',
        r'.*©.*(?:20\d{2}|19\d{2}).*',  # © avec année
        r'.*\(c\).*(?:20\d{2}|19\d{2}).*',  # (c) avec année
        r'.*propriété\s+intellectuelle.*',
        r'.*all\s+rights\s+reserved.*',
        r'.*reproduction\s+interdite.*',
        r'.*usage\s+(?:strictement\s+)?(?:privé|personnel).*',
        r'.*ne\s+pas\s+(?:reproduire|diffuser|distribuer).*',
        r'.*confidential.*',
        r'.*document\s+protégé.*',
    ]

    for line in lines:
        stripped = line.strip()
        is_copyright = False

        for pattern in copyright_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                is_copyright = True
                break

        if not is_copyright:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_metadata_and_headers(text: str) -> str:
    """
    Supprime les métadonnées et en-têtes de documents (date de publication, auteurs, références, etc.)
    """
    lines = text.split('\n')
    cleaned_lines = []

    # Patterns pour détecter les métadonnées de document
    metadata_patterns = [
        r'^Date\s+de\s+publication\s*:.*$',
        r'^Mots-clés.*$',
        r'^Keywords.*$',
        r'^Pour\s+toute\s+question\s*:.*$',
        r'^Par\s+(?:mail|téléphone|email)\s*:.*$',
        r'^Service\s+Relation\s+clientèle.*$',
        r'^Immeuble\s+.*$',
        r'^Réf\.\s*:.*$',
        r'^Cet\s+article\s+est\s+issu\s+de\s*:.*$',
        r'^par\s+[A-Z][a-z]+.*[A-Z][A-Z]+.*$',  # par Nom PRENOM
        r'^Document\s+téléchargé\s+le\s*:.*$',
        r'^Pour\s+le\s+compte\s*:.*$',
        r'^Techniques\s+de\s+l\'Ingénieur.*$',
        r'^Résumé\s+Cet\s+article.*$',
        r'^Abstract\s+This\s+article.*$',
        r'^\d{2}/\d{2}/\d{4}$',  # Dates seules
        r'^Photo\s*:.*$',
        r'^ARCHIVE$',
        r'^INSTALLATIONS$',
        r'^En\s+poursuivant\s+votre\s+navigation.*$',
        r'^.*cookies.*statistiques.*$',
        r'^Le\s+magazine.*$',
        r'^Feuilleter.*$',
        r'^Voir\s+le\s+sommaire.*$',
        r'^Suivez-nous$',
        r'^S\'inscrire\s+aux\s+newsletters.*$',
        r'^Mon\s+compte.*$',
        r'^Connexion.*$',
        r'^Accès\s+annonceur.*$',
        r'^PLUS\s+DE\s+PHOTOS$',
        r'^Consulter\s+l\'intégralité.*$',
        r'^.*larpf\.fr/.*$',  # URLs de site web
        r'^\d+/\d+$',  # Numéros de page web (1/3, 2/3)
    ]

    for line in lines:
        stripped = line.strip()
        is_metadata = False

        for pattern in metadata_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_metadata = True
                break

        if not is_metadata:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_short_isolated_lines(text: str, min_length: int = 15) -> str:
    """
    Supprime les lignes isolées très courtes qui sont souvent des artefacts.
    Une ligne est considérée isolée si elle est entourée de lignes vides.
    """
    lines = text.split('\n')
    cleaned_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Vérifier si c'est une ligne courte
        if len(stripped) < min_length:
            # Vérifier si elle est isolée (entourée de lignes vides)
            prev_empty = (i == 0 or not lines[i-1].strip())
            next_empty = (i == len(lines)-1 or not lines[i+1].strip())

            # Si isolée et courte, la supprimer (sauf si c'est un titre markdown)
            if prev_empty and next_empty and not stripped.startswith('#'):
                continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def detect_and_remove_repetitive_content(text: str, min_occurrences: int = 3, similarity_threshold: float = 0.9) -> str:
    """
    Détecte et supprime le contenu répétitif (headers, footers, watermarks).

    Args:
        text: Texte à analyser
        min_occurrences: Nombre minimum d'occurrences pour considérer comme répétitif
        similarity_threshold: Seuil de similarité (0-1) pour considérer deux lignes comme identiques

    Returns:
        Texte sans contenu répétitif
    """
    lines = text.split('\n')

    if len(lines) < 10:  # Pas assez de lignes pour détecter des patterns
        return text

    # Compter les occurrences de chaque ligne (en ignorant les lignes vides et très courtes)
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 5:  # Ignorer lignes trop courtes
            continue

        # Normaliser pour la comparaison (minuscules, sans espaces multiples)
        normalized = re.sub(r'\s+', ' ', stripped.lower())

        if normalized not in line_counts:
            line_counts[normalized] = {'original': stripped, 'count': 0}
        line_counts[normalized]['count'] += 1

    # Identifier les lignes répétitives
    repetitive_patterns = set()
    for normalized, data in line_counts.items():
        if data['count'] >= min_occurrences:
            repetitive_patterns.add(normalized)

    # Si trop de lignes sont répétitives, c'est probablement du vrai contenu
    if len(repetitive_patterns) > len(lines) * 0.3:
        return text  # Ne pas filtrer

    # Filtrer les lignes répétitives
    cleaned_lines = []
    removed_count = 0

    for line in lines:
        stripped = line.strip()

        # Garder les lignes vides et courtes
        if len(stripped) < 5:
            cleaned_lines.append(line)
            continue

        # Vérifier si la ligne est répétitive
        normalized = re.sub(r'\s+', ' ', stripped.lower())

        if normalized in repetitive_patterns:
            removed_count += 1
            # Garder la première occurrence de chaque pattern répétitif
            if removed_count == 1 or normalized not in {re.sub(r'\s+', ' ', l.strip().lower()) for l in cleaned_lines[-min_occurrences:]}:
                # Ne pas la garder du tout si c'est vraiment répétitif
                continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def fix_broken_headings(text: str) -> str:
    """
    Répare les titres coupés/cassés sur plusieurs lignes.
    """
    lines = text.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Si c'est un titre markdown
        if stripped.startswith('#'):
            heading_prefix = ''
            heading_text = stripped

            # Extraire le préfixe (# ## ### etc.)
            while heading_text and heading_text[0] == '#':
                heading_prefix += '#'
                heading_text = heading_text[1:].lstrip()

            # Vérifier si le titre est coupé (pas de point final, ligne courte)
            if (i + 1 < len(lines) and
                heading_text and
                not heading_text.endswith('.') and
                not heading_text.endswith('?') and
                not heading_text.endswith('!') and
                len(heading_text) < 100):

                next_line = lines[i + 1].strip()

                # Si la ligne suivante n'est pas vide et n'est pas un titre
                if (next_line and
                    not next_line.startswith('#') and
                    len(next_line) < 100 and
                    next_line[0].islower()):  # Continue avec minuscule

                    # Fusionner les deux lignes
                    merged_heading = f"{heading_prefix} {heading_text} {next_line}"
                    fixed_lines.append(merged_heading)
                    i += 2
                    continue

            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

        i += 1

    return '\n'.join(fixed_lines)


def normalize_headings(text: str) -> str:
    """
    Normalise les titres : supprime espaces multiples, capitalisation cohérente.
    """
    lines = text.split('\n')
    normalized_lines = []

    for line in lines:
        stripped = line.strip()

        # Si c'est un titre markdown
        if stripped.startswith('#'):
            heading_prefix = ''
            heading_text = stripped

            # Extraire le préfixe
            while heading_text and heading_text[0] == '#':
                heading_prefix += '#'
                heading_text = heading_text[1:].lstrip()

            # Nettoyer le texte du titre
            heading_text = re.sub(r'\s+', ' ', heading_text)  # Espaces multiples
            heading_text = heading_text.strip()

            # Reconstruire le titre
            if heading_text:
                normalized_lines.append(f"{heading_prefix} {heading_text}")
            else:
                continue  # Skip les titres vides
        else:
            normalized_lines.append(line)

    return '\n'.join(normalized_lines)


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


def clean_text(text: str, is_ocr: bool = False, remove_repetitive: bool = True, clean_emojis: bool = True) -> str:
    """
    Applique tous les nettoyages sur le texte.

    Args:
        text: Texte à nettoyer
        is_ocr: True si le texte provient d'OCR (nettoyage plus agressif)
        remove_repetitive: Supprimer le contenu répétitif (headers/footers)
        clean_emojis: Nettoyer les emojis et caractères inconnus

    Returns:
        Texte nettoyé
    """
    # Nettoyages de base
    text = remove_page_numbers(text)
    text = remove_figure_references(text)
    text = remove_update_dates(text)
    text = remove_table_of_contents(text)

    # Supprimer emails, URLs et mentions de droits
    text = remove_emails(text)
    text = remove_urls(text)
    text = remove_copyright_notices(text)
    text = remove_metadata_and_headers(text)

    # Nettoyer les caractères inconnus et emojis
    if clean_emojis:
        text = remove_unknown_characters(text, keep_common_symbols=True)

    # Supprimer le contenu répétitif (headers/footers)
    if remove_repetitive:
        text = detect_and_remove_repetitive_content(text, min_occurrences=3)

    text = normalize_whitespace(text)

    # Améliorer la structure des titres
    text = fix_broken_headings(text)
    text = normalize_headings(text)
    text = improve_markdown_structure(text)

    # Nettoyages spécifiques OCR
    if is_ocr:
        text = remove_ocr_artifacts(text)

    # Supprimer les lignes isolées trop courtes
    text = remove_short_isolated_lines(text, min_length=15)

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
