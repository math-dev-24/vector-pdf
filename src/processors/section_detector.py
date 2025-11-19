"""
Module pour détecter la structure hiérarchique des documents (sections, sous-sections).
Permet d'ajouter du contexte sémantique aux chunks.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Section:
    """Représente une section du document."""
    level: int  # 1, 2, 3 pour #, ##, ###
    title: str
    start_line: int
    end_line: int
    content: str
    parent: Optional['Section'] = None
    children: List['Section'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def get_hierarchy_path(self) -> List[str]:
        """Retourne le chemin hiérarchique complet."""
        path = []
        current = self
        while current:
            path.insert(0, current.title)
            current = current.parent
        return path

    def get_hierarchy_string(self, separator: str = " > ") -> str:
        """Retourne la hiérarchie sous forme de string."""
        return separator.join(self.get_hierarchy_path())


class SectionDetector:
    """Détecte la structure hiérarchique d'un document markdown."""

    def __init__(self):
        self.sections: List[Section] = []

    def detect_heading(self, line: str) -> Optional[Tuple[int, str]]:
        """
        Détecte si une ligne est un titre et retourne son niveau.

        Args:
            line: Ligne à analyser

        Returns:
            (niveau, titre) ou None
        """
        stripped = line.strip()

        # Titres markdown (# ## ###)
        match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            return (level, title)

        # Titres en MAJUSCULES (considérés comme niveau 3)
        if stripped.isupper() and 5 < len(stripped) < 100:
            # Vérifier que c'est bien du texte, pas des artefacts
            if sum(c.isalpha() for c in stripped) / len(stripped) > 0.6:
                return (3, stripped)

        return None

    def parse_document(self, text: str) -> List[Section]:
        """
        Parse un document et détecte toutes les sections.

        Args:
            text: Texte markdown complet

        Returns:
            Liste de sections détectées
        """
        lines = text.split('\n')
        self.sections = []
        current_sections = {0: None}  # {niveau: Section}

        i = 0
        while i < len(lines):
            line = lines[i]
            heading_info = self.detect_heading(line)

            if heading_info:
                level, title = heading_info

                # Trouver le parent (section de niveau inférieur)
                parent = None
                for parent_level in range(level - 1, 0, -1):
                    if parent_level in current_sections:
                        parent = current_sections[parent_level]
                        break

                # Créer la section
                section = Section(
                    level=level,
                    title=title,
                    start_line=i,
                    end_line=i,  # Sera mis à jour
                    content='',
                    parent=parent
                )

                # Ajouter aux enfants du parent
                if parent:
                    parent.children.append(section)

                # Mettre à jour current_sections
                current_sections[level] = section

                # Supprimer les niveaux plus profonds
                for l in list(current_sections.keys()):
                    if l > level:
                        del current_sections[l]

                self.sections.append(section)

            i += 1

        # Calculer le contenu de chaque section
        for i, section in enumerate(self.sections):
            if i + 1 < len(self.sections):
                section.end_line = self.sections[i + 1].start_line - 1
            else:
                section.end_line = len(lines) - 1

            section.content = '\n'.join(lines[section.start_line:section.end_line + 1])

        return self.sections

    def get_section_at_line(self, line_num: int) -> Optional[Section]:
        """
        Retourne la section active à une ligne donnée.

        Args:
            line_num: Numéro de ligne

        Returns:
            Section ou None
        """
        for section in reversed(self.sections):
            if section.start_line <= line_num <= section.end_line:
                return section
        return None

    def get_section_for_text_position(self, text: str, position: int) -> Optional[Section]:
        """
        Retourne la section correspondant à une position dans le texte.

        Args:
            text: Texte complet du document
            position: Position du caractère dans le texte

        Returns:
            Section ou None
        """
        # Calculer le numéro de ligne
        text_before = text[:position]
        line_num = text_before.count('\n')

        return self.get_section_at_line(line_num)

    def get_deepest_section_at_line(self, line_num: int) -> Optional[Section]:
        """
        Retourne la section la plus profonde (plus spécifique) à une ligne donnée.

        Args:
            line_num: Numéro de ligne

        Returns:
            Section la plus profonde ou None
        """
        matching_sections = []

        for section in self.sections:
            if section.start_line <= line_num <= section.end_line:
                matching_sections.append(section)

        if not matching_sections:
            return None

        # Retourner la section avec le niveau le plus élevé (plus profond)
        return max(matching_sections, key=lambda s: s.level)

    def add_section_context_to_chunks(
        self,
        chunks: List[Dict],
        document_text: str
    ) -> List[Dict]:
        """
        Ajoute le contexte de section à chaque chunk.

        Args:
            chunks: Liste de chunks
            document_text: Texte complet du document

        Returns:
            Chunks avec métadonnées de section enrichies
        """
        # Parser le document
        self.parse_document(document_text)

        enriched_chunks = []
        char_position = 0

        for chunk in chunks:
            content = chunk['content']
            metadata = chunk['metadata'].copy()

            # Trouver la section correspondante
            section = self.get_section_for_text_position(document_text, char_position)

            if section:
                metadata['section_title'] = section.title
                metadata['section_level'] = section.level
                metadata['section_hierarchy'] = section.get_hierarchy_path()
                metadata['section_hierarchy_string'] = section.get_hierarchy_string()

                # Ajouter titres des parents directs
                if section.parent:
                    metadata['parent_section'] = section.parent.title
            else:
                metadata['section_title'] = None
                metadata['section_level'] = None
                metadata['section_hierarchy'] = []
                metadata['section_hierarchy_string'] = ''

            enriched_chunks.append({
                'content': content,
                'metadata': metadata
            })

            char_position += len(content)

        return enriched_chunks

    def get_document_outline(self) -> str:
        """
        Génère un plan du document.

        Returns:
            Plan formaté en string
        """
        outline = []

        for section in self.sections:
            indent = '  ' * (section.level - 1)
            outline.append(f"{indent}{section.level}. {section.title}")

        return '\n'.join(outline)

    def print_structure(self):
        """Affiche la structure du document."""
        print("\n" + "=" * 60)
        print("📑 STRUCTURE DU DOCUMENT")
        print("=" * 60)

        if not self.sections:
            print("Aucune section détectée.")
            return

        print(f"\nNombre de sections: {len(self.sections)}\n")
        print(self.get_document_outline())
        print("\n" + "=" * 60)


def add_section_metadata(
    chunking_results: List[Dict],
    verbose: bool = True
) -> List[Dict]:
    """
    Ajoute les métadonnées de section à tous les chunks.

    Args:
        chunking_results: Résultats du chunking
        verbose: Afficher progression

    Returns:
        Résultats avec métadonnées de section
    """
    if verbose:
        print("\n📑 Détection de la structure des documents...")

    detector = SectionDetector()
    enriched_results = []

    for result in chunking_results:
        if verbose:
            print(f"\n  📄 {result['file_name']}")

        # Lire le fichier pour avoir le texte complet
        try:
            with open(result['file_path'], 'r', encoding='utf-8') as f:
                full_text = f.read()
        except Exception as e:
            print(f"    ⚠️  Impossible de lire le fichier: {e}")
            enriched_results.append(result)
            continue

        # Ajouter contexte de section
        enriched_chunks = detector.add_section_context_to_chunks(
            chunks=result['chunks'],
            document_text=full_text
        )

        if verbose:
            sections_found = len(detector.sections)
            print(f"    ✅ {sections_found} section(s) détectée(s)")

        enriched_results.append({
            'file_path': result['file_path'],
            'file_name': result['file_name'],
            'num_chunks': result['num_chunks'],
            'total_chars': result['total_chars'],
            'chunks': enriched_chunks,
            'sections': detector.sections  # Ajouter la structure pour référence
        })

    if verbose:
        total_sections = sum(len(r.get('sections', [])) for r in enriched_results)
        print(f"\n✅ Structure détectée: {total_sections} sections au total")

    return enriched_results


if __name__ == "__main__":
    # Test
    test_document = """
# Introduction

Ceci est l'introduction du document.

## Contexte

Le contexte général du sujet.

### Historique

L'historique détaillé.

## Objectifs

Les objectifs principaux.

# Méthodologie

Description de la méthodologie utilisée.

## Collecte de données

Comment les données ont été collectées.

## Analyse

Méthodes d'analyse employées.

# Résultats

Les résultats obtenus.

## Résultats quantitatifs

Données chiffrées.

## Résultats qualitatifs

Observations qualitatives.

# Conclusion

Conclusions et perspectives.
"""

    detector = SectionDetector()
    sections = detector.parse_document(test_document)

    detector.print_structure()

    # Tester get_section_at_line
    section = detector.get_section_at_line(15)
    if section:
        print(f"\nSection à la ligne 15: {section.get_hierarchy_string()}")