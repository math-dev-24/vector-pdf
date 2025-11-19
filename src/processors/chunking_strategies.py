"""
Module de stratégies de chunking adaptatives selon le type de contenu.
Permet un chunking intelligent pour tableaux, listes, code, texte narratif.
"""

import re
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """Configuration pour une stratégie de chunking."""
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    strategy_name: str


class ContentTypeDetector:
    """Détecte le type de contenu pour adapter le chunking."""

    @staticmethod
    def detect_content_type(text: str) -> str:
        """
        Détecte le type dominant de contenu.

        Args:
            text: Texte à analyser

        Returns:
            Type de contenu ("table", "list", "code", "narrative", "mixed")
        """
        # Compter les indicateurs de chaque type
        table_score = ContentTypeDetector._score_table(text)
        list_score = ContentTypeDetector._score_list(text)
        code_score = ContentTypeDetector._score_code(text)

        scores = {
            'table': table_score,
            'list': list_score,
            'code': code_score
        }

        max_score = max(scores.values())

        if max_score < 0.2:  # Seuil minimum
            return 'narrative'

        if table_score >= 0.3:
            return 'table'
        elif list_score >= 0.3:
            return 'list'
        elif code_score >= 0.3:
            return 'code'
        elif max_score < 0.5:
            return 'mixed'
        else:
            return max(scores, key=scores.get)

    @staticmethod
    def _score_table(text: str) -> float:
        """Score pour détecter un tableau."""
        lines = text.split('\n')
        if not lines:
            return 0.0

        # Markdown tables (|)
        table_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)

        # Tab-separated (TSV)
        tab_lines = sum(1 for line in lines if '\t' in line and line.count('\t') >= 2)

        # Alignement (plusieurs lignes avec même structure)
        # ...

        total_lines = len([l for l in lines if l.strip()])
        if total_lines == 0:
            return 0.0

        return max(table_lines, tab_lines) / total_lines

    @staticmethod
    def _score_list(text: str) -> float:
        """Score pour détecter une liste."""
        lines = text.split('\n')
        non_empty = [l for l in lines if l.strip()]

        if not non_empty:
            return 0.0

        # Listes markdown (-, *, +)
        markdown_list = sum(1 for line in non_empty if re.match(r'^\s*[-*+]\s+', line))

        # Listes numérotées
        numbered_list = sum(1 for line in non_empty if re.match(r'^\s*\d+\.\s+', line))

        return max(markdown_list, numbered_list) / len(non_empty)

    @staticmethod
    def _score_code(text: str) -> float:
        """Score pour détecter du code."""
        # Blocs de code markdown
        if '```' in text:
            return 0.9

        # Inline code fréquent
        code_blocks = len(re.findall(r'`[^`]+`', text))
        if code_blocks > 3:
            return 0.6

        # Indentation systématique
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))

        if lines and indented_lines / len(lines) > 0.5:
            return 0.7

        return 0.0


class AdaptiveChunker:
    """Chunker adaptatif selon le type de contenu."""

    def __init__(self):
        self.configs = {
            'narrative': ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
                strategy_name="narrative"
            ),
            'table': ChunkingConfig(
                chunk_size=2000,  # Plus grand pour ne pas couper les tableaux
                chunk_overlap=0,  # Pas d'overlap pour tableaux
                separators=["\n\n", "\n"],  # Couper entre tables
                strategy_name="table"
            ),
            'list': ChunkingConfig(
                chunk_size=1500,
                chunk_overlap=100,
                separators=["\n\n", "\n- ", "\n* ", "\n+ ", "\n"],
                strategy_name="list"
            ),
            'code': ChunkingConfig(
                chunk_size=1500,
                chunk_overlap=100,
                separators=["\n\n", "\n```", "\n", ""],
                strategy_name="code"
            ),
            'mixed': ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
                strategy_name="mixed"
            )
        }

    def chunk_text(self, text: str, content_type: str = None) -> List[str]:
        """
        Découpe le texte selon son type.

        Args:
            text: Texte à chunker
            content_type: Type de contenu (auto-détecté si None)

        Returns:
            Liste de chunks
        """
        # Auto-détecter le type si non fourni
        if content_type is None:
            content_type = ContentTypeDetector.detect_content_type(text)

        # Récupérer la config
        config = self.configs.get(content_type, self.configs['narrative'])

        # Créer le splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators,
            is_separator_regex=False
        )

        chunks = splitter.split_text(text)

        return chunks


class SemanticChunker:
    """Chunking sémantique basé sur la cohérence thématique."""

    def __init__(self, min_chunk_size: int = 300, max_chunk_size: int = 1500):
        """
        Initialise le chunker sémantique.

        Args:
            min_chunk_size: Taille minimale d'un chunk
            max_chunk_size: Taille maximale d'un chunk
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Découpe le texte par sections logiques.

        Args:
            text: Texte à chunker

        Returns:
            Liste de (section_title, content)
        """
        lines = text.split('\n')
        chunks = []
        current_section = None
        current_content = []

        for line in lines:
            # Détecter un titre
            if line.strip().startswith('#'):
                # Sauvegarder la section précédente
                if current_section and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        chunks.append((current_section, content))

                # Nouvelle section
                current_section = line.strip()
                current_content = [line]
            else:
                current_content.append(line)

        # Dernière section
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                chunks.append((current_section, content))

        # Fusionner les sections trop petites
        merged_chunks = self._merge_small_chunks(chunks)

        return merged_chunks

    def _merge_small_chunks(
        self,
        chunks: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Fusionne les chunks trop petits avec leurs voisins.

        Args:
            chunks: Liste de (titre, contenu)

        Returns:
            Chunks fusionnés
        """
        if not chunks:
            return []

        merged = []
        i = 0

        while i < len(chunks):
            title, content = chunks[i]
            content_len = len(content)

            # Si trop petit, fusionner avec le suivant
            if content_len < self.min_chunk_size and i + 1 < len(chunks):
                next_title, next_content = chunks[i + 1]
                merged_content = content + '\n\n' + next_content
                merged_title = f"{title} / {next_title}"
                merged.append((merged_title, merged_content))
                i += 2
            else:
                # Si trop grand, découper
                if content_len > self.max_chunk_size:
                    sub_chunks = self._split_large_chunk(content, title)
                    merged.extend(sub_chunks)
                else:
                    merged.append((title, content))
                i += 1

        return merged

    def _split_large_chunk(
        self,
        content: str,
        base_title: str
    ) -> List[Tuple[str, str]]:
        """
        Découpe un chunk trop grand.

        Args:
            content: Contenu à découper
            base_title: Titre de base

        Returns:
            Liste de chunks
        """
        # Utiliser le chunker récursif standard
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        sub_texts = splitter.split_text(content)

        return [(f"{base_title} (partie {i+1})", text) for i, text in enumerate(sub_texts)]


class SentenceWindowChunker:
    """
    Chunking par fenêtre de phrases.
    Stocke des phrases individuelles mais avec contexte autour.
    """

    def __init__(self, window_size: int = 3):
        """
        Initialise le chunker par fenêtre.

        Args:
            window_size: Nombre de phrases avant/après pour contexte
        """
        self.window_size = window_size

    def chunk_with_windows(self, text: str) -> List[Dict]:
        """
        Découpe en phrases avec fenêtre de contexte.

        Args:
            text: Texte à chunker

        Returns:
            Liste de dicts avec 'sentence', 'context', 'window'
        """
        # Découper en phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []

        for i, sentence in enumerate(sentences):
            # Fenêtre de contexte
            start = max(0, i - self.window_size)
            end = min(len(sentences), i + self.window_size + 1)

            context_window = sentences[start:end]
            context = ' '.join(context_window)

            chunks.append({
                'sentence': sentence,
                'context': context,
                'window_indices': (start, end),
                'sentence_index': i
            })

        return chunks


if __name__ == "__main__":
    # Test des différentes stratégies

    # 1. Test détection de type
    print("=" * 60)
    print("TEST DÉTECTION DE TYPE")
    print("=" * 60)

    test_texts = {
        'narrative': "Ceci est un texte narratif normal. Il contient plusieurs phrases. Elles forment un paragraphe cohérent.",
        'table': """
| Nom | Âge | Ville |
|-----|-----|-------|
| Alice | 30 | Paris |
| Bob | 25 | Lyon |
""",
        'list': """
- Premier élément
- Deuxième élément
- Troisième élément
    - Sous-élément A
    - Sous-élément B
""",
        'code': """
```python
def hello():
    print("Hello World")
    return True
```
"""
    }

    detector = ContentTypeDetector()
    for expected_type, text in test_texts.items():
        detected = detector.detect_content_type(text)
        print(f"{expected_type.upper()}: détecté comme '{detected}' ✅" if detected == expected_type else f"❌")

    # 2. Test chunking adaptatif
    print("\n" + "=" * 60)
    print("TEST CHUNKING ADAPTATIF")
    print("=" * 60)

    chunker = AdaptiveChunker()
    text = "Ceci est un long texte. " * 100

    chunks = chunker.chunk_text(text)
    print(f"Chunks créés: {len(chunks)}")
    print(f"Taille moyenne: {sum(len(c) for c in chunks) / len(chunks):.0f} caractères")

    # 3. Test semantic chunking
    print("\n" + "=" * 60)
    print("TEST SEMANTIC CHUNKING")
    print("=" * 60)

    doc = """
# Introduction

Texte d'introduction.

## Section 1

Contenu de la section 1.

### Sous-section 1.1

Détails de la sous-section.

## Section 2

Contenu de la section 2.
"""

    semantic_chunker = SemanticChunker()
    semantic_chunks = semantic_chunker.chunk_by_sections(doc)

    for title, content in semantic_chunks:
        print(f"\n{title}")
        print(f"  Longueur: {len(content)} caractères")