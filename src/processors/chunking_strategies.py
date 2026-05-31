"""
Module de stratégies de chunking adaptatives selon le type de contenu.
Permet un chunking intelligent pour tableaux, listes, code, texte narratif.
"""

import re
from typing import List, Dict, Tuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass

from src.processors.section_detector import SectionDetector


@dataclass
class ChunkingConfig:
    """Configuration pour une stratégie de chunking."""
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    strategy_name: str


def is_valid_section_title(title: str) -> bool:
    """
    Filtre les faux titres issus des PDFs techniques (formules, labels courts).
    """
    clean = re.sub(r"\*+", "", title).strip()
    if len(clean) < 4:
        return False
    if re.match(r"^[A-Z0-9]{1,3}$", clean):
        return False
    if re.match(r"^\d+$", clean):
        return False
    if re.match(r"^[WλΦΔθ]+$", clean):
        return False
    alpha_ratio = sum(c.isalpha() for c in clean) / len(clean)
    return alpha_ratio >= 0.4


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
        table_score = ContentTypeDetector._score_table(text)
        list_score = ContentTypeDetector._score_list(text)
        code_score = ContentTypeDetector._score_code(text)

        scores = {
            'table': table_score,
            'list': list_score,
            'code': code_score
        }

        max_score = max(scores.values())

        if max_score < 0.2:
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

        table_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        tab_lines = sum(1 for line in lines if '\t' in line and line.count('\t') >= 2)

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

        markdown_list = sum(1 for line in non_empty if re.match(r'^\s*[-*+]\s+', line))
        numbered_list = sum(1 for line in non_empty if re.match(r'^\s*\d+\.\s+', line))

        return max(markdown_list, numbered_list) / len(non_empty)

    @staticmethod
    def _score_code(text: str) -> float:
        """Score pour détecter du code."""
        if '```' in text:
            return 0.9

        code_blocks = len(re.findall(r'`[^`]+`', text))
        if code_blocks > 3:
            return 0.6

        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))

        if lines and indented_lines / len(lines) > 0.5:
            return 0.7

        return 0.0


class AdaptiveChunker:
    """Chunker adaptatif selon le type de contenu."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap
        self.configs = {
            'narrative': ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
                strategy_name="narrative"
            ),
            'table': ChunkingConfig(
                chunk_size=max(chunk_size * 2, 2000),
                chunk_overlap=0,
                separators=["\n\n", "\n"],
                strategy_name="table"
            ),
            'list': ChunkingConfig(
                chunk_size=int(chunk_size * 1.5),
                chunk_overlap=100,
                separators=["\n\n", "\n- ", "\n* ", "\n+ ", "\n"],
                strategy_name="list"
            ),
            'code': ChunkingConfig(
                chunk_size=int(chunk_size * 1.5),
                chunk_overlap=100,
                separators=["\n\n", "\n```", "\n", ""],
                strategy_name="code"
            ),
            'mixed': ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
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
        if content_type is None:
            content_type = ContentTypeDetector.detect_content_type(text)

        config = self.configs.get(content_type, self.configs['narrative'])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators,
            is_separator_regex=False
        )

        return splitter.split_text(text)


class SemanticChunker:
    """Chunking par sections markdown avec contrôle de taille."""

    def __init__(
        self,
        min_chunk_size: int = 400,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 200,
        target_level: Optional[int] = None,
    ):
        """
        Args:
            min_chunk_size: Fusionner les sections plus petites
            max_chunk_size: Découper les sections plus grandes
            chunk_overlap: Overlap lors du découpage des grosses sections
            target_level: Niveau markdown (#=1). None = auto-détection.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.target_level = target_level
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n1.", "\n2.", "\n3.", "\n4.", "\n5.", "\n6.", "\n7.", "\n8.", "\n9.",
                "\n#### ", "\n### ", "\n## ", "\n# ",
                "\n\n", "\n", ". ", " ",
            ],
        )

    def _detect_heading_level(self, text: str) -> int:
        """Choisit le niveau de titre le plus pertinent pour ce document."""
        detector = SectionDetector()
        sections = detector.parse_document(text)
        level_counts: Dict[int, int] = {}

        for section in sections:
            if is_valid_section_title(section.title):
                level_counts[section.level] = level_counts.get(section.level, 0) + 1

        if not level_counts:
            return 3

        # Privilégier les niveaux avec plusieurs titres valides (souvent ## ou #####)
        return max(level_counts, key=lambda lvl: (level_counts[lvl], -abs(lvl - 3)))

    def chunk_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Découpe par sections logiques avec bornes min/max.

        Returns:
            Liste de (section_title, content)
        """
        level = self.target_level or self._detect_heading_level(text)
        raw_chunks = self._split_by_heading_level(text, level)

        if not raw_chunks:
            return self._split_large_chunk(text, "Document")

        processed: List[Tuple[str, str]] = []
        for title, content in raw_chunks:
            content = content.strip()
            if len(content) < 30:
                continue
            if len(content) > self.max_chunk_size:
                processed.extend(self._split_large_chunk(content, title))
            else:
                processed.append((title, content))

        if not processed:
            return self._split_large_chunk(text, "Document")

        return self._merge_small_chunks(processed)

    def _split_by_heading_level(
        self,
        text: str,
        level: int,
    ) -> List[Tuple[str, str]]:
        """Découpe au niveau de titre choisi (ex. ##### Chapitre)."""
        prefix = "#" * level + " "
        lines = text.split("\n")
        chunks: List[Tuple[str, str]] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []
        preamble: List[str] = []

        def flush() -> None:
            nonlocal current_title, current_lines
            if current_title and current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append((current_title, content))
            current_title = None
            current_lines = []

        for line in lines:
            if line.startswith(prefix):
                title_text = line[len(prefix):].strip()
                if is_valid_section_title(title_text):
                    flush()
                    current_title = line.strip()
                    current_lines = [line]
                    continue
            if current_title:
                current_lines.append(line)
            else:
                preamble.append(line)

        flush()

        if preamble:
            preamble_text = "\n".join(preamble).strip()
            if len(preamble_text) >= 30:
                chunks.insert(0, ("Page de garde", preamble_text))

        if len(chunks) < 3 and level < 6:
            return self._split_by_heading_level(text, level + 1)

        return chunks

    def _merge_small_chunks(
        self,
        chunks: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Fusionne les sections trop petites sans créer de chunks géants."""
        if not chunks:
            return []

        merged: List[Tuple[str, str]] = []
        i = 0

        while i < len(chunks):
            title, content = chunks[i]
            content_len = len(content)

            if content_len < self.min_chunk_size and i + 1 < len(chunks):
                next_title, next_content = chunks[i + 1]
                combined_len = content_len + len(next_content) + 2

                if combined_len <= self.max_chunk_size:
                    merged.append((f"{title} / {next_title}", content + "\n\n" + next_content))
                    i += 2
                    continue

            if content_len > self.max_chunk_size:
                merged.extend(self._split_large_chunk(content, title))
            else:
                merged.append((title, content))
            i += 1

        return merged

    def _split_large_chunk(
        self,
        content: str,
        base_title: str
    ) -> List[Tuple[str, str]]:
        """Découpe un chunk trop grand en respectant max_chunk_size."""
        sub_texts = self._splitter.split_text(content)
        if len(sub_texts) == 1:
            return [(base_title, content)]

        return [
            (f"{base_title} (partie {i + 1}/{len(sub_texts)})", text)
            for i, text in enumerate(sub_texts)
        ]


class SentenceWindowChunker:
    """
    Chunking par fenêtre de phrases.
    Stocke des phrases individuelles mais avec contexte autour.
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def chunk_with_windows(self, text: str) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []

        for i, sentence in enumerate(sentences):
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
