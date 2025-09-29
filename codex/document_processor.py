"""Utilities for processing large documents via chunking.

This module provides a light-weight implementation that can read large text
content, split it into overlapping chunks, and expose metadata about the split
segments.  The behaviour is intentionally simple: chunk sizes are measured in
characters to avoid external dependencies on tokenizers while still providing
predictable behaviour that works well for long documents.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List
import re

__all__ = ["DocumentChunk", "DocumentProcessor", "chunk_text"]


_TOKEN_RE = re.compile(r"\S+\s*")


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of a larger document.

    Attributes
    ----------
    text:
        The chunked text segment.
    index:
        Zero-based index of the chunk.
    start:
        The starting character offset within the original document.
    end:
        The exclusive ending character offset within the original document.
    """

    text: str
    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the length of the chunk in characters."""

        return len(self.text)


def _tokenize(text: str) -> List[str]:
    """Tokenise *text* into a list of tokens preserving whitespace."""

    return [match.group(0) for match in _TOKEN_RE.finditer(text)]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1_000,
    overlap: int = 200,
) -> List[DocumentChunk]:
    """Split *text* into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if chunk_size <= overlap and overlap:
        raise ValueError("chunk_size must be larger than overlap")

    if not text:
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []

    token_lengths = [len(token) for token in tokens]
    prefix_lengths = [0]
    for length in token_lengths:
        prefix_lengths.append(prefix_lengths[-1] + length)

    chunks: List[DocumentChunk] = []
    start_token = 0

    while start_token < len(tokens):
        chunk_start_token = start_token
        chunk_start_char = prefix_lengths[chunk_start_token]
        current_length = 0
        end_token = chunk_start_token

        while end_token < len(tokens):
            token_length = token_lengths[end_token]
            if current_length and current_length + token_length > chunk_size:
                break
            current_length += token_length
            end_token += 1
            if current_length >= chunk_size:
                break

        if end_token == chunk_start_token:
            token_length = token_lengths[end_token]
            chunk_end_char = prefix_lengths[end_token + 1]
            chunks.append(
                DocumentChunk(
                    text=tokens[end_token],
                    index=len(chunks),
                    start=chunk_start_char,
                    end=chunk_end_char,
                )
            )
            start_token = end_token + 1
            continue

        chunk_end_char = prefix_lengths[end_token]
        chunk_text_value = "".join(tokens[chunk_start_token:end_token])
        chunks.append(
            DocumentChunk(
                text=chunk_text_value,
                index=len(chunks),
                start=chunk_start_char,
                end=chunk_end_char,
            )
        )

        if end_token >= len(tokens):
            break

        if overlap == 0:
            start_token = end_token
            continue

        chunk_length = chunk_end_char - chunk_start_char
        if chunk_length <= 0:
            start_token = end_token
            continue

        desired_start_char = chunk_end_char - overlap
        if desired_start_char <= chunk_start_char:
            start_token = min(end_token, chunk_start_token + 1)
            continue

        new_start_token = bisect_left(
            prefix_lengths,
            desired_start_char,
            lo=chunk_start_token,
            hi=end_token,
        )
        if new_start_token == chunk_start_token:
            new_start_token = min(end_token, chunk_start_token + 1)
        start_token = new_start_token

    return chunks


class DocumentProcessor:
    """High level helper that wraps :func:`chunk_text`."""

    def __init__(self, *, chunk_size: int = 1_000, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[DocumentChunk]:
        """Return a list of chunks for *text*."""

        return chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)

    def iter_chunks(self, text: str) -> Iterator[DocumentChunk]:
        """Yield chunks for *text* lazily."""

        for chunk in self.chunk(text):
            yield chunk

    def chunk_file(self, path: Path | str, encoding: str = "utf-8") -> List[DocumentChunk]:
        """Read a file and return its chunks."""

        data = Path(path).read_text(encoding=encoding)
        return self.chunk(data)

    def iter_file_chunks(
        self, path: Path | str, encoding: str = "utf-8"
    ) -> Iterator[DocumentChunk]:
        """Yield chunks from a file lazily."""

        for chunk in self.chunk_file(path, encoding=encoding):
            yield chunk
