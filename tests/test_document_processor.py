from pathlib import Path

import pytest

from codex.document_processor import DocumentProcessor, chunk_text


def build_text(word: str, count: int, separator: str = " ") -> str:
    return separator.join(f"{word}{i}" for i in range(count))


def test_chunk_text_produces_overlapping_chunks():
    text = build_text("token", 1200)
    chunks = chunk_text(text, chunk_size=200, overlap=50)

    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert all(len(chunk.text) <= 200 or i == len(chunks) - 1 for i, chunk in enumerate(chunks))

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start >= previous.end - 50
        assert current.start <= previous.end


def test_chunk_text_handles_oversized_tokens():
    text = "A" * 500
    chunks = chunk_text(text, chunk_size=100, overlap=10)

    # Oversized token should still appear in output and not crash.
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start == 0 and chunks[0].end == len(text)


def test_chunk_file_roundtrip(tmp_path: Path):
    sample_text = "This is a test document.\n" * 100
    file_path = tmp_path / "doc.txt"
    file_path.write_text(sample_text)

    processor = DocumentProcessor(chunk_size=120, overlap=20)
    file_chunks = list(processor.iter_file_chunks(file_path))

    assert file_chunks[0].start == 0
    assert file_chunks[-1].end == len(sample_text)
    assert len(file_chunks) > 1
    assert file_chunks[1].start >= file_chunks[0].end - 20


def test_invalid_configuration():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=5, overlap=6)

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=5, overlap=-1)


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
