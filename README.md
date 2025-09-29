# codex

Utilities for chunking long documents.

## Document processing

```python
from codex import DocumentProcessor

text = "..."  # a large document
processor = DocumentProcessor(chunk_size=1000, overlap=200)
chunks = processor.chunk(text)

for chunk in chunks:
    print(chunk.index, chunk.start, chunk.end)
```

The processor can also read content directly from files via
`DocumentProcessor.iter_file_chunks`.
