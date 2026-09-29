from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

class Document():
    _chunks: list = []
    _document: DoclingDocument

    def __init__(self, filepath: str):
        converter = DocumentConverter()
        result = converter.convert(filepath)
        self._document = result.document

    def compute_chunks(self) -> list[str]:
        # TODO check if this is a good model for tokenization
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            max_tokens=25,
        )

        chunker = LineBasedTokenChunker(
            tokenizer=tokenizer,
            prefix="",  # No prefix for general documents,
            omit_prefix_on_overflow=False
        ) # pyright: ignore[reportCallIssue]

        chunks = list(chunker.chunk(self._document))
        chunk_texts = []
        for chunk in chunks:
            chunk_texts.append(chunk.text)

        return chunk_texts