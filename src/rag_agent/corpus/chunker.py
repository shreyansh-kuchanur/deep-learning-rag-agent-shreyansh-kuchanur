"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager

try:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
except ImportError:  # older LangChain layout
    from langchain.text_splitter import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

# Canonical topic names (lower-case filename part → display name)
KNOWN_TOPICS = {
    "ann": "ANN",
    "cnn": "CNN",
    "rnn": "RNN",
    "lstm": "LSTM",
    "seq2seq": "Seq2Seq",
    "autoencoder": "Autoencoder",
    "som": "SOM",
    "boltzmannmachine": "BoltzmannMachine",
    "boltzmann": "BoltzmannMachine",
    "gan": "GAN",
}
BONUS_TOPICS = {"SOM", "BoltzmannMachine", "GAN"}
DIFFICULTIES = {"beginner", "intermediate", "advanced"}


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        file_path = Path(file_path)

        # 1. Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 2. Route to the right loader based on the file extension
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        elif suffix == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix} ({file_path.name})")

        # 3. Metadata from the filename, with overrides applied
        metadata = self._infer_metadata(file_path, metadata_overrides)

        # 4. Build DocumentChunks with deterministic IDs
        chunks: list[DocumentChunk] = []
        for raw in raw_chunks:
            text = raw["text"].strip()
            if not text:
                continue
            chunk_id = VectorStoreManager.generate_chunk_id(metadata.source, text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata(
                        topic=metadata.topic,
                        difficulty=metadata.difficulty,
                        type=metadata.type,
                        source=metadata.source,
                        related_topics=list(metadata.related_topics),
                        is_bonus=metadata.is_bonus,
                    ),
                )
            )

        logger.info(f"Chunked '{file_path.name}' into {len(chunks)} chunks")
        # 5. Return list[DocumentChunk]
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        all_chunks: list[DocumentChunk] = []
        for file_path in file_paths:
            try:
                all_chunks.extend(self.chunk_file(file_path, metadata_overrides))
            except Exception as exc:
                logger.error(f"Failed to chunk '{file_path}': {exc}")
        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        # Optional for the workshop (Markdown-only demo)
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(file_path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        raw_chunks: list[dict] = []
        for page in pages:
            for text in splitter.split_text(page.page_content):
                raw_chunks.append({"text": text, "page": page.metadata.get("page", 0)})
        return raw_chunks

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        # utf-8-sig also handles files saved by Notepad with a BOM
        text = file_path.read_text(encoding="utf-8-sig", errors="replace")

        # 1. Split on headings so each section stays together
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,  # keep the heading text inside the chunk
        )
        sections = header_splitter.split_text(text)

        # 2. Split any section that is still too long
        size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        raw_chunks: list[dict] = []
        for section in sections:
            header = " > ".join(
                section.metadata[key]
                for key in ("h1", "h2", "h3")
                if key in section.metadata
            )
            for piece in size_splitter.split_text(section.page_content):
                if piece.strip():
                    raw_chunks.append({"text": piece, "header": header})
        return raw_chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        # Filename convention: <topic>_<difficulty>.md  e.g. lstm_intermediate.md
        parts = file_path.stem.split("_")
        topic_part = parts[0] if parts else file_path.stem
        topic = KNOWN_TOPICS.get(topic_part.lower(), topic_part.upper())

        difficulty = "intermediate"
        for part in parts[1:]:
            if part.lower() in DIFFICULTIES:
                difficulty = part.lower()
                break

        values = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": topic in BONUS_TOPICS,
        }

        # Explicit overrides win over anything inferred from the filename
        if overrides:
            for key, value in overrides.items():
                if key in values and value is not None:
                    values[key] = value
            if "topic" in overrides and "is_bonus" not in overrides:
                values["is_bonus"] = values["topic"] in BONUS_TOPICS

        return ChunkMetadata(**values)