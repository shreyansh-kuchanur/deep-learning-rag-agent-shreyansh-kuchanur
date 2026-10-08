"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

Workshop flow (Markdown-only demo):
  1. Upload one or more .md notes in the sidebar and click "Ingest Documents"
  2. Chunks are embedded locally (all-MiniLM-L6-v2) and stored in ChromaDB
  3. Ask a question in the chat — the most relevant chunks are retrieved
     and passed to the Groq LLM, which answers only from that context

Run with: uv run streamlit run src/rag_agent/ui/app.py

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from rag_agent.agent.prompts import NO_CONTEXT_RESPONSE, SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource(show_spinner="Loading embedding model and ChromaDB...")
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_llm():
    """
    Return the cached chat model (Groq by default, set in .env).

    Cached so the LLM client is created once, not on every rerun.
    """
    return LLMFactory().create()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of Markdown files. Displays ingestion
    results (chunks added, duplicates skipped, errors) and the list
    of documents currently stored in ChromaDB.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Markdown notes",
        type=["md", "markdown"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "Ingest Documents", disabled=not uploaded_files
    ):
        all_chunks = []
        file_errors: list[str] = []

        with st.spinner("Chunking and embedding documents..."):
            # Save uploads to a temp folder, keeping the original filenames
            # so the filename-based metadata (topic_difficulty.md) still works.
            with tempfile.TemporaryDirectory() as tmp_dir:
                for uploaded in uploaded_files:
                    file_path = Path(tmp_dir) / Path(uploaded.name).name
                    file_path.write_bytes(uploaded.getvalue())
                    try:
                        all_chunks.extend(chunker.chunk_file(file_path))
                    except Exception as exc:
                        file_errors.append(f"{uploaded.name}: {exc}")

            result = store.ingest(all_chunks)

        result.errors.extend(file_errors)
        st.session_state.last_ingestion_result = result
        st.session_state.ingested_documents = store.list_documents()

    # Show the result of the most recent ingestion
    result = st.session_state.last_ingestion_result
    if result is not None:
        summary = (
            f"{result.ingested} chunks added, "
            f"{result.skipped} duplicates skipped"
        )
        if result.errors:
            st.sidebar.error(summary + "\n\n" + "\n".join(result.errors))
        elif result.ingested > 0:
            st.sidebar.success(summary)
        else:
            st.sidebar.warning(summary)

    # List of documents currently in ChromaDB
    documents = store.list_documents()
    st.session_state.ingested_documents = documents

    st.sidebar.subheader("Ingested documents")
    if not documents:
        st.sidebar.info("Upload .md files to populate the corpus.")
        return

    for doc in documents:
        col_name, col_btn = st.sidebar.columns([4, 1])
        col_name.caption(
            f"**{doc['source']}**  \n{doc['topic']} · {doc['chunk_count']} chunks"
        )
        if col_btn.button("🗑", key=f"remove_{doc['source']}", help="Remove"):
            store.delete_document(doc["source"])
            st.session_state.last_ingestion_result = None
            if st.session_state.selected_document == doc["source"]:
                st.session_state.selected_document = None
            st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present.

    Parameters
    ----------
    store : VectorStoreManager
    """
    stats = store.get_collection_stats()
    st.sidebar.divider()
    st.sidebar.metric("ChromaDB chunk count", stats["total_chunks"])
    if stats["topics"]:
        st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    documents = st.session_state.ingested_documents
    if not documents:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [doc["source"] for doc in documents]
    current = st.session_state.selected_document
    index = sources.index(current) if current in sources else 0

    selected = st.selectbox("Select document", options=sources, index=index)
    st.session_state.selected_document = selected

    chunks = store.get_document_chunks(selected)
    with st.container(height=500):
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            st.caption(
                f"Chunk {i} · {meta.topic} | {meta.difficulty} | {meta.type}"
            )
            st.markdown(chunk.chunk_text)
            st.divider()

    st.caption(f"{len(chunks)} chunks in {selected}")


# ---------------------------------------------------------------------------
# RAG Question Answering
# ---------------------------------------------------------------------------


def answer_question(
    query: str,
    store: VectorStoreManager,
    topic_filter: str | None = None,
    difficulty_filter: str | None = None,
) -> AgentResponse:
    """
    Retrieve relevant chunks and ask the LLM to answer from them only.

    Returns NO_CONTEXT_RESPONSE (hallucination guard) when no chunk
    passes the similarity threshold.
    """
    chunks: list[RetrievedChunk] = store.query(
        query,
        topic_filter=topic_filter,
        difficulty_filter=difficulty_filter,
    )

    if not chunks:
        return AgentResponse(answer=NO_CONTEXT_RESPONSE, no_context_found=True)

    context = "\n\n---\n\n".join(
        f"[SOURCE: {c.metadata.topic} | {c.metadata.source}] "
        f"(difficulty: {c.metadata.difficulty})\n{c.chunk_text}"
        for c in chunks
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {query}\n\n"
                "Answer the question using only the context above and cite "
                "the sources you used."
            )
        ),
    ]

    response = get_llm().invoke(messages)

    sources: list[str] = []
    for c in chunks:
        citation = f"{c.to_citation()} — similarity {c.score:.2f}"
        if citation not in sources:
            sources.append(citation)

    return AgentResponse(
        answer=str(response.content),
        sources=sources,
        confidence=sum(c.score for c in chunks) / len(chunks),
    )


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(store: VectorStoreManager) -> None:
    """
    Render the chat interface in the right column.

    Displays source citations with every response and a clear
    "no relevant context" indicator when the hallucination guard fires.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("💬 RAG Question & Answer")

    # Filters
    stats = store.get_collection_stats()
    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic = st.selectbox("Topic filter", ["All"] + stats["topics"])
        st.session_state.topic_filter = None if topic == "All" else topic
    with col_diff:
        difficulty = st.selectbox(
            "Difficulty filter", ["All", "beginner", "intermediate", "advanced"]
        )
        st.session_state.difficulty_filter = (
            None if difficulty == "All" else difficulty
        )

    # Chat history display
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources", expanded=True):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    query = st.chat_input("Ask a question about your notes...")
    if not query:
        return

    st.session_state.chat_history.append({"role": "user", "content": query})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    response = answer_question(
                        query,
                        store,
                        topic_filter=st.session_state.topic_filter,
                        difficulty_filter=st.session_state.difficulty_filter,
                    )
                    assistant_message = {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "no_context_found": response.no_context_found,
                    }
                except Exception as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"❌ Error while generating the answer: {exc}",
                    }

    st.session_state.chat_history.append(assistant_message)
    st.rerun()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, ChromaDB and Groq"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(store)


if __name__ == "__main__":
    main()