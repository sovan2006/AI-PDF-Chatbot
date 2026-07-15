import streamlit as st

from services.document_service import DocumentService
from services.rag_service import RAGService

from config import (
    UPLOAD_DIRECTORY,
    VECTORSTORE_DIRECTORY,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-header {
        padding: 1.5rem 1.8rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 18px;
        margin-bottom: 1.5rem;
    }

    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .app-subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 0;
    }

    .retrieval-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    .source-card {
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# INITIALIZE SERVICES
# ==================================================

@st.cache_resource
def initialize_services():

    document_service = DocumentService(
        upload_directory=UPLOAD_DIRECTORY,
        vectorstore_directory=VECTORSTORE_DIRECTORY,
    )

    rag_service = RAGService(
        vector_store=document_service.vector_store,
        embedding_manager=document_service.embedding_manager,
    )

    return document_service, rag_service


try:

    doc_service, rag_service = initialize_services()

except Exception as error:

    st.error(
        f"Failed to initialize AI services: {error}"
    )

    st.stop()


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_indexed_chunk_count():

    try:

        return (
            doc_service
            .vector_store
            .get_collection_count()
        )

    except Exception:

        return 0


def get_document_options():

    try:

        return (
            doc_service
            .vector_store
            .get_document_options()
        )

    except Exception as error:

        st.error(
            "Failed to load indexed documents: "
            f"{error}"
        )

        return {}


def render_sources(sources):

    if not sources:
        return

    with st.expander(
        f"📚 View Sources ({len(sources)})"
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

            file_name = source.get(
                "file",
                "Unknown document",
            )

            page = source.get(
                "page",
                "Unknown",
            )

            chunk_index = source.get(
                "chunk_index",
                "Unknown",
            )

            similarity_score = source.get(
                "similarity_score"
            )

            hybrid_score = source.get(
                "hybrid_score"
            )


            st.markdown(
                f"""
                <div class="source-card">

                <strong>📄 Source {index}</strong><br><br>

                <strong>File:</strong> {file_name}<br>

                <strong>Page:</strong> {page}<br>

                <strong>Chunk:</strong> {chunk_index}

                </div>
                """,
                unsafe_allow_html=True,
            )


            col1, col2 = st.columns(2)


            with col1:

                if similarity_score is not None:

                    st.metric(
                        "Semantic Similarity",
                        f"{similarity_score:.4f}",
                    )


            with col2:

                if hybrid_score is not None:

                    st.metric(
                        "Hybrid RRF Score",
                        f"{hybrid_score:.6f}",
                    )


            if index < len(sources):

                st.divider()


# ==================================================
# HEADER
# ==================================================

st.title(
    "🧠 AI Knowledge Assistant"
)

st.write(
    "Chat with your PDF documents using "
    "Retrieval-Augmented Generation."
)

st.caption(
    "⚡ Hybrid Retrieval · Dense + BM25 + RRF"
)

st.divider()

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title(
        "📚 Knowledge Base"
    )

    st.caption(
        "Upload and index PDF documents."
    )


    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
    )


    process_button = st.button(
        "🚀 Process Documents",
        use_container_width=True,
        type="primary",
    )


    st.divider()


    indexed_chunks = (
        get_indexed_chunk_count()
    )


    document_options = (
        get_document_options()
    )


    metric_col1, metric_col2 = (
        st.columns(2)
    )


    with metric_col1:

        st.metric(
            "Chunks",
            indexed_chunks,
        )


    with metric_col2:

        st.metric(
            "PDFs",
            len(document_options),
        )


    if indexed_chunks > 0:

        st.success(
            "Knowledge base ready"
        )

    else:

        st.info(
            "Upload a PDF to begin"
        )


    # ==============================================
    # SEARCH SCOPE
    # ==============================================

    st.divider()

    st.subheader(
        "🔎 Search Scope"
    )


    search_options = [
        "All Documents"
    ]


    search_options.extend(
        document_options.keys()
    )


    selected_document = st.selectbox(
        "Select knowledge source",
        options=search_options,
    )


    if (
        selected_document
        == "All Documents"
    ):

        selected_source = None

        st.caption(
            "🌐 Searching entire knowledge base"
        )


    else:

        selected_source = (
            document_options.get(
                selected_document
            )
        )

        st.caption(
            f"📄 Searching: {selected_document}"
        )


    st.divider()


    st.caption(
        "Retrieval Engine"
    )

    st.markdown(
        """
        **Hybrid Search**

        🧠 Dense Semantic Retrieval

        🔤 BM25 Keyword Retrieval

        🔀 Reciprocal Rank Fusion
        """
    )


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# DOCUMENT PROCESSING
# ==================================================

if process_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF document."
        )

    else:

        processing_results = []


        try:

            with st.spinner(
                "Extracting, chunking and indexing PDFs..."
            ):

                for uploaded_file in uploaded_files:

                    result = (
                        doc_service
                        .process_uploaded_file(
                            uploaded_file
                        )
                    )

                    processing_results.append(
                        result
                    )


            indexed_count = sum(
                result.get("status") == "indexed"
                for result in processing_results
            )


            duplicate_count = sum(
                result.get("status") == "duplicate"
                for result in processing_results
            )


            if indexed_count:

                st.success(
                    f"✅ {indexed_count} document(s) indexed."
                )


            if duplicate_count:

                st.warning(
                    f"⚠️ {duplicate_count} duplicate document(s)."
                )


            with st.expander(
                "📊 Processing Details"
            ):

                for result in processing_results:

                    filename = result.get(
                        "filename",
                        "Unknown File",
                    )

                    status = result.get(
                        "status",
                        "unknown",
                    )


                    st.subheader(
                        f"📄 {filename}"
                    )


                    st.write(
                        f"**Status:** {status}"
                    )


                    if status == "indexed":

                        col1, col2, col3 = (
                            st.columns(3)
                        )


                        col1.metric(
                            "Pages",
                            result.get(
                                "pages",
                                0,
                            ),
                        )


                        col2.metric(
                            "Chunks",
                            result.get(
                                "chunks",
                                0,
                            ),
                        )


                        col3.metric(
                            "Embeddings",
                            result.get(
                                "embeddings",
                                0,
                            ),
                        )


                    elif status == "duplicate":

                        st.info(
                            result.get(
                                "message",
                                "Document already indexed.",
                            )
                        )


                    st.divider()


            st.rerun()


        except Exception as error:

            st.error(
                "Document processing failed: "
                f"{error}"
            )


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==================================================
# EMPTY CHAT
# ==================================================

if not st.session_state.messages:

    st.info(
        """
        👋 **Your AI knowledge assistant is ready.**

        Upload PDF documents, select your search scope,
        and ask questions about the document content.
        """
    )


# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        render_sources(
            message.get(
                "sources",
                [],
            )
        )


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ==================================================
# PROCESS QUESTION
# ==================================================

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    try:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🔎 Running hybrid retrieval..."
            ):

                result = (
                    rag_service
                    .ask_question(
                        question=question,
                        top_k=5,
                        score_threshold=0.35,
                        source_filter=selected_source,
                    )
                )


            if isinstance(
                result,
                dict,
            ):

                answer = result.get(
                    "answer",
                    "I could not generate an answer.",
                )


                sources = result.get(
                    "sources",
                    [],
                )


            else:

                answer = str(
                    result
                )

                sources = []


            st.markdown(
                answer
            )


            render_sources(
                sources
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


    except Exception as error:

        st.error(
            "Failed to generate answer: "
            f"{error}"
        )