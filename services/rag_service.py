from rag.retriever import RAGRetriever
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.context_builder import ContextBuilder
from rag.prompt_template import RAGPromptTemplate
from rag.rag_generator import RAGGenerator

from services.llm_manager import LLMManager


class RAGService:
    """
    Main service for the Retrieval-Augmented
    Generation pipeline.
    """

    def __init__(
        self,
        vector_store,
        embedding_manager,
    ):

        if vector_store is None:
            raise ValueError(
                "Vector store cannot be None."
            )

        if embedding_manager is None:
            raise ValueError(
                "Embedding manager cannot be None."
            )

        self.vector_store = vector_store

        self.embedding_manager = (
            embedding_manager
        )

        # ==========================================
        # DENSE RETRIEVER
        # ==========================================

        self.dense_retriever = RAGRetriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
        )

        # ==========================================
        # BM25 RETRIEVER
        # ==========================================

        self.bm25_retriever = BM25Retriever(
            vector_store=self.vector_store,
        )

        # ==========================================
        # HYBRID RETRIEVER
        # ==========================================

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
        )

        # ==========================================
        # DEFAULT RETRIEVER
        # ==========================================

        self.retriever = self.hybrid_retriever

        # ==========================================
        # CONTEXT BUILDER
        # ==========================================

        self.context_builder = ContextBuilder(
            max_context_length=6000,
        )

        # ==========================================
        # PROMPT TEMPLATE
        # ==========================================

        self.prompt_template = (
            RAGPromptTemplate()
        )

        # ==========================================
        # LLM MANAGER
        # ==========================================

        self.llm_manager = LLMManager()

        # ==========================================
        # RAG GENERATOR
        # ==========================================

        self.rag_generator = RAGGenerator(
            retriever=self.retriever,
            context_builder=self.context_builder,
            prompt_template=self.prompt_template,
            llm_manager=self.llm_manager,
        )

        print(
            "RAG Service Initialized."
        )

        print(
            "Retrieval Mode: HYBRID"
        )

    # ==============================================
    # ASK QUESTION
    # ==============================================

    def ask_question(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
        source_filter: str | None = None,
    ):
        """
        Ask a question using the complete
        Hybrid RAG pipeline.
        """

        # ==========================================
        # VALIDATE QUESTION
        # ==========================================

        if not isinstance(
            question,
            str,
        ):

            raise TypeError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        # ==========================================
        # VALIDATE TOP K
        # ==========================================

        if not isinstance(
            top_k,
            int,
        ):

            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )

        # ==========================================
        # VALIDATE SCORE THRESHOLD
        # ==========================================

        if not isinstance(
            score_threshold,
            (int, float),
        ):

            raise TypeError(
                "score_threshold must be numeric."
            )

        if not 0 <= score_threshold <= 1:

            raise ValueError(
                "score_threshold must be "
                "between 0 and 1."
            )

        # ==========================================
        # VALIDATE SOURCE FILTER
        # ==========================================

        if source_filter is not None:

            if not isinstance(
                source_filter,
                str,
            ):

                raise TypeError(
                    "source_filter must be "
                    "a string or None."
                )

            source_filter = (
                source_filter.strip()
            )

            if not source_filter:

                source_filter = None

        # ==========================================
        # LOG REQUEST
        # ==========================================

        print(
            "\n"
            "========================================"
        )

        print(
            "RAG QUESTION:",
            question,
        )

        print(
            "RAG TOP K:",
            top_k,
        )

        print(
            "RAG SCORE THRESHOLD:",
            score_threshold,
        )

        print(
            "RAG SOURCE FILTER:",
            source_filter or "ALL DOCUMENTS",
        )

        print(
            "========================================"
        )

        # ==========================================
        # GENERATE ANSWER
        # ==========================================

        result = (
            self.rag_generator
            .generate_answer(
                query=question,
                top_k=top_k,
                score_threshold=score_threshold,
                source_filter=source_filter,
            )
        )

        return result