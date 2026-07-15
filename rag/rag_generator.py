class RAGGenerator:
    """
    Orchestrates the complete RAG generation pipeline.

    Pipeline:
    Query
        ↓
    Hybrid Retrieval
        ↓
    Build Context
        ↓
    Build Prompt
        ↓
    Generate LLM Answer
        ↓
    Return Answer and Sources
    """

    def __init__(
        self,
        retriever,
        context_builder,
        prompt_template,
        llm_manager,
    ):

        if retriever is None:
            raise ValueError(
                "Retriever cannot be None."
            )

        if context_builder is None:
            raise ValueError(
                "Context builder cannot be None."
            )

        if prompt_template is None:
            raise ValueError(
                "Prompt template cannot be None."
            )

        if llm_manager is None:
            raise ValueError(
                "LLM manager cannot be None."
            )

        self.retriever = retriever

        self.context_builder = (
            context_builder
        )

        self.prompt_template = (
            prompt_template
        )

        self.llm_manager = (
            llm_manager
        )


    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
        source_filter: str | None = None,
    ):
        """
        Generate a RAG-based answer.

        score_threshold is preserved in the public API
        for compatibility with RAGService.

        Hybrid retrieval uses rank fusion, so the dense
        similarity threshold is not forwarded directly
        to HybridRetriever.
        """

        # ==============================================
        # VALIDATE QUERY
        # ==============================================

        if not isinstance(
            query,
            str,
        ):

            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:

            raise ValueError(
                "Query cannot be empty."
            )


        # ==============================================
        # VALIDATE TOP K
        # ==============================================

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


        # ==============================================
        # VALIDATE SCORE THRESHOLD
        # ==============================================

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


        # ==============================================
        # RETRIEVE DOCUMENTS
        # ==============================================

        retrieved_documents = (
            self.retriever
            .retrieve(
                query=query,
                top_k=top_k,
                document_filter=source_filter,
            )
        )


        # ==============================================
        # HANDLE NO RESULTS
        # ==============================================

        if not retrieved_documents:

            return {
                "answer": (
                    "I could not find relevant "
                    "information in the selected "
                    "document or knowledge base."
                ),
                "sources": [],
            }


        # ==============================================
        # BUILD CONTEXT
        # ==============================================

        context_result = (
            self.context_builder
            .build_context(
                retrieved_documents
            )
        )

        context = context_result.get(
            "context",
            "",
        )

        sources = context_result.get(
            "sources",
            [],
        )


        # ==============================================
        # HANDLE EMPTY CONTEXT
        # ==============================================

        if not context:

            return {
                "answer": (
                    "I could not find relevant "
                    "information in the selected "
                    "document or knowledge base."
                ),
                "sources": [],
            }


        # ==============================================
        # BUILD PROMPT
        # ==============================================

        prompt = (
            self.prompt_template
            .build_prompt(
                query=query,
                context=context,
            )
        )


        # ==============================================
        # GENERATE ANSWER
        # ==============================================

        answer = (
            self.llm_manager
            .generate_response(
                prompt
            )
        )


        # ==============================================
        # VALIDATE ANSWER
        # ==============================================

        if answer is None:

            answer = (
                "I could not generate an answer."
            )

        answer = str(
            answer
        ).strip()

        if not answer:

            answer = (
                "I could not generate an answer."
            )


        # ==============================================
        # RETURN RESULT
        # ==============================================

        return {
            "answer": answer,
            "sources": sources,
        }