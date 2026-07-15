from pathlib import Path


class ContextBuilder:
    """
    Builds structured context for the RAG pipeline.

    Responsibilities:
    - Convert retrieved chunks into LLM context
    - Preserve source metadata
    - Limit maximum context length
    - Generate source information
    """

    def __init__(
        self,
        max_context_length=6000,
    ):

        if max_context_length <= 0:

            raise ValueError(
                "max_context_length must be greater than 0."
            )

        self.max_context_length = (
            max_context_length
        )


    # ==================================================
    # BUILD CONTEXT
    # ==================================================

    def build_context(
        self,
        retrieved_documents,
    ):
        """
        Build context and source metadata.

        Returns:

        {
            "context": str,
            "sources": list
        }
        """

        if retrieved_documents is None:

            raise ValueError(
                "Retrieved documents cannot be None."
            )


        if not retrieved_documents:

            return {
                "context": "",
                "sources": [],
            }


        context_blocks = []

        sources = []

        current_length = 0


        # ==================================================
        # PROCESS RETRIEVED DOCUMENTS
        # ==================================================

        for document in retrieved_documents:

            content = document.get(
                "document",
                "",
            )


            if not isinstance(
                content,
                str,
            ):

                continue


            content = content.strip()


            if not content:

                continue


            metadata = document.get(
                "metadata",
                {},
            ) or {}


            rank = document.get(
                "rank",
                len(context_blocks) + 1,
            )


            similarity_score = document.get(
                "similarity_score",
                0.0,
            )


            hybrid_score = document.get(
                "hybrid_score",
                None,
            )


            bm25_score = document.get(
                "bm25_score",
                None,
            )


            # ==================================================
            # SOURCE INFORMATION
            # ==================================================

            source_path = metadata.get(
                "source",
                "Unknown",
            )


            source_name = Path(
                source_path
            ).name


            page = metadata.get(
                "page"
            )


            if isinstance(
                page,
                int,
            ):

                page_number = page + 1

            else:

                page_number = "Unknown"


            chunk_index = metadata.get(
                "chunk_index",
                "Unknown",
            )


            # ==================================================
            # BUILD CONTEXT BLOCK
            # ==================================================

            context_block = (
                f"[SOURCE {rank}]\n"
                f"File: {source_name}\n"
                f"Page: {page_number}\n"
                f"Chunk: {chunk_index}\n\n"
                f"{content}\n"
            )


            block_length = len(
                context_block
            )


            # ==================================================
            # CONTEXT LENGTH LIMIT
            # ==================================================

            if (
                current_length
                + block_length
                > self.max_context_length
            ):

                break


            context_blocks.append(
                context_block
            )


            current_length += (
                block_length
            )


            # ==================================================
            # SOURCE METADATA
            # ==================================================

            source_data = {
                "rank": rank,
                "file": source_name,
                "page": page_number,
                "chunk_index": chunk_index,
            }


            if similarity_score is not None:

                try:

                    source_data[
                        "similarity_score"
                    ] = round(
                        float(
                            similarity_score
                        ),
                        4,
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass


            if hybrid_score is not None:

                try:

                    source_data[
                        "hybrid_score"
                    ] = round(
                        float(
                            hybrid_score
                        ),
                        4,
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass


            if bm25_score is not None:

                try:

                    source_data[
                        "bm25_score"
                    ] = round(
                        float(
                            bm25_score
                        ),
                        4,
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass


            sources.append(
                source_data
            )


        # ==================================================
        # JOIN CONTEXT
        # ==================================================

        context = "\n---\n\n".join(
            context_blocks
        )


        # ==================================================
        # LOGGING
        # ==================================================

        print(
            "Context Sources:",
            len(context_blocks),
        )


        print(
            "Context Length:",
            len(context),
        )


        # ==================================================
        # RETURN RESULT
        # ==================================================

        return {
            "context": context,
            "sources": sources,
        }