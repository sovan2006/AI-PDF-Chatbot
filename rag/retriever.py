from typing import Any


class RAGRetriever:
    """
    Dense semantic retriever using embeddings
    and ChromaDB vector search.
    """

    def __init__(
        self,
        embedding_manager,
        vector_store,
    ):

        if embedding_manager is None:
            raise ValueError(
                "Embedding manager cannot be None."
            )

        if vector_store is None:
            raise ValueError(
                "Vector store cannot be None."
            )

        self.embedding_manager = (
            embedding_manager
        )

        self.vector_store = (
            vector_store
        )

        print(
            "Dense Retriever Initialized."
        )


    # ==================================================
    # RETRIEVE
    # ==================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        document_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve semantically relevant chunks.

        Supports optional document-level filtering.
        """


        # ==================================================
        # VALIDATE QUERY
        # ==================================================

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


        # ==================================================
        # VALIDATE TOP K
        # ==================================================

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


        # ==================================================
        # VALIDATE SCORE THRESHOLD
        # ==================================================

        if not isinstance(
            score_threshold,
            (int, float),
        ):

            raise TypeError(
                "score_threshold must be numeric."
            )


        # ==================================================
        # COLLECTION COUNT
        # ==================================================

        collection_count = (
            self.vector_store
            .get_collection_count()
        )


        if collection_count == 0:

            print(
                "Vector store is empty."
            )

            return []


        # ==================================================
        # QUERY EMBEDDING
        # ==================================================

        query_embedding = (
            self.embedding_manager
            .generate_query_embedding(
                query
            )
        )


        if hasattr(
            query_embedding,
            "tolist",
        ):

            query_embedding = (
                query_embedding.tolist()
            )


        # ==================================================
        # RESULT COUNT
        # ==================================================

        n_results = min(
            top_k,
            collection_count,
        )


        # ==================================================
        # CHROMA WHERE FILTER
        # ==================================================

        where_filter = None


        if document_filter:

            if not isinstance(
                document_filter,
                str,
            ):

                raise TypeError(
                    "document_filter must be a string."
                )


            document_filter = (
                document_filter.strip()
            )


            if document_filter:

                where_filter = {
                    "source": document_filter
                }


        # ==================================================
        # VECTOR SEARCH
        # ==================================================

        query_arguments = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": n_results,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }


        if where_filter is not None:

            query_arguments[
                "where"
            ] = where_filter


        try:

            results = (
                self.vector_store
                .collection
                .query(
                    **query_arguments
                )
            )

        except Exception as error:

            raise RuntimeError(
                "Dense vector retrieval failed: "
                f"{error}"
            ) from error


        # ==================================================
        # VALIDATE RESULTS
        # ==================================================

        if (
            not results
            or not results.get(
                "documents"
            )
            or not results[
                "documents"
            ][0]
        ):

            print(
                "No dense documents retrieved "
                f"for query: '{query}'"
            )


            if document_filter:

                print(
                    "Dense Document Filter:",
                    document_filter,
                )


            return []


        # ==================================================
        # EXTRACT RESULTS
        # ==================================================

        ids = results.get(
            "ids",
            [[]],
        )[0]


        documents = results.get(
            "documents",
            [[]],
        )[0]


        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]


        distances = results.get(
            "distances",
            [[]],
        )[0]


        # ==================================================
        # BUILD RESULTS
        # ==================================================

        retrieved_documents = []


        for rank, (
            document_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            distance = float(
                distance
            )


            similarity_score = (
                1.0
                - distance
            )


            print(
                f"Dense Rank {rank} | "
                f"Distance: "
                f"{distance:.4f} | "
                f"Similarity: "
                f"{similarity_score:.4f}"
            )


            if (
                similarity_score
                < score_threshold
            ):

                continue


            retrieved_documents.append(
                {
                    "id": document_id,
                    "document": document,
                    "metadata": (
                        metadata
                        or {}
                    ),
                    "distance": distance,
                    "similarity_score": (
                        similarity_score
                    ),
                    "rank": rank,
                }
            )


        # ==================================================
        # LOG RESULT
        # ==================================================

        print()


        print(
            f"Dense Retrieved "
            f"{len(retrieved_documents)} "
            f"documents for query: "
            f"'{query}'"
        )


        if document_filter:

            print(
                "Dense Document Filter:",
                document_filter,
            )


        return retrieved_documents