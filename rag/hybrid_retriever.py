from typing import Any


class HybridRetriever:
    """
    Hybrid retrieval using weighted Reciprocal Rank Fusion.

    Dense Semantic Retrieval
            +
    BM25 Keyword Retrieval
            ↓
    Weighted Reciprocal Rank Fusion
    """


    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):

        if dense_retriever is None:

            raise ValueError(
                "Dense retriever cannot be None."
            )


        if bm25_retriever is None:

            raise ValueError(
                "BM25 retriever cannot be None."
            )


        if not isinstance(
            dense_weight,
            (int, float),
        ):

            raise TypeError(
                "dense_weight must be numeric."
            )


        if not isinstance(
            bm25_weight,
            (int, float),
        ):

            raise TypeError(
                "bm25_weight must be numeric."
            )


        if dense_weight < 0:

            raise ValueError(
                "dense_weight cannot be negative."
            )


        if bm25_weight < 0:

            raise ValueError(
                "bm25_weight cannot be negative."
            )


        if (
            dense_weight
            + bm25_weight
            <= 0
        ):

            raise ValueError(
                "At least one retrieval weight "
                "must be greater than zero."
            )


        if not isinstance(
            rrf_k,
            int,
        ):

            raise TypeError(
                "rrf_k must be an integer."
            )


        if rrf_k <= 0:

            raise ValueError(
                "rrf_k must be greater than 0."
            )


        total_weight = (
            dense_weight
            + bm25_weight
        )


        self.dense_retriever = (
            dense_retriever
        )

        self.bm25_retriever = (
            bm25_retriever
        )


        self.dense_weight = (
            float(dense_weight)
            / total_weight
        )


        self.bm25_weight = (
            float(bm25_weight)
            / total_weight
        )


        self.rrf_k = rrf_k


        print(
            "Hybrid Retriever Initialized."
        )

        print(
            "Dense Weight:",
            self.dense_weight,
        )

        print(
            "BM25 Weight:",
            self.bm25_weight,
        )

        print(
            "RRF K:",
            self.rrf_k,
        )


    # ==================================================
    # DOCUMENT KEY
    # ==================================================

    @staticmethod
    def _document_key(
        document: dict[str, Any],
    ):
        """
        Generate a stable identity for one chunk.
        """

        if not isinstance(
            document,
            dict,
        ):

            raise TypeError(
                "Retriever result must be a dictionary."
            )


        document_id = document.get(
            "id"
        )


        if document_id:

            return (
                "id",
                str(document_id),
            )


        metadata = (
            document.get(
                "metadata",
                {},
            )
            or {}
        )


        source = metadata.get(
            "source",
            "unknown",
        )


        page = metadata.get(
            "page",
            0,
        )


        chunk_index = metadata.get(
            "chunk_index",
            0,
        )


        content = document.get(
            "document",
            "",
        )


        return (
            str(source),
            page,
            chunk_index,
            str(content),
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
        Retrieve chunks using weighted RRF.
        """


        # ==================================================
        # VALIDATE INPUT
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


        if not isinstance(
            score_threshold,
            (int, float),
        ):

            raise TypeError(
                "score_threshold must be numeric."
            )


        # ==================================================
        # CANDIDATE COUNT
        # ==================================================

        candidate_k = max(
            top_k * 4,
            20,
        )


        # ==================================================
        # DENSE RETRIEVAL
        # ==================================================

        dense_results = (
            self.dense_retriever
            .retrieve(
                query=query,
                top_k=candidate_k,
                score_threshold=score_threshold,
                document_filter=document_filter,
            )
        )


        # ==================================================
        # BM25 RETRIEVAL
        # ==================================================

        bm25_results = (
            self.bm25_retriever
            .retrieve(
                query=query,
                top_k=candidate_k,
                document_filter=document_filter,
            )
        )


        # ==================================================
        # WEIGHTED RRF
        # ==================================================

        fused_results = {}


        for dense_rank, document in enumerate(
            dense_results,
            start=1,
        ):

            key = self._document_key(
                document
            )


            if key not in fused_results:

                fused_results[key] = {
                    "document": document,
                    "hybrid_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }


            fused_results[
                key
            ][
                "hybrid_score"
            ] += (
                self.dense_weight
                / (
                    self.rrf_k
                    + dense_rank
                )
            )


            fused_results[
                key
            ][
                "dense_rank"
            ] = dense_rank


        for bm25_rank, document in enumerate(
            bm25_results,
            start=1,
        ):

            key = self._document_key(
                document
            )


            if key not in fused_results:

                fused_results[key] = {
                    "document": document,
                    "hybrid_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }


            fused_results[
                key
            ][
                "hybrid_score"
            ] += (
                self.bm25_weight
                / (
                    self.rrf_k
                    + bm25_rank
                )
            )


            fused_results[
                key
            ][
                "bm25_rank"
            ] = bm25_rank


        # ==================================================
        # SORT RESULTS
        # ==================================================

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: item[
                "hybrid_score"
            ],
            reverse=True,
        )


        # ==================================================
        # BUILD STANDARD RESULT CONTRACT
        # ==================================================

        retrieved_documents = []


        for hybrid_rank, item in enumerate(
            ranked_results[:top_k],
            start=1,
        ):

            original_document = item[
                "document"
            ]


            result = dict(
                original_document
            )


            result[
                "hybrid_score"
            ] = float(
                item["hybrid_score"]
            )


            result[
                "dense_rank"
            ] = item[
                "dense_rank"
            ]


            result[
                "bm25_rank"
            ] = item[
                "bm25_rank"
            ]


            result[
                "rank"
            ] = hybrid_rank


            retrieved_documents.append(
                result
            )


        # ==================================================
        # LOG RESULTS
        # ==================================================

        print()


        for result in retrieved_documents:

            metadata = (
                result.get(
                    "metadata",
                    {},
                )
                or {}
            )


            print(
                f"Hybrid Rank "
                f"{result['rank']} | "
                f"RRF: "
                f"{result['hybrid_score']:.6f} | "
                f"Dense Rank: "
                f"{result.get('dense_rank')} | "
                f"BM25 Rank: "
                f"{result.get('bm25_rank')} | "
                f"File: "
                f"{metadata.get('source', 'unknown')}"
            )


        print(
            f"\nHybrid Retrieved "
            f"{len(retrieved_documents)} "
            f"documents for query: "
            f"'{query}'"
        )


        if document_filter:

            print(
                "Hybrid Document Filter:",
                document_filter,
            )


        return retrieved_documents