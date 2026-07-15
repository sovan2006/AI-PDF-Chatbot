import os


class RetrieverComparisonEvaluator:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        hybrid_retriever,
    ):

        if dense_retriever is None:

            raise ValueError(
                "Dense retriever cannot be None."
            )


        if bm25_retriever is None:

            raise ValueError(
                "BM25 retriever cannot be None."
            )


        if hybrid_retriever is None:

            raise ValueError(
                "Hybrid retriever cannot be None."
            )


        self.retrievers = {
            "dense": dense_retriever,
            "bm25": bm25_retriever,
            "hybrid": hybrid_retriever,
        }


    # ==================================================
    # GET SOURCE FILE
    # ==================================================

    def _get_source_file(
        self,
        document,
    ):

        metadata = document.get(
            "metadata",
            {},
        )


        source = metadata.get(
            "source",
            "",
        )


        return os.path.basename(
            source
        )


    # ==================================================
    # EVALUATE SINGLE QUESTION
    # ==================================================

    def evaluate_question(
        self,
        retriever,
        question,
        relevant_document,
        top_k=5,
    ):

        retrieved_documents = (
            retriever.retrieve(
                query=question,
                top_k=top_k,
            )
        )


        relevant_rank = None


        for rank, document in enumerate(
            retrieved_documents,
            start=1,
        ):

            source_file = (
                self._get_source_file(
                    document
                )
            )


            if (
                source_file
                == relevant_document
            ):

                relevant_rank = rank

                break


        hit = relevant_rank is not None


        reciprocal_rank = (
            1 / relevant_rank
            if relevant_rank is not None
            else 0.0
        )


        recall_at_k = (
            1.0
            if hit
            else 0.0
        )


        return {
            "question": question,
            "hit": hit,
            "relevant_rank":
                relevant_rank,
            "reciprocal_rank": round(
                reciprocal_rank,
                4,
            ),
            "recall_at_k": recall_at_k,
            "retrieved_chunks": len(
                retrieved_documents
            ),
        }


    # ==================================================
    # EVALUATE RETRIEVER
    # ==================================================

    def evaluate_retriever(
        self,
        retriever,
        dataset,
        top_k=5,
    ):

        results = []


        for item in dataset:

            result = (
                self.evaluate_question(
                    retriever=retriever,
                    question=item[
                        "question"
                    ],
                    relevant_document=item[
                        "relevant_document"
                    ],
                    top_k=top_k,
                )
            )


            results.append(
                result
            )


        total_questions = len(
            results
        )


        hit_count = sum(
            result["hit"]
            for result in results
        )


        hit_rate = (
            hit_count / total_questions
            if total_questions > 0
            else 0.0
        )


        mean_reciprocal_rank = (
            sum(
                result[
                    "reciprocal_rank"
                ]
                for result in results
            )
            / total_questions
            if total_questions > 0
            else 0.0
        )


        mean_recall = (
            sum(
                result[
                    "recall_at_k"
                ]
                for result in results
            )
            / total_questions
            if total_questions > 0
            else 0.0
        )


        return {
            "hit_rate_at_k": round(
                hit_rate,
                4,
            ),
            "mrr": round(
                mean_reciprocal_rank,
                4,
            ),
            "recall_at_k": round(
                mean_recall,
                4,
            ),
            "results": results,
        }


    # ==================================================
    # COMPARE RETRIEVERS
    # ==================================================

    def compare(
        self,
        dataset,
        top_k=5,
    ):

        comparison = {}


        for name, retriever in (
            self.retrievers.items()
        ):

            print(
                "\nEvaluating:",
                name.upper(),
            )


            comparison[name] = (
                self.evaluate_retriever(
                    retriever=retriever,
                    dataset=dataset,
                    top_k=top_k,
                )
            )


        return comparison