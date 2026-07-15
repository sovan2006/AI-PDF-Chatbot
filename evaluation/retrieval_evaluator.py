class RetrievalEvaluator:

    def __init__(
        self,
        retriever,
    ):

        if retriever is None:

            raise ValueError(
                "Retriever cannot be None."
            )

        self.retriever = retriever


    def evaluate_question(
        self,
        question,
        expected_keywords,
        top_k=5,
    ):

        retrieved_documents = (
            self.retriever.retrieve(
                query=question,
                top_k=top_k,
            )
        )


        if not retrieved_documents:

            return {
                "question": question,
                "hit": False,
                "recall": 0.0,
                "retrieved_chunks": 0,
            }


        combined_text = " ".join(
            document.get(
                "document",
                "",
            )
            for document
            in retrieved_documents
        ).lower()


        matched_keywords = []

        for keyword in expected_keywords:

            if keyword.lower() in combined_text:

                matched_keywords.append(
                    keyword
                )


        total_keywords = len(
            expected_keywords
        )


        matched_count = len(
            matched_keywords
        )


        recall = (
            matched_count / total_keywords
            if total_keywords > 0
            else 0.0
        )


        return {
            "question": question,
            "hit": matched_count > 0,
            "recall": round(
                recall,
                4,
            ),
            "matched_keywords":
                matched_keywords,
            "retrieved_chunks": len(
                retrieved_documents
            ),
        }


    def evaluate_dataset(
        self,
        dataset,
        top_k=5,
    ):

        results = []


        for item in dataset:

            result = (
                self.evaluate_question(
                    question=item[
                        "question"
                    ],
                    expected_keywords=item[
                        "expected_keywords"
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
            1
            for result in results
            if result["hit"]
        )


        average_recall = (
            sum(
                result["recall"]
                for result in results
            )
            / total_questions
            if total_questions > 0
            else 0.0
        )


        hit_rate = (
            hit_count / total_questions
            if total_questions > 0
            else 0.0
        )


        return {
            "total_questions":
                total_questions,
            "hit_rate": round(
                hit_rate,
                4,
            ),
            "average_keyword_recall":
                round(
                    average_recall,
                    4,
                ),
            "results": results,
        }