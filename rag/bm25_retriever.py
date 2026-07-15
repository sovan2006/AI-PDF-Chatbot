import re

import numpy as np

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(
        self,
        vector_store,
    ):

        if vector_store is None:
            raise ValueError(
                "Vector store cannot be None."
            )

        self.vector_store = vector_store

        print(
            "BM25 Retriever Initialized."
        )


    @staticmethod
    def _tokenize(text):
        """
        Normalize and tokenize text for BM25.
        """

        if not isinstance(text, str):
            return []

        text = text.lower()

        return re.findall(
            r"\b[a-z0-9_]+\b",
            text,
        )


    def _load_documents(
        self,
        document_filter=None,
    ):
        """
        Load indexed chunks from ChromaDB.

        document_filter:
            Exact source path stored in metadata.
        """

        collection_count = (
            self.vector_store
            .get_collection_count()
        )

        if collection_count == 0:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }


        get_arguments = {
            "include": [
                "documents",
                "metadatas",
            ],
        }


        if document_filter:

            get_arguments["where"] = {
                "source": document_filter,
            }


        try:

            results = (
                self.vector_store
                .collection
                .get(
                    **get_arguments
                )
            )

        except Exception as error:

            raise RuntimeError(
                "Failed to load documents "
                "for BM25 retrieval: "
                f"{error}"
            ) from error


        return {
            "ids": results.get(
                "ids",
                [],
            ),
            "documents": results.get(
                "documents",
                [],
            ),
            "metadatas": results.get(
                "metadatas",
                [],
            ),
        }


    def retrieve(
        self,
        query,
        top_k=5,
        document_filter=None,
    ):
        """
        Retrieve chunks using BM25 keyword search.
        """

        if not isinstance(query, str):

            raise TypeError(
                "Query must be a string."
            )


        query = query.strip()


        if not query:

            raise ValueError(
                "Query cannot be empty."
            )


        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )


        stored_data = self._load_documents(
            document_filter=document_filter,
        )


        ids = stored_data["ids"]

        documents = stored_data["documents"]

        metadatas = stored_data["metadatas"]


        if not documents:

            print(
                "No documents available "
                "for BM25 retrieval."
            )

            return []


        tokenized_corpus = [
            self._tokenize(document)
            for document in documents
        ]


        if not any(tokenized_corpus):

            print(
                "BM25 corpus contains "
                "no searchable tokens."
            )

            return []


        query_tokens = self._tokenize(
            query
        )


        if not query_tokens:

            return []


        bm25 = BM25Okapi(
            tokenized_corpus
        )


        scores = bm25.get_scores(
            query_tokens
        )


        ranked_indices = np.argsort(
            scores
        )[::-1]


        retrieved_documents = []


        for index in ranked_indices:

            score = float(
                scores[index]
            )


            if score <= 0:
                continue


            metadata = (
                metadatas[index]
                or {}
            )


            retrieved_documents.append(
                {
                    "id": ids[index],
                    "document": documents[index],
                    "metadata": metadata,
                    "bm25_score": score,
                    "rank": (
                        len(
                            retrieved_documents
                        )
                        + 1
                    ),
                }
            )


            if (
                len(retrieved_documents)
                >= top_k
            ):

                break


        print()


        for result in retrieved_documents:

            metadata = result[
                "metadata"
            ]

            print(
                f"BM25 Rank "
                f"{result['rank']} | "
                f"Score: "
                f"{result['bm25_score']:.4f} | "
                f"File: "
                f"{metadata.get('source', 'unknown')}"
            )


        print(
            f"\nBM25 Retrieved "
            f"{len(retrieved_documents)} "
            f"documents for query: "
            f"'{query}'"
        )


        if document_filter:

            print(
                "BM25 Document Filter:",
                document_filter,
            )


        return retrieved_documents