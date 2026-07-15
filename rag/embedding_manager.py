from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Handles document and query embedding generation.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model_name = model_name

        print(
            "Loading embedding model:",
            self.model_name
        )

        try:
            self.model = SentenceTransformer(
                self.model_name
            )

        except Exception as error:
            raise RuntimeError(
                "Failed to load embedding model: "
                f"{error}"
            ) from error

        # --------------------------------------------------
        # Get embedding dimension
        # --------------------------------------------------

        try:
            self.embedding_dimension = (
                self.model.get_embedding_dimension()
            )

        except Exception as error:
            raise RuntimeError(
                "Failed to determine "
                "embedding dimension."
            ) from error

        if self.embedding_dimension is None:
            raise RuntimeError(
                "Could not determine "
                "embedding dimension."
            )

        self.embedding_dimension = int(
            self.embedding_dimension
        )

        print(
            "Embedding Dimension:",
            self.embedding_dimension
        )

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not isinstance(texts, list):
            raise TypeError(
                "Texts must be provided as a list."
            )

        if not texts:
            raise ValueError(
                "Texts list cannot be empty."
            )

        cleaned_texts = []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "Every text must be a string."
                )

            text = text.strip()

            if text:
                cleaned_texts.append(text)

        if not cleaned_texts:
            raise ValueError(
                "No valid text found for embedding."
            )

        try:
            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )

        except Exception as error:
            raise RuntimeError(
                "Failed to generate embeddings: "
                f"{error}"
            ) from error

        print(
            "Embedding Shape:",
            embeddings.shape
        )

        return embeddings.tolist()

    def generate_query_embedding(
        self,
        query: str
    ) -> list[float]:
        """
        Generate embedding for a single query.
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

        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )

        except Exception as error:
            raise RuntimeError(
                "Failed to generate query embedding: "
                f"{error}"
            ) from error

        return embedding.tolist()

    def get_embedding_dimension(
        self
    ) -> int:
        """
        Return embedding vector dimension.
        """

        return self.embedding_dimension