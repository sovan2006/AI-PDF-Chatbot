import os
import hashlib

import chromadb


class VectorStoreManager:

    def __init__(
        self,
        persist_directory="vectorstore",
        collection_name="pdf_documents",
    ):

        self.persist_directory = os.path.abspath(
            persist_directory
        )

        self.collection_name = collection_name

        self.client = None
        self.collection = None

        self._initialize_store()


    # --------------------------------------------------
    # INITIALIZE VECTOR STORE
    # --------------------------------------------------

    def _initialize_store(self):

        os.makedirs(
            self.persist_directory,
            exist_ok=True,
        )

        try:

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self.collection = (
                self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "description":
                        "PDF documents for AI Knowledge Assistant",

                        "hnsw:space": "cosine",
                    },
                )
            )

        except Exception as error:

            raise RuntimeError(
                "Failed to initialize vector store: "
                f"{error}"
            ) from error


        print(
            "Vector Store Directory:",
            self.persist_directory,
        )

        print(
            "Initialized Collection:",
            self.collection_name,
        )

        print(
            "Documents in Collection:",
            self.collection.count(),
        )


    # --------------------------------------------------
    # CREATE UNIQUE CHUNK ID
    # --------------------------------------------------

    def _create_chunk_id(
        self,
        document,
    ):

        metadata = document.metadata

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

        content = document.page_content


        unique_content = (
            f"{source}|"
            f"{page}|"
            f"{chunk_index}|"
            f"{content}"
        )


        chunk_hash = hashlib.sha256(
            unique_content.encode("utf-8")
        ).hexdigest()


        return f"chunk_{chunk_hash}"


    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(
        self,
        documents,
        embeddings,
    ):

        if documents is None:

            raise ValueError(
                "Documents cannot be None."
            )


        if embeddings is None:

            raise ValueError(
                "Embeddings cannot be None."
            )


        if len(documents) == 0:

            raise ValueError(
                "Documents cannot be empty."
            )


        if len(documents) != len(embeddings):

            raise ValueError(
                "Number of documents and embeddings "
                "must be equal."
            )


        ids = []

        metadatas = []

        document_contents = []

        embedding_list = []


        for document, embedding in zip(
            documents,
            embeddings,
        ):

            doc_id = self._create_chunk_id(
                document
            )


            ids.append(
                doc_id
            )


            metadata = dict(
                document.metadata
            )


            metadatas.append(
                metadata
            )


            document_contents.append(
                document.page_content
            )


            if hasattr(
                embedding,
                "tolist",
            ):

                embedding = embedding.tolist()


            embedding_list.append(
                embedding
            )


        try:

            self.collection.upsert(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=document_contents,
            )

        except Exception as error:

            raise RuntimeError(
                "Failed to store documents: "
                f"{error}"
            ) from error


        print(
            f"Stored {len(documents)} chunks."
        )

        print(
            "Total Documents in Collection:",
            self.collection.count(),
        )


    # --------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------

    def search(
        self,
        query_embedding,
        top_k=5,
        source_filter=None,
    ):
        """
        Search ChromaDB using query embedding.

        Optional source_filter limits search
        to one specific PDF document.
        """

        if query_embedding is None:

            raise ValueError(
                "Query embedding cannot be None."
            )


        collection_count = (
            self.collection.count()
        )


        if collection_count == 0:

            return []


        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )


        if hasattr(
            query_embedding,
            "tolist",
        ):

            query_embedding = (
                query_embedding.tolist()
            )


        query_parameters = {
            "query_embeddings": [
                query_embedding
            ],

            "n_results": min(
                top_k,
                collection_count,
            ),

            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }


        # ----------------------------------------------
        # DOCUMENT SOURCE FILTER
        # ----------------------------------------------

        if source_filter:

            query_parameters[
                "where"
            ] = {
                "source": source_filter
            }


        try:

            results = self.collection.query(
                **query_parameters
            )

        except Exception as error:

            raise RuntimeError(
                "Vector search failed: "
                f"{error}"
            ) from error


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


        search_results = []


        for rank, (
            doc_id,
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
                1.0 - distance
            )


            search_results.append(
                {
                    "id": doc_id,

                    "document": document,

                    "metadata":
                    metadata or {},

                    "distance":
                    distance,

                    "similarity_score":
                    similarity_score,

                    "rank":
                    rank,
                }
            )


        return search_results


    # --------------------------------------------------
    # GET INDEXED DOCUMENT SOURCES
    # --------------------------------------------------

    def get_document_sources(self):
        """
        Return unique document source paths
        stored in ChromaDB.
        """

        try:

            results = self.collection.get(
                include=[
                    "metadatas"
                ]
            )


            metadatas = results.get(
                "metadatas",
                []
            )


            sources = set()


            for metadata in metadatas:

                if not metadata:

                    continue


                source = metadata.get(
                    "source"
                )


                if source:

                    sources.add(
                        source
                    )


            return sorted(
                sources
            )


        except Exception as error:

            raise RuntimeError(
                "Failed to retrieve document sources: "
                f"{error}"
            ) from error


    # --------------------------------------------------
    # GET DOCUMENT OPTIONS
    # --------------------------------------------------

    def get_document_options(self):
        """
        Return:

        {
            filename: source_path
        }
        """

        sources = (
            self.get_document_sources()
        )


        document_options = {}


        for source in sources:

            filename = os.path.basename(
                source
            )


            document_options[
                filename
            ] = source


        return document_options


    # --------------------------------------------------
    # COLLECTION COUNT
    # --------------------------------------------------

    def get_collection_count(self):

        return self.collection.count()