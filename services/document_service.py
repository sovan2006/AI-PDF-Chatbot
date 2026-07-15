import hashlib
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader

from rag.text_cleaner import clean_documents
from rag.text_splitter import split_documents
from rag.embedding_manager import EmbeddingManager
from rag.vector_store import VectorStoreManager


class DocumentService:
    """
    Handles PDF upload, processing, and indexing.

    Pipeline:
        Uploaded PDF
        -> Validate
        -> SHA-256 Hash
        -> Duplicate Check
        -> Save
        -> Load
        -> Clean
        -> Chunk
        -> Embed
        -> ChromaDB
    """

    def __init__(
        self,
        upload_directory: str,
        vectorstore_directory: str,
        collection_name: str = "pdf_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.upload_directory = Path(
            upload_directory
        ).resolve()

        self.vectorstore_directory = Path(
            vectorstore_directory
        ).resolve()

        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.upload_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vectorstore_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.embedding_manager = (
            EmbeddingManager()
        )

        self.vector_store = (
            VectorStoreManager(
                persist_directory=str(
                    self.vectorstore_directory
                ),
                collection_name=self.collection_name
            )
        )

        print(
            "Document Service Initialized."
        )

    @staticmethod
    def _calculate_file_hash(
        file_bytes: bytes
    ) -> str:
        return hashlib.sha256(
            file_bytes
        ).hexdigest()

    def _is_file_indexed(
        self,
        file_hash: str
    ) -> bool:
        result = self.vector_store.collection.get(
            where={
                "file_hash": file_hash
            },
            limit=1
        )

        return bool(
            result
            and result.get("ids")
        )

    def save_uploaded_file(
        self,
        uploaded_file: Any
    ) -> tuple[Path, str]:
        if uploaded_file is None:
            raise ValueError(
                "Uploaded file cannot be None."
            )

        filename = Path(
            uploaded_file.name
        ).name

        if not filename.lower().endswith(
            ".pdf"
        ):
            raise ValueError(
                "Only PDF files are supported."
            )

        file_bytes = uploaded_file.getvalue()

        if not file_bytes:
            raise ValueError(
                "Uploaded PDF is empty."
            )

        file_hash = (
            self._calculate_file_hash(
                file_bytes
            )
        )

        file_path = (
            self.upload_directory
            / filename
        )

        file_path.write_bytes(
            file_bytes
        )

        print(
            f"Saved PDF: {filename}"
        )

        return file_path, file_hash

    def load_pdf(
        self,
        file_path: Path
    ):
        try:
            loader = PyPDFLoader(
                str(file_path)
            )

            documents = loader.load()

        except Exception as error:
            raise RuntimeError(
                f"Failed to load PDF: {error}"
            ) from error

        if not documents:
            raise ValueError(
                "No readable text found in PDF."
            )

        print(
            f"Loaded PDF: {file_path.name}"
        )

        print(
            "Total Pages:",
            len(documents)
        )

        return documents

    def process_uploaded_file(
        self,
        uploaded_file: Any
    ) -> dict:
        file_path, file_hash = (
            self.save_uploaded_file(
                uploaded_file
            )
        )

        if self._is_file_indexed(
            file_hash
        ):
            return {
                "status": "duplicate",
                "filename": file_path.name,
                "message": (
                    "This PDF is already indexed."
                ),
                "indexed_documents": (
                    self.vector_store
                    .get_collection_count()
                )
            }

        documents = self.load_pdf(
            file_path
        )

        for document in documents:
            document.metadata[
                "file_hash"
            ] = file_hash

            document.metadata[
                "filename"
            ] = file_path.name

        cleaned_documents = (
            clean_documents(
                documents
            )
        )

        chunks = split_documents(
            documents=cleaned_documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        if not chunks:
            raise ValueError(
                "No document chunks were created."
            )

        chunk_texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_manager
            .generate_embeddings(
                chunk_texts
            )
        )

        self.vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings
        )

        result = {
            "status": "indexed",
            "filename": file_path.name,
            "file_hash": file_hash,
            "pages": len(documents),
            "cleaned_pages": len(
                cleaned_documents
            ),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "indexed_documents": (
                self.vector_store
                .get_collection_count()
            )
        }

        print(
            "Document Processing Complete:"
        )

        print(result)

        return result
    