import os
import shutil

from config import (
    UPLOAD_DIRECTORY,
    VECTORSTORE_DIRECTORY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from rag.document_loader import load_all_pdfs
from rag.text_cleaner import clean_documents
from rag.text_splitter import split_documents
from rag.embedding_manager import EmbeddingManager
from rag.vector_store import VectorStoreManager


def index_documents():

    print("=" * 80)
    print("DOCUMENT INDEXING PIPELINE")
    print("=" * 80)

    print(f"\nUpload Directory: {UPLOAD_DIRECTORY}")
    print(f"Vector Store Directory: {VECTORSTORE_DIRECTORY}")

    # ---------------------------------------------------------
    # STEP 1: VALIDATE UPLOAD DIRECTORY
    # ---------------------------------------------------------

    if not os.path.exists(UPLOAD_DIRECTORY):
        raise FileNotFoundError(
            f"Upload directory does not exist: {UPLOAD_DIRECTORY}"
        )

    pdf_files = [
        filename
        for filename in os.listdir(UPLOAD_DIRECTORY)
        if filename.lower().endswith(".pdf")
    ]

    print(f"\nPDF Files Found: {len(pdf_files)}")

    for filename in pdf_files:
        print(f"  - {filename}")

    if not pdf_files:
        print("\nNo PDF files found.")
        print(f"Add PDF files to: {UPLOAD_DIRECTORY}")
        return

    # ---------------------------------------------------------
    # STEP 2: LOAD PDF DOCUMENTS
    # ---------------------------------------------------------

    documents = load_all_pdfs(UPLOAD_DIRECTORY)

    print(f"\nOriginal Documents: {len(documents)}")

    if not documents:
        print("\nNo PDF pages were loaded.")
        return

    # ---------------------------------------------------------
    # STEP 3: CLEAN DOCUMENTS
    # ---------------------------------------------------------

    cleaned_documents = clean_documents(documents)

    print(f"Cleaned Documents: {len(cleaned_documents)}")

    if not cleaned_documents:
        print("\nNo documents remained after cleaning.")
        return

    # ---------------------------------------------------------
    # STEP 4: SPLIT DOCUMENTS
    # ---------------------------------------------------------

    chunks = split_documents(
        cleaned_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(f"Generated Chunks: {len(chunks)}")

    if not chunks:
        print("\nNo chunks were generated.")
        return

    # ---------------------------------------------------------
    # STEP 5: REMOVE OLD VECTOR STORE
    # ---------------------------------------------------------

    if os.path.exists(VECTORSTORE_DIRECTORY):
        print("\nRemoving old vector store...")

        shutil.rmtree(VECTORSTORE_DIRECTORY)

        print("Old vector store removed.")

    # ---------------------------------------------------------
    # STEP 6: INITIALIZE EMBEDDING MODEL
    # ---------------------------------------------------------

    embedding_manager = EmbeddingManager()

    # ---------------------------------------------------------
    # STEP 7: INITIALIZE VECTOR STORE
    # ---------------------------------------------------------

    vector_store = VectorStoreManager(
        persist_directory=VECTORSTORE_DIRECTORY
    )

    # ---------------------------------------------------------
    # STEP 8: GENERATE EMBEDDINGS
    # ---------------------------------------------------------

    texts = [
        document.page_content
        for document in chunks
    ]

    print("\nGenerating embeddings...")

    embeddings = embedding_manager.generate_embeddings(texts)

    print(f"Generated Embeddings: {len(embeddings)}")

    # ---------------------------------------------------------
    # STEP 9: STORE DOCUMENTS
    # ---------------------------------------------------------

    print("\nStoring chunks in vector database...")

    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings,
    )

    # ---------------------------------------------------------
    # STEP 10: VERIFY VECTOR STORE
    # ---------------------------------------------------------

    document_count = vector_store.collection.count()

    print("\n" + "=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)

    print(f"PDF Files Indexed: {len(pdf_files)}")
    print(f"PDF Pages Loaded: {len(documents)}")
    print(f"Cleaned Documents: {len(cleaned_documents)}")
    print(f"Chunks Generated: {len(chunks)}")
    print(f"Documents in Vector Store: {document_count}")

    print("=" * 80)


if __name__ == "__main__":
    index_documents()