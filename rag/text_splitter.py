from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=50
):

    if documents is None:
        raise ValueError(
            "Documents cannot be None."
        )

    if not documents:
        raise ValueError(
            "Documents cannot be empty."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunked_documents = (
        text_splitter.split_documents(
            documents
        )
    )

    # Add chunk index to metadata
    for index, document in enumerate(
        chunked_documents
    ):
        document.metadata["chunk_index"] = index

    print(
        f"Original Documents: "
        f"{len(documents)}"
    )

    print(
        f"Total Chunks Created: "
        f"{len(chunked_documents)}"
    )

    print(
        f"Chunk Size: {chunk_size}"
    )

    print(
        f"Chunk Overlap: {chunk_overlap}"
    )

    return chunked_documents