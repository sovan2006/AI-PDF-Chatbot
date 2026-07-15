import os

from langchain_community.document_loaders import PyPDFLoader


def load_all_pdfs(folder_path):

    if not folder_path:
        raise ValueError(
            "PDF folder path cannot be empty."
        )

    folder_path = os.path.abspath(
        folder_path
    )

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"PDF folder not found: {folder_path}"
        )

    all_documents = []

    total_pdfs = 0


    for filename in sorted(
        os.listdir(folder_path)
    ):

        if not filename.lower().endswith(
            ".pdf"
        ):
            continue


        pdf_path = os.path.abspath(
            os.path.join(
                folder_path,
                filename
            )
        )


        try:

            print(
                f"Loading PDF: {filename}"
            )


            loader = PyPDFLoader(
                pdf_path
            )


            documents = loader.load()


            for document in documents:

                document.metadata["source"] = (
                    pdf_path
                )

                document.metadata["filename"] = (
                    filename
                )


            all_documents.extend(
                documents
            )


            total_pdfs += 1


            print(
                f"Loaded PDF: {filename} "
                f"({len(documents)} pages)"
            )


        except Exception as error:

            print(
                f"Failed to load PDF "
                f"'{filename}': {error}"
            )


    print(
        f"\nTotal PDFs loaded: "
        f"{total_pdfs}"
    )


    print(
        f"Total pages loaded: "
        f"{len(all_documents)}"
    )


    return all_documents