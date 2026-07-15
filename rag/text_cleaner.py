import re


def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Replace multiple spaces/tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Replace excessive line breaks
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([.,!?;:])",
        r"\1",
        text
    )

    return text.strip()


def clean_documents(documents):

    if documents is None:
        raise ValueError(
            "Documents cannot be None."
        )

    cleaned_documents = []


    for document in documents:

        cleaned_content = clean_text(
            document.page_content
        )

        # Skip empty pages
        if not cleaned_content:
            continue

        document.page_content = cleaned_content

        cleaned_documents.append(
            document
        )


    print(
        f"Original Documents: "
        f"{len(documents)}"
    )

    print(
        f"Cleaned Documents: "
        f"{len(cleaned_documents)}"
    )


    return cleaned_documents