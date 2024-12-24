import os
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
import numpy as np
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
import json


def load_markdown_files(directory_path):
    # Initialize the DirectoryLoader with the specified directory and file pattern
    loader = DirectoryLoader(
        path=directory_path,
        glob="**/*.md",  # Adjust the pattern to match your file naming conventions
        loader_cls=UnstructuredMarkdownLoader,
    )

    # Load the documents
    documents = loader.load()

    return documents


def extract_sentences(documents):
    sentences = []
    for doc in documents:
        # Assuming each document's content is a single string
        sentences.extend(
            doc.page_content.split(". ")
        )  # Simple split; consider using NLP tools for better accuracy
    return [sentence.strip() for sentence in sentences if sentence]


def compute_embeddings(sentences, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings


def group_sentences(sentences, embeddings, similarity_threshold=0.75):
    groups = []
    current_group = [sentences[0]]
    current_embedding = embeddings[0].unsqueeze(0)

    for i in range(1, len(sentences) - 2):
        triplet = sentences[i : i + 3]
        triplet_embedding = embeddings[i : i + 3].mean(dim=0, keepdim=True)

        similarity = util.pytorch_cos_sim(current_embedding, triplet_embedding).item()

        if similarity >= similarity_threshold:
            current_group.extend(triplet)
            current_embedding = (
                current_embedding * len(current_group) + triplet_embedding * 3
            ) / (len(current_group) + 3)
        else:
            groups.append(" ".join(current_group))
            current_group = triplet
            current_embedding = triplet_embedding

    if current_group:
        groups.append(" ".join(current_group))

    return groups


def save_grouped_sentences(grouped_sentences, file_path):
    """
    Save grouped sentences to a JSON file.

    Parameters:
    - grouped_sentences (list of dict): A list where each dictionary contains
      'group_id' (int) and 'sentences' (list of str).
    - file_path (str): The path to the file where the data should be saved.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(grouped_sentences, file, ensure_ascii=False, indent=4)
        print(f"Grouped sentences successfully saved to {file_path}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")


def main():
    documents = load_markdown_files(
        "/Users/sanderhergarten/datasources/bookhelper/markdown_files/"
    )
    sentences = extract_sentences(documents)
    embeddings = compute_embeddings(sentences)
    grouped_sentences = group_sentences(sentences, embeddings)
    save_grouped_sentences(
        grouped_sentences,
        "/Users/sanderhergarten/datasources/bookhelper/grouped_sentences.json",
    )


if __name__ == "__main__":
    main()
