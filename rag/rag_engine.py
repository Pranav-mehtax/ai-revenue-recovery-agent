import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Directory containing this file:
# Razor_pay_agent/rag/

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Razor_pay_agent/rag/knowledge_base/

KNOWLEDGE_BASE_PATH = os.path.join(
    BASE_DIR,
    "knowledge_base"
)


TOP_K = 3


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_documents():

    documents = []

    print("Loading knowledge base...")

    if not os.path.exists(KNOWLEDGE_BASE_PATH):

        raise FileNotFoundError(
            f"Knowledge base not found at:\n"
            f"{KNOWLEDGE_BASE_PATH}"
        )

    for filename in sorted(
        os.listdir(KNOWLEDGE_BASE_PATH)
    ):

        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(
            KNOWLEDGE_BASE_PATH,
            filename
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        # Split at Markdown headings such as:
        # ## NETWORK_ERROR
        sections = re.split(
            r"\n(?=## )",
            text
        )

        for section in sections:

            section = section.strip()

            if not section:
                continue

            documents.append(
                {
                    "source": filename,
                    "text": section
                }
            )

    print(
        f"Loaded {len(documents)} knowledge chunks."
    )

    return documents


# ============================================================
# BUILD TF-IDF RETRIEVER
# ============================================================

documents = load_documents()

document_texts = [
    document["text"]
    for document in documents
]


if not document_texts:

    raise ValueError(
        "No knowledge chunks found."
    )


vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)


document_vectors = vectorizer.fit_transform(
    document_texts
)


# ============================================================
# RETRIEVE RELEVANT KNOWLEDGE
# ============================================================

def retrieve(
    query,
    top_k=TOP_K
):

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    # Don't request more results than we have
    k = min(
        top_k,
        len(documents)
    )

    top_indices = similarities.argsort()[
        ::-1
    ][:k]

    results = []

    for index in top_indices:

        results.append(
            {
                "source":
                    documents[index]["source"],

                "text":
                    documents[index]["text"],

                "score":
                    float(
                        similarities[index]
                    )
            }
        )

    return results


# ============================================================
# TEST RETRIEVER
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nAsk the recovery knowledge base: "
    )

    results = retrieve(
        query
    )

    print("\n")
    print("=" * 70)
    print("RAG RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\nQuery:\n{query}"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Result {i}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print("\nContent:")

        print(
            result["text"]
        )