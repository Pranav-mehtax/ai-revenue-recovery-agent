import os
import re

from sentence_transformers import SentenceTransformer
import chromadb



# Directory containing this file:
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Razor_pay_agent/rag/knowledge_base/

KNOWLEDGE_BASE_PATH = os.path.join(
    BASE_DIR,
    "knowledge_base"
)

# Razor_pay_agent/rag/chroma_store/

CHROMA_PERSIST_DIR = os.path.join(
    BASE_DIR,
    "chroma_store"
)

COLLECTION_NAME = "razorpay_knowledge"

TOP_K = 3


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model (all-MiniLM-L6-v2)...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    # The model is provisioned with the application.  Avoid a network metadata
    # check during startup: offline and restricted environments otherwise retry
    # against Hugging Face before the dashboard can serve a request.
    local_files_only=True,
)

print("Embedding model loaded.")


# ============================================================
# STEP 1 – LOAD DOCUMENTS FROM KNOWLEDGE BASE
# ============================================================

def load_documents():
    """
    Read all .md files from the knowledge_base folder and
    return a flat list of dicts:
        { "source": <filename>, "text": <raw section text> }
    """

    documents = []

    print("Loading knowledge base...")

    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(
            f"Knowledge base not found at:\n{KNOWLEDGE_BASE_PATH}"
        )

    for filename in sorted(os.listdir(KNOWLEDGE_BASE_PATH)):

        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(
            KNOWLEDGE_BASE_PATH,
            filename
        )

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(
            {
                "source": filename,
                "text": text
            }
        )

    print(f"Loaded {len(documents)} documents.")

    return documents


# ============================================================
# STEP 2 – CHUNK DOCUMENTS
# ============================================================

def chunk_documents(documents):
    """
    Split each document into sections at Markdown H2 headings (## ...).
    Returns a list of dicts:
        { "id": <unique str>, "source": <filename>, "text": <chunk text> }
    """

    chunks = []

    for doc in documents:

        # Split on lines that begin a new ## heading
        sections = re.split(r"\n(?=## )", doc["text"])

        for idx, section in enumerate(sections):

            section = section.strip()

            if not section:
                continue

            chunk_id = f"{doc['source']}__chunk_{idx}"

            chunks.append(
                {
                    "id":     chunk_id,
                    "source": doc["source"],
                    "text":   section
                }
            )

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ============================================================
# STEP 3 – EMBED CHUNKS & INGEST INTO CHROMA DB
# ============================================================

def build_vector_store(chunks):
    """
    Embed every chunk with MiniLM-L6-v2 and upsert into a
    persistent ChromaDB collection.
    Returns the collection object.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # If already populated, skip re-ingestion
    if collection.count() == len(chunks):
        print(
            f"ChromaDB collection already has {collection.count()} "
            "documents. Skipping ingestion."
        )
        return collection

    print(
        f"Embedding {len(chunks)} chunks and ingesting into ChromaDB..."
    )

    texts     = [chunk["text"]   for chunk in chunks]
    ids       = [chunk["id"]     for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    # Embed all chunks in one batched call
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Ingested {collection.count()} chunks into ChromaDB."
    )

    return collection


# ============================================================
# STARTUP – BUILD / LOAD THE VECTOR STORE
# ============================================================

_documents = load_documents()
_chunks    = chunk_documents(_documents)
collection = build_vector_store(_chunks)


# ============================================================
# STEP 4 – RETRIEVE (QUERY EMBEDDING + VECTOR SEARCH)
# ============================================================

def retrieve(
    query,
    top_k=TOP_K
):
    """
    Embed the query with MiniLM-L6-v2, then perform a
    cosine-similarity search against the ChromaDB collection.

    Returns a list of dicts:
        { "source": str, "text": str, "score": float }
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    output = []

    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # ChromaDB returns cosine *distance* (0 = identical).
        # Convert to similarity score: similarity = 1 - distance
        similarity = 1.0 - distance

        output.append(
            {
                "source": meta["source"],
                "text":   text,
                "score":  similarity
            }
        )

    return output


# ============================================================
# TEST RETRIEVER  (python rag_engine.py)
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nAsk the recovery knowledge base: "
    )

    results = retrieve(query)

    print("\n")
    print("=" * 70)
    print("RAG RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"\nQuery:\n{query}")

    for i, result in enumerate(results, start=1):

        print("\n" + "-" * 70)
        print(f"Result {i}")
        print(f"Source:     {result['source']}")
        print(f"Similarity: {result['score']:.4f}")
        print("\nContent:")
        print(result["text"])
