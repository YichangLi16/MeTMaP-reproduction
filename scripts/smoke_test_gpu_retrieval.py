from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name, device=device)

    documents = [
        "Metamorphic testing checks whether transformed inputs produce related outputs.",
        "Retrieval augmented generation uses vector search to find relevant context.",
        "A database index can improve query efficiency.",
        "The weather is sunny and suitable for outdoor activities.",
        "False vector matching can make a RAG system retrieve irrelevant documents.",
    ]
    query = "How can metamorphic testing find errors in RAG vector retrieval?"

    doc_embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False,
    ).astype("float32")
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    scores, ids = index.search(query_embedding, k=3)

    print("\nquery:")
    print(query)
    print("\ntop-3 results:")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        print(f"{rank}. score={score:.4f} | {documents[int(idx)]}")


if __name__ == "__main__":
    main()
