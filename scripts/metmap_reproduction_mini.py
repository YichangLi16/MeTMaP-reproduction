from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class TestCase:
    case_id: str
    mr_id: str
    expected_doc_id: str
    source_query: str
    followup_query: str
    negative_query: str | None = None


DOCUMENTS = [
    Document(
        "rag_false_match",
        "False vector matching in RAG systems retrieves irrelevant context and can cause generated answers to contain unsupported claims.",
    ),
    Document(
        "rag_retrieval",
        "Retrieval augmented generation first retrieves relevant documents from a vector database and then feeds them to a language model.",
    ),
    Document(
        "metamorphic_testing",
        "Metamorphic testing generates follow-up test cases from source test cases and checks whether their outputs satisfy expected relations.",
    ),
    Document(
        "embedding_similarity",
        "Text embedding models map sentences into vectors so that semantic similarity can be estimated by vector distance or cosine similarity.",
    ),
    Document(
        "database_index",
        "A database index improves query performance by reducing the number of records that must be scanned.",
    ),
    Document(
        "weather_forecast",
        "A weather forecast predicts temperature, rain, wind and other atmospheric conditions for a future time period.",
    ),
    Document(
        "software_testing",
        "Software testing evaluates a system by executing test cases and observing whether the behavior satisfies expected requirements.",
    ),
    Document(
        "machine_learning",
        "Machine learning models learn patterns from data and are commonly evaluated with accuracy, precision, recall and robustness metrics.",
    ),
]


TEST_CASES = [
    TestCase(
        case_id="tc01",
        mr_id="MR-Similar",
        expected_doc_id="rag_false_match",
        source_query="How can false vector matching harm a RAG system?",
        followup_query="Why is incorrect vector matching dangerous for retrieval augmented generation?",
    ),
    TestCase(
        case_id="tc02",
        mr_id="MR-Similar",
        expected_doc_id="metamorphic_testing",
        source_query="What is the main idea of metamorphic testing?",
        followup_query="How does metamorphic testing use follow-up test cases to find faults?",
    ),
    TestCase(
        case_id="tc03",
        mr_id="MR-Irrelevant-Noise",
        expected_doc_id="rag_retrieval",
        source_query="How does retrieval augmented generation use vector search?",
        followup_query="In a classroom homework example, how does retrieval augmented generation use vector search?",
    ),
    TestCase(
        case_id="tc04",
        mr_id="MR-Irrelevant-Noise",
        expected_doc_id="embedding_similarity",
        source_query="How do text embeddings represent semantic similarity?",
        followup_query="For a temporary experiment with unrelated background details, how do text embeddings represent semantic similarity?",
    ),
    TestCase(
        case_id="tc05",
        mr_id="MR-Contradictory",
        expected_doc_id="rag_false_match",
        source_query="False vector matching can make RAG systems retrieve irrelevant documents.",
        followup_query="Wrong vector matches may retrieve unrelated context in RAG pipelines.",
        negative_query="Weather forecasts predict rain and temperature for tomorrow.",
    ),
    TestCase(
        case_id="tc06",
        mr_id="MR-Contradictory",
        expected_doc_id="metamorphic_testing",
        source_query="Metamorphic testing checks relations between source and follow-up outputs.",
        followup_query="Follow-up test cases should preserve expected output relations in metamorphic testing.",
        negative_query="A database index can speed up SQL query execution.",
    ),
]


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    ).astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index: faiss.IndexFlatIP, query_embedding: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    scores, ids = index.search(query_embedding, k)
    return ids[0].astype(int).tolist(), scores[0].astype(float).tolist()


def rank_of_doc(ids: list[int], documents: list[Document], expected_doc_id: str) -> int | None:
    for rank, idx in enumerate(ids, start=1):
        if documents[idx].doc_id == expected_doc_id:
            return rank
    return None


def evaluate_case(
    case: TestCase,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    documents: list[Document],
    k: int,
) -> dict[str, object]:
    queries = [case.source_query, case.followup_query]
    if case.negative_query:
        queries.append(case.negative_query)

    query_embeddings = encode(model, queries)
    source_ids, source_scores = search(index, query_embeddings[0:1], k)
    followup_ids, followup_scores = search(index, query_embeddings[1:2], k)

    source_rank = rank_of_doc(source_ids, documents, case.expected_doc_id)
    followup_rank = rank_of_doc(followup_ids, documents, case.expected_doc_id)
    top_overlap = len(set(source_ids).intersection(followup_ids)) / k

    source_expected_score = source_scores[source_rank - 1] if source_rank is not None else None
    followup_expected_score = followup_scores[followup_rank - 1] if followup_rank is not None else None
    negative_expected_score = None

    if case.mr_id == "MR-Similar":
        violated = followup_rank is None or top_overlap < 0.4
    elif case.mr_id == "MR-Irrelevant-Noise":
        violated = followup_rank is None or (
            source_rank is not None and followup_rank > source_rank + 2
        )
    elif case.mr_id == "MR-Contradictory":
        negative_ids, negative_scores = search(index, query_embeddings[2:3], k)
        negative_rank = rank_of_doc(negative_ids, documents, case.expected_doc_id)
        negative_expected_score = negative_scores[negative_rank - 1] if negative_rank is not None else None
        violated = (
            source_expected_score is not None
            and followup_expected_score is not None
            and negative_expected_score is not None
            and negative_expected_score >= min(source_expected_score, followup_expected_score)
        )
    else:
        raise ValueError(f"Unsupported MR: {case.mr_id}")

    return {
        "case_id": case.case_id,
        "mr_id": case.mr_id,
        "expected_doc_id": case.expected_doc_id,
        "source_rank": source_rank,
        "followup_rank": followup_rank,
        "top_overlap": round(top_overlap, 3),
        "source_expected_score": round(source_expected_score, 4) if source_expected_score is not None else None,
        "followup_expected_score": round(followup_expected_score, 4) if followup_expected_score is not None else None,
        "negative_expected_score": round(negative_expected_score, 4) if negative_expected_score is not None else None,
        "violated": violated,
        "source_top1": documents[source_ids[0]].doc_id,
        "followup_top1": documents[followup_ids[0]].doc_id,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"model: {model_name}")
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(model_name, device=device)
    doc_embeddings = encode(model, [doc.text for doc in DOCUMENTS])
    index = build_index(doc_embeddings)

    rows = [evaluate_case(case, model, index, DOCUMENTS, k=3) for case in TEST_CASES]
    results = pd.DataFrame(rows)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "metmap_mini_results.csv"
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = (
        results.groupby("mr_id")["violated"]
        .agg(total="count", violations="sum")
        .reset_index()
    )
    summary["violation_rate"] = (summary["violations"] / summary["total"]).round(3)

    print("\nsummary:")
    print(summary.to_string(index=False))
    print(f"\nresults saved to: {output_path}")


if __name__ == "__main__":
    main()
