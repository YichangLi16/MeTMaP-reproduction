from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATASET_SPLIT = "validation"


@dataclass(frozen=True)
class Triplet:
    dataset: str
    mr_id: str
    base: str
    positive: str
    negative: str


MODEL_PRESETS = {
    "quick": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
    "expanded": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "intfloat/e5-small-v2",
        "BAAI/bge-small-en-v1.5",
    ],
    "near_paper": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-mpnet-base-v2",
        "sentence-transformers/msmarco-MiniLM-L6-cos-v5",
        "sentence-transformers/gtr-t5-base",
        "intfloat/e5-small-v2",
        "thenlper/gte-small",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ],
}


def normalize_rows(rows: list[dict], limit: int) -> list[dict]:
    random.shuffle(rows)
    return rows[:limit]


def token_set(text: str) -> set[str]:
    return {token.strip(".,!?;:'\"()[]{}").lower() for token in text.split() if token.strip()}


def jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def hardest_by_overlap(base: str, candidates: list[str], sample_size: int = 256) -> str:
    pool = random.sample(candidates, min(sample_size, len(candidates)))
    return max(pool, key=lambda candidate: jaccard(base, candidate))


def build_glue_stsb(limit: int) -> list[Triplet]:
    dataset = load_dataset("glue", "stsb", split=DATASET_SPLIT)
    high = [row for row in dataset if row["label"] >= 4.5]
    low_sentences = [row["sentence2"] for row in dataset if row["label"] <= 1.0]
    rows = normalize_rows(high, limit)
    return [
        Triplet("glue-stsb", "MR1-Semantic-Similarity", row["sentence1"], row["sentence2"], hardest_by_overlap(row["sentence1"], low_sentences))
        for row in rows
        if row["sentence1"] and row["sentence2"]
    ]


def build_glue_mrpc(limit: int) -> list[Triplet]:
    dataset = load_dataset("glue", "mrpc", split=DATASET_SPLIT)
    positives = [row for row in dataset if row["label"] == 1]
    negative_sentences = [row["sentence2"] for row in dataset if row["label"] == 0]
    rows = normalize_rows(positives, limit)
    return [
        Triplet("glue-mrpc", "MR2-Paraphrase", row["sentence1"], row["sentence2"], hardest_by_overlap(row["sentence1"], negative_sentences))
        for row in rows
        if row["sentence1"] and row["sentence2"]
    ]


def build_glue_qqp(limit: int) -> list[Triplet]:
    dataset = load_dataset("glue", "qqp", split=DATASET_SPLIT)
    positives = [row for row in dataset if row["label"] == 1 and row["question1"] and row["question2"]]
    negative_questions = [row["question2"] for row in dataset if row["label"] == 0 and row["question2"]]
    rows = normalize_rows(positives, limit)
    return [
        Triplet("glue-qqp", "MR3-Duplicate-Question", row["question1"], row["question2"], hardest_by_overlap(row["question1"], negative_questions))
        for row in rows
    ]


def build_snli(limit: int) -> list[Triplet]:
    dataset = load_dataset("snli", split=DATASET_SPLIT)
    by_premise: dict[str, dict[int, list[str]]] = {}
    for row in dataset:
        label = row["label"]
        if label not in (0, 2):
            continue
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        if not premise or not hypothesis:
            continue
        by_premise.setdefault(premise, {0: [], 2: []})[label].append(hypothesis)

    triplets = []
    for premise, hypotheses in by_premise.items():
        if hypotheses[0] and hypotheses[2]:
            triplets.append(
                Triplet(
                    "snli",
                    "MR4-Entailment-vs-Contradiction",
                    premise,
                    random.choice(hypotheses[0]),
                    random.choice(hypotheses[2]),
                )
            )
    random.shuffle(triplets)
    return triplets[:limit]


def build_synthetic_mrs(limit: int) -> list[Triplet]:
    seeds = [
        ("RAG systems use vector retrieval to find relevant external knowledge.", "Retrieval augmented generation finds related context using vector search.", "A recipe explains how to bake bread in an oven."),
        ("Metamorphic testing checks relations between source and follow-up outputs.", "Follow-up tests should preserve expected relations in metamorphic testing.", "A football match can end with a penalty shootout."),
        ("Embedding models map text into vectors for semantic matching.", "Sentence encoders represent meanings as dense numerical vectors.", "The mountain trail is closed during heavy snow."),
        ("False vector matching retrieves irrelevant documents for a query.", "Wrong vector matches can return unrelated context to the user query.", "The museum opens at ten o'clock on Sunday."),
    ]
    rows = []
    for idx in range(limit):
        base, positive, negative = seeds[idx % len(seeds)]
        rows.append(Triplet("synthetic", "MR5-Domain-Knowledge", base, positive, negative))
    return rows


DATASET_BUILDERS: dict[str, Callable[[int], list[Triplet]]] = {
    "stsb": build_glue_stsb,
    "mrpc": build_glue_mrpc,
    "qqp": build_glue_qqp,
    "snli": build_snli,
    "synthetic": build_synthetic_mrs,
}


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype("float32")


def safe_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = safe_norm(a) * safe_norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def bray_curtis_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sum(np.abs(a + b)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(a - b)) / denom)


def lance_williams_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sum(np.abs(a) + np.abs(b)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(a - b)) / denom)


def pearson_correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_distance(a - np.mean(a), b - np.mean(b))


BASE_DISTANCES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "cosine_distance": cosine_distance,
    "euclidean": lambda a, b: float(np.linalg.norm(a - b)),
    "manhattan": lambda a, b: float(np.sum(np.abs(a - b))),
    "bray_curtis": bray_curtis_distance,
    "lance_williams": lance_williams_distance,
    "pearson_correlation": pearson_correlation_distance,
}


def collect_triplets(dataset_names: list[str], per_dataset: int) -> list[Triplet]:
    triplets: list[Triplet] = []
    for name in dataset_names:
        builder = DATASET_BUILDERS[name]
        print(f"loading dataset: {name}")
        triplets.extend(builder(per_dataset))
    random.shuffle(triplets)
    return triplets


def run_model(
    model_name: str,
    triplets: list[Triplet],
    distance_names: list[str],
    batch_size: int,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nmodel: {model_name}")
    print(f"device: {device}")
    model = SentenceTransformer(model_name, device=device)

    texts: list[str] = []
    for triplet in triplets:
        texts.extend([triplet.base, triplet.positive, triplet.negative])
    embeddings = encode_texts(model, texts, batch_size=batch_size)
    covariance = np.cov(embeddings, rowvar=False)
    inverse_covariance = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * 1e-5)

    def mahalanobis_distance(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.sqrt(np.dot(np.dot(diff, inverse_covariance), diff.T)))

    distances: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        **BASE_DISTANCES,
        "mahalanobis": mahalanobis_distance,
    }

    rows = []
    for triplet_idx, triplet in enumerate(tqdm(triplets, desc="evaluating triplets")):
        base = embeddings[triplet_idx * 3]
        positive = embeddings[triplet_idx * 3 + 1]
        negative = embeddings[triplet_idx * 3 + 2]

        for distance_name in distance_names:
            distance = distances[distance_name]
            positive_distance = distance(base, positive)
            negative_distance = distance(base, negative)
            violated = negative_distance <= positive_distance
            rows.append(
                {
                    "model": model_name,
                    "distance": distance_name,
                    "dataset": triplet.dataset,
                    "mr_id": triplet.mr_id,
                    "positive_distance": round(positive_distance, 6),
                    "negative_distance": round(negative_distance, 6),
                    "violated": violated,
                    "base": triplet.base,
                    "positive": triplet.positive,
                    "negative": triplet.negative,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style MeTMaP reproduction.")
    parser.add_argument("--preset", choices=MODEL_PRESETS.keys(), default="quick")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=["stsb", "mrpc", "qqp", "snli", "synthetic"])
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--per-dataset", type=int, default=20)
    parser.add_argument("--distances", nargs="*", default=[*BASE_DISTANCES.keys(), "mahalanobis"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    global DATASET_SPLIT
    DATASET_SPLIT = args.dataset_split

    model_names = args.models if args.models else MODEL_PRESETS[args.preset]
    triplets = collect_triplets(args.datasets, args.per_dataset)
    print(f"triplets: {len(triplets)}")
    print(f"models: {len(model_names)}")
    print(f"distances: {len(args.distances)}")

    frames = []
    failures = []
    for model_name in model_names:
        try:
            frames.append(run_model(model_name, triplets, args.distances, args.batch_size))
        except Exception as exc:
            failures.append({"model": model_name, "error": repr(exc)})
            print(f"model failed: {model_name}")
            print(repr(exc))

    if not frames:
        raise RuntimeError("No model completed successfully.")

    results = pd.concat(frames, ignore_index=True)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    detail_path = output_dir / "metmap_paper_style_results.csv"
    summary_path = output_dir / "metmap_paper_style_summary.csv"
    results.to_csv(detail_path, index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "metmap_model_failures.csv", index=False, encoding="utf-8-sig")

    summary = (
        results.groupby(["model", "distance", "dataset", "mr_id"])["violated"]
        .agg(total="count", violations="sum")
        .reset_index()
    )
    summary["accuracy"] = ((summary["total"] - summary["violations"]) / summary["total"]).round(4)
    summary["violation_rate"] = (summary["violations"] / summary["total"]).round(4)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    overall = (
        results.groupby(["model", "distance"])["violated"]
        .agg(total="count", violations="sum")
        .reset_index()
    )
    overall["accuracy"] = ((overall["total"] - overall["violations"]) / overall["total"]).round(4)
    print("\noverall accuracy by configuration:")
    print(overall.sort_values("accuracy", ascending=False).to_string(index=False))
    print(f"\ndetail results: {detail_path}")
    print(f"summary results: {summary_path}")


if __name__ == "__main__":
    main()
