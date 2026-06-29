from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    output_dir = Path("outputs")
    results_path = output_dir / "metmap_paper_style_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    results = pd.read_csv(results_path)

    by_config = (
        results.groupby(["model", "distance"])["violated"]
        .agg(total="count", violations="sum")
        .reset_index()
    )
    by_config["accuracy"] = (by_config["total"] - by_config["violations"]) / by_config["total"]
    by_config["violation_rate"] = by_config["violations"] / by_config["total"]
    by_config.to_csv(output_dir / "metmap_by_config.csv", index=False, encoding="utf-8-sig")

    by_dataset = (
        results.groupby(["dataset", "mr_id", "model", "distance"])["violated"]
        .agg(total="count", violations="sum")
        .reset_index()
    )
    by_dataset["accuracy"] = (by_dataset["total"] - by_dataset["violations"]) / by_dataset["total"]
    by_dataset.to_csv(output_dir / "metmap_by_dataset_mr.csv", index=False, encoding="utf-8-sig")

    failures = results[results["violated"]].copy()
    failures.to_csv(output_dir / "metmap_failure_cases.csv", index=False, encoding="utf-8-sig")

    pivot = by_config.pivot(index="model", columns="distance", values="accuracy")
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title("MeTMaP-style Accuracy by Embedding Model and Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "metmap_accuracy_heatmap.png", dpi=180)
    plt.close()

    top_failures = (
        failures.groupby(["model", "distance"])
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="violations")
    )
    top_failures.to_csv(output_dir / "metmap_top_failure_configs.csv", index=False, encoding="utf-8-sig")

    print(f"config summary: {output_dir / 'metmap_by_config.csv'}")
    print(f"dataset/MR summary: {output_dir / 'metmap_by_dataset_mr.csv'}")
    print(f"failure cases: {output_dir / 'metmap_failure_cases.csv'}")
    print(f"heatmap: {output_dir / 'metmap_accuracy_heatmap.png'}")


if __name__ == "__main__":
    main()
