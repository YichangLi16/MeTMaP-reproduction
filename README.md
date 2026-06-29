# MeTMaP Reproduction for Industrial Software Analysis and Testing

This repository contains a course reproduction project for:

**MeTMaP: Metamorphic Testing for Detecting False Vector Matching Problems in LLM Augmented Generation**

The project reproduces the core metamorphic-testing workflow for detecting false vector matching in embedding-based retrieval modules. The reproduced workflow builds base-positive-negative text triplets, encodes them with sentence-transformer embedding models, computes multiple distance metrics, and checks whether the expected metamorphic relation holds:

```text
distance(base, positive) < distance(base, negative)
```

If the negative sentence is closer than the positive sentence, the configuration is treated as a false vector matching case.

## Environment

- OS: Windows 10/11
- Python: 3.11
- GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU
- PyTorch: CUDA-enabled PyTorch

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Project Structure

```text
scripts/
  metmap_paper_style.py          Main experiment script
  analyze_metmap_results.py      Result aggregation and visualization

outputs/
  metmap_paper_style_results.csv Detailed test results
  metmap_failure_cases.csv       False vector matching cases
  metmap_by_config.csv           Summary by model-distance configuration
  metmap_by_dataset_mr.csv       Summary by dataset and metamorphic relation
  metmap_accuracy_heatmap.png    Accuracy heatmap
  metmap_top_failure_configs_en.png
                                  Top failure-prone configurations
  metmap_violations_by_dataset_en.png
                                  Violation counts by dataset

requirements.txt                 Python dependencies
environment_setup.md             Environment setup notes
```

## Reproduction Scale

The local reproduction uses:

- 2500 triplets
- 13 embedding models
- 7 distance metrics
- 91 model-distance configurations
- 227500 metamorphic-relation checks

## Main Result Files

- `outputs/metmap_paper_style_results.csv`: row-level results for each model, distance metric, dataset and triplet.
- `outputs/metmap_failure_cases.csv`: selected violation cases where the metamorphic relation fails.
- `outputs/metmap_by_dataset_mr.csv`: aggregated results by dataset and MR type.
- `outputs/metmap_accuracy_heatmap.png`: model-distance accuracy heatmap.

## Notes

The original paper used larger-scale hardware and a broader experimental configuration. This repository focuses on a high-similarity local reproduction of the core testing idea, experimental workflow, metamorphic relation, distance metrics and qualitative false-vector-matching behavior.
