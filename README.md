# MeTMaP 复现实验项目

本仓库用于保存“工业软件分析与测试”课程大作业的复现实验材料，复现对象为论文：

**MeTMaP: Metamorphic Testing for Detecting False Vector Matching Problems in LLM Augmented Generation**

本项目复现了 MeTMaP 方法的核心蜕变测试流程，用于检测 LLM 增强生成系统中向量检索模块的 false vector matching 问题。实验流程包括构造 `base-positive-negative` 三元组，使用 sentence-transformers 模型进行文本向量编码，计算多种距离度量，并检查如下蜕变关系是否成立：

```text
distance(base, positive) < distance(base, negative)
```

如果 `negative sentence` 在向量空间中比 `positive sentence` 更接近 `base sentence`，则认为当前模型与距离度量配置在该测试用例上发生了一次 false vector matching。

## 实验环境

- 操作系统：Windows 10/11
- Python 版本：3.11
- GPU：NVIDIA GeForce RTX 3080 Ti Laptop GPU
- PyTorch：支持 CUDA 的 PyTorch 版本

安装依赖：

```powershell
pip install -r requirements.txt
```

## 项目结构

```text
scripts/
  metmap_paper_style.py          主实验脚本，完成数据构造、向量编码、距离计算和蜕变关系检查
  analyze_metmap_results.py      结果分析脚本，用于生成汇总表和可视化图
  metmap_reproduction_mini.py    小规模验证脚本
  smoke_test_gpu_retrieval.py    GPU 与向量检索流程验证脚本

outputs/
  metmap_paper_style_results.csv 逐条实验结果
  metmap_failure_cases.csv       false vector matching 失败案例
  metmap_by_config.csv           按模型-距离配置汇总的结果
  metmap_by_dataset_mr.csv       按数据集和蜕变关系类型汇总的结果
  metmap_accuracy_heatmap.png    模型-距离配置准确率热力图
  metmap_top_failure_configs_en.png
                                  高违反率模型-距离配置统计图
  metmap_violations_by_dataset_en.png
                                  不同数据集违反次数统计图

requirements.txt                 Python 依赖列表
```

## 复现实验规模

本地复现实验使用的规模如下：

- 2500 个三元组测试用例
- 13 个 embedding 模型
- 7 种距离度量
- 91 种模型-距离配置
- 227500 次蜕变关系检查

## 主要结果文件

- `outputs/metmap_paper_style_results.csv`：保存每一次模型、距离度量、数据集和三元组组合下的详细测试结果。
- `outputs/metmap_failure_cases.csv`：保存违反蜕变关系的 false vector matching 案例。
- `outputs/metmap_by_dataset_mr.csv`：保存按数据集和 MR 类型聚合后的统计结果。
- `outputs/metmap_accuracy_heatmap.png`：展示不同 embedding 模型和距离度量组合下的准确率热力图。
- `outputs/metmap_top_failure_configs_en.png`：展示违反次数较高的模型-距离配置。
- `outputs/metmap_violations_by_dataset_en.png`：展示不同数据集上的违反次数差异。

## 说明

原论文使用了更大规模的硬件资源和实验配置。受本地计算资源限制，本仓库主要复现 MeTMaP 的核心测试思想、实验流程、蜕变关系、距离度量和 false vector matching 的定性现象，用于课程大作业中的方法复现与结果分析。
