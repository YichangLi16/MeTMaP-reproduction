# 基于蜕变测试的 RAG 向量匹配错误检测方法复现与分析

## 一、背景介绍

近年来，大语言模型增强生成系统（LLM-augmented generation）在工业软件中被广泛用于智能问答、知识库检索、代码助手、运维诊断和企业文档分析等场景。典型系统通常采用 RAG（Retrieval-Augmented Generation）架构：首先将用户查询和知识库文档编码为向量，然后通过向量距离或相似度度量检索相关文档，最后将检索结果作为上下文输入大语言模型生成回答。该流程可以降低模型幻觉、增强回答的事实依据，但也引入了新的可靠性问题：如果向量匹配阶段错误地把不相关文本判断为相似，后续生成模型就可能基于错误上下文生成不可靠答案。

本文复现的工作为 Wang 等人在 FORGE 2024 发表的论文 *MeTMaP: Metamorphic Testing for Detecting False Vector Matching Problems in LLM Augmented Generation*。该论文针对 LLM 增强生成系统中的 false vector matching 问题提出 MeTMaP 框架，使用蜕变测试思想构造三元组测试用例，检测 embedding 模型与距离度量组合是否会把负样本错误匹配到查询文本。论文指出，向量匹配错误会影响 RAG、CAG 等系统的上下文选择，是当前工业智能软件测试中的重要问题。

该工作与课程内容具有直接关联。课程 3.9 与 3.10 介绍了蜕变测试的基本流程：构建蜕变关系、生成源测试用例、执行源测试用例和后续测试用例，并检查输出是否违反预期关系。MeTMaP 中的 base sentence 可视为源测试用例，positive/negative sentence 可视为根据语义关系构造出的后续输入。若被测向量匹配方法使 base 与 negative 的距离小于或等于 base 与 positive 的距离，则说明输出违反了“语义相关文本应比无关文本更接近”的蜕变关系，进而揭示潜在的软件缺陷。

## 二、复现环境搭建

原论文实验环境为 Ubuntu 20.04.3 LTS，硬件包含 4 张 NVIDIA V100 32GB GPU。受本地硬件条件限制，本文采用单机 Windows 环境进行高相似度复现，重点保持测试思想、输入结构、距离度量和评价指标与论文一致，并在可承受范围内扩大测试规模。

本文复现实验环境如下：

| 类别 | 配置 |
|---|---|
| 操作系统 | Windows 10/11 |
| Python | Python 3.11 虚拟环境 |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU，16GB 显存 |
| CUDA/PyTorch | PyTorch 2.6.0 + cu124 |
| 核心依赖 | sentence-transformers 3.3.1、datasets 3.2.0、numpy 2.1.3、pandas 2.2.3、scikit-learn 1.5.2、matplotlib 3.9.3、seaborn 0.13.2 |
| 实验脚本 | `scripts/metmap_paper_style.py`、`scripts/analyze_metmap_results.py` |

环境搭建步骤如下：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

GPU 验证命令如下：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

验证结果表明，PyTorch 可以正常识别 CUDA 设备，后续 embedding 生成阶段使用 GPU 加速。Windows 环境下 FAISS GPU 版本安装复杂，因此本文使用 PyTorch 在 GPU 上生成向量，并使用 Python/NumPy 完成向量距离计算。

## 三、数据复现

MeTMaP 原论文使用 8 种 metamorphic relations（MR）构造 40,000 个 triplets，每个 triplet 由 base、positive、negative 三个文本组成。原论文涉及的数据集包括 Stanford Contradiction Corpora、PAWS、VitaminC、Inference-is-Everything、HEROS、NEVIR 等，并使用大语言模型辅助生成部分正负样本。由于论文匿名 artifact 仓库当前无法稳定访问，本文未能直接获取官方 40,000 个 triplets，因此采用“同构数据复现”的方式：保持论文三元组结构和判断规则不变，使用公开 NLP 数据集构造可复现的 base-positive-negative 测试样本。

本文使用的数据来源如下：

| 数据来源 | 用途 | 构造方式 |
|---|---|---|
| GLUE STS-B | 语义相似度关系 | 高相似句对作为 base-positive，低相似句作为 negative |
| GLUE MRPC | 释义关系 | 同义/释义句对作为 base-positive，非释义句作为 negative |
| GLUE QQP | 重复问题关系 | 重复问题作为 base-positive，非重复问题作为 negative |
| SNLI | 蕴含与矛盾关系 | premise 与 entailment hypothesis 作为 base-positive，contradiction hypothesis 作为 negative |
| Synthetic | 领域知识关系 | 手工构造 RAG、embedding、蜕变测试相关样本，用于验证流程稳定性 |

为了增加测试难度，本文没有随机选择 negative，而是采用词面重叠度筛选 hard negative：对于每个 base，从候选负样本中选择词面重叠较高但标签为不相关或矛盾的句子。该设计更容易触发 false vector matching，也更接近 MeTMaP 关注的“语义上应区分但向量空间中可能混淆”的问题。

最终本文构造得到 928 个有效 triplets。由于 STS-B 等数据集在严格筛选后可用高质量样本数量有限，最终数量略低于 5 个数据源 × 每源 200 条的目标规模。每个 triplet 会在 6 个 embedding 模型和 7 种距离度量下分别测试，因此总测试判断次数为：

```text
928 triplets × 6 models × 7 distances = 38,976 tests
```

## 四、方法复现

### 4.1 MeTMaP 核心思想

MeTMaP 的核心思想是把向量匹配方法视为被测程序，将文本语义关系转化为蜕变关系。对于任意一个三元组 `(base, positive, negative)`，如果 positive 与 base 在语义上更相关，而 negative 与 base 在语义上不相关或相反，那么一个合理的向量匹配方法应满足：

```text
distance(base, positive) < distance(base, negative)
```

若实验中出现：

```text
distance(base, negative) <= distance(base, positive)
```

则说明向量匹配方法将负样本错误地判断为更接近 base，记为一次 false vector matching，也即蜕变关系违反。

### 4.2 复现流程

本文复现流程包括以下步骤：

1. 从公开数据集中构造 base-positive-negative 三元组。
2. 加载 embedding 模型，将三元组文本编码为向量。
3. 对 base-positive 和 base-negative 分别计算距离。
4. 判断 negative distance 是否小于或等于 positive distance。
5. 对每个 embedding 模型和距离度量配置统计准确率与违反率。
6. 输出详细结果、按配置汇总结果、按数据集/MR 汇总结果和典型失败案例。

核心判断逻辑如下：

```python
positive_distance = distance(base, positive)
negative_distance = distance(base, negative)
violated = negative_distance <= positive_distance
```

### 4.3 模型与距离度量

本文当前复现实验使用 6 个 embedding 模型：

| 模型 |
|---|
| sentence-transformers/all-MiniLM-L6-v2 |
| sentence-transformers/all-mpnet-base-v2 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 |
| sentence-transformers/paraphrase-MiniLM-L6-v2 |
| intfloat/e5-small-v2 |
| BAAI/bge-small-en-v1.5 |

距离度量尽量对齐 MeTMaP 原论文的 7 种设置：

| 距离度量 | 说明 |
|---|---|
| Cosine Distance | 基于向量夹角衡量语义距离 |
| Euclidean Distance | 欧氏距离 |
| Mahalanobis Distance | 基于协方差矩阵的马氏距离 |
| Bray-Curtis Distance | 常用于向量差异度量 |
| Lance-Williams Distance | 基于绝对差异的归一化距离 |
| Pearson Correlation Distance | 基于相关系数的距离 |
| Manhattan Distance | 曼哈顿距离 |

### 4.4 与原论文的差异

本文复现尽量保持方法流程与论文一致，但与原论文仍存在以下差异：

| 项目 | 原论文 | 本文复现 |
|---|---|---|
| 硬件 | 4 × NVIDIA V100 32GB | 1 × RTX 3080 Ti Laptop 16GB |
| triplet 数量 | 40,000 | 928 |
| embedding 模型 | 29 个 | 6 个 |
| 距离度量 | 7 种 | 7 种，已尽量对齐 |
| 配置数量 | 203 | 42 |
| 数据来源 | 论文 6 个原始数据集与 LLM 生成样本 | 公开 NLP 数据集与少量 synthetic 样本 |
| 官方 artifact | 可用性受限 | 当前未能直接访问匿名仓库 |

因此，本文不声称完成 1:1 全量复现，而是完成了 MeTMaP 核心测试机制的高相似度复现。

## 五、结果对比

### 5.1 总体结果

本文主实验共执行 38,976 次向量匹配判断。结果表明，不同 embedding 模型和距离度量配置均可能出现 false vector matching，说明 MeTMaP 所关注的问题在本地复现实验中同样存在。

在 42 个配置中，表现最好的配置为：

| 模型 | 距离度量 | 测试数 | 违反数 | 准确率 |
|---|---|---:|---:|---:|
| sentence-transformers/all-mpnet-base-v2 | euclidean | 928 | 23 | 0.9752 |
| sentence-transformers/all-mpnet-base-v2 | cosine_distance | 928 | 23 | 0.9752 |
| sentence-transformers/all-mpnet-base-v2 | pearson_correlation | 928 | 23 | 0.9752 |

表现最差的配置为：

| 模型 | 距离度量 | 测试数 | 违反数 | 准确率 |
|---|---|---:|---:|---:|
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | mahalanobis | 928 | 66 | 0.9289 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | lance_williams | 928 | 57 | 0.9386 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | bray_curtis | 928 | 57 | 0.9386 |

实验结果说明，即使在缩小规模的复现环境中，模型与距离度量的选择仍然会显著影响 false vector matching 的发生频率。这与原论文“不同向量匹配配置可靠性差异明显”的结论一致。

### 5.2 按数据集分析

按数据集统计，SNLI 产生的违反数量最多，违反率约为 12.81%；STS-B 次之，违反率约为 2.96%；MRPC 约为 1.80%；QQP 约为 0.39%；synthetic 样本未观察到违反。

| 数据集 | 测试判断数 | 违反数 | 违反率 |
|---|---:|---:|---:|
| SNLI | 8,400 | 1,076 | 0.1281 |
| GLUE STS-B | 5,376 | 159 | 0.0296 |
| GLUE MRPC | 8,400 | 151 | 0.0180 |
| GLUE QQP | 8,400 | 33 | 0.0039 |
| Synthetic | 8,400 | 0 | 0.0000 |

SNLI 的违反率较高，说明蕴含与矛盾关系在 embedding 空间中更容易被混淆。该现象与 RAG 系统实际风险相关：如果系统只依赖向量距离，可能会把词面相似但语义相反的文本作为相关上下文，从而影响后续生成结果。

### 5.3 与原论文结果的比较

MeTMaP 原论文报告了 203 种向量匹配配置，并指出最高准确率仅为 41.51%，强调 false vector matching 在大量 embedding 配置中普遍存在。本文复现实验的准确率整体高于原论文，最佳配置约为 97.52%，最差配置约为 92.89%。造成差异的可能原因包括：

1. 原论文使用 40,000 个 triplets，覆盖 8 类 MR 和更复杂的语义变换，测试难度更高。
2. 本文未能直接获取官方 artifact，使用公开数据集重构三元组，负样本复杂度与论文可能不同。
3. 原论文覆盖 29 个 embedding 模型和更多大模型、量化模型，本文当前仅测试 6 个模型。
4. 本文 synthetic 样本较简单，未产生违反，拉高了整体准确率。
5. 原论文涉及部分 LLM 生成样本，可能包含更隐蔽的语义扰动。

尽管数值结果存在差异，但本文成功复现了核心现象：同一组三元组在不同 embedding 模型和距离度量下会出现不同程度的 false vector matching，说明 MeTMaP 的蜕变测试框架能够用于发现 LLM 增强生成系统中的向量检索可靠性问题。

## 六、经验总结

本次复现过程中遇到的主要问题包括环境兼容、官方 artifact 获取、硬件资源差异和数据构造差异。

首先，Python 3.14 对部分机器学习库支持不稳定，因此最终选择 Python 3.11 构建虚拟环境。PyTorch 使用 CUDA 12.4 对应版本，GPU 可以正常用于 embedding 生成。Windows 环境下 FAISS GPU 安装复杂，因此本文未使用 FAISS GPU，而是直接用 NumPy 实现论文式 pairwise distance 判断。

其次，原论文匿名 artifact 仓库当前无法稳定访问，导致无法直接读取官方 40,000 个 triplets。因此本文采用公开数据集构造同构三元组，并明确将实验定位为“核心方法高相似度复现”，而非官方数据 1:1 复现。

再次，原论文硬件资源为 4 张 V100 32GB，而本文使用单张 RTX 3080 Ti Laptop 16GB。受资源限制，当前实验选择 6 个 embedding 模型和 928 个有效 triplets。但实验仍执行了 38,976 次判断，能够观察到稳定的 false vector matching 现象。

最后，本文发现 hard negative 的构造方式对结果影响明显。若随机选择 negative，实验结果容易过于理想；使用词面重叠度较高的负样本后，false vector matching 现象更加明显。这说明在工业 RAG 系统测试中，测试数据生成策略会显著影响缺陷暴露能力。

后续工作可从三个方向扩展：第一，继续增加 embedding 模型数量，向原论文 29 个模型靠近；第二，补充 PAWS、VitaminC、NEVIR 等论文原始数据集；第三，增加真实向量数据库实验，如 Chroma、Annoy 或 ScaNN，以进一步复现论文中的 vector database case study。

## 参考文献

[1] Guanyu Wang, Yuekang Li, Yi Liu, Gelei Deng, Tianlin Li, Guosheng Xu, Yang Liu, Haoyu Wang, Kailong Wang. MeTMaP: Metamorphic Testing for Detecting False Vector Matching Problems in LLM Augmented Generation. FORGE 2024. https://arxiv.org/abs/2402.14480

[2] FORGE 2024 Research Track: MeTMaP: Metamorphic Testing for Detecting False Vector Matching Problems in LLM Augmented Generation. https://conf.researchr.org/details/forge-2024/forge-2024-papers/10/MeTMaP-Metamorphic-Testing-for-Detecting-False-Vector-Matching-Problems-in-LLM-Augme

