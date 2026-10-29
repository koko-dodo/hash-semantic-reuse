# Hash-based Semantic Reuse System

算法设计与分析课程 Lab：使用 **Exact Hash + Random Hyperplane LSH + Cosine Similarity** 检索并复用历史 Prompt。项目已完成算法实现、自动化测试、参数研究、Baseline、内存及规模扩展实验。

重点配置在 10,000 条历史记录上达到 **97.9425% Macro Recall**，平均只验证 **37.165** 个候选。当前 Python 实现的语义查询仍慢于 NumPy Brute-force；减少最终余弦验证次数并不保证更低的实际延迟。

[实验报告 PDF（4 页 A4）](report/report.pdf) · [报告 Markdown](report/report.md) · [Baseline 结果](results/baseline_comparison.csv)

## 检索流程

1. 接收原始 Query 文本，首先查询 Exact Hash。
2. 精确命中时返回一个历史 Prompt ID，跳过 Embedding 和 LSH。
3. 未命中时使用预计算 Query Embedding，或按需调用 Embedding Model。
4. 合并并去重 L 张 LSH 表的候选，只对候选计算余弦相似度，返回达到阈值的历史 ID。

统一入口为 [`SemanticReuseRetriever`](src/retrieval.py)。返回状态为 `exact`、`semantic` 或 `no_match`，并提供 ID、语义相似度、候选数、最终余弦验证次数及路径元数据。Exact Hit 是单个复用结果，不等同于返回全部阈值语义近邻；Semantic 路径返回候选中全部通过验证的近邻。

## 数据与实验设置

| 项目 | 设置 |
|---|---|
| Python | 3.12 |
| 数据集 | `sentence-transformers/quora-duplicates`，`pair-class`，`train` |
| 历史库 / 测试查询 | 10,000 / 200 |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2`，CPU |
| 向量 | 384 维、float32、L2 归一化 |
| Cosine Threshold | 0.8 |
| 重点 LSH 配置 | k=16，L=80，seed=42 |
| 参数研究 | k ∈ {12,16,20}，L ∈ {40,80,120}，共 9 组 |
| Scalability | N ∈ {2000,5000,10000} |

历史文本和查询各自唯一，两个集合没有相同字符串。每条查询来自一个人工标注的 Duplicate Pair，另一条问题被放入历史库。

**人工 Duplicate Label 不等于 Ground Truth。** Ground Truth 通过全部历史向量与查询向量的余弦相似度 ≥ 0.8 定义。当前 200 条查询中，173 条具有阈值近邻，27 条 Ground Truth 为空。

## 安装与复现

以下命令均在仓库根目录运行，使用模块方式执行脚本。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 保存了实验环境的完整依赖版本，包含传递依赖。首次准备数据和 Embedding 时需要访问 Hugging Face，下载数据集及模型；生成的本地数据和模型缓存不随仓库分发。

### 1. 准备数据、Embedding 和 Ground Truth

```bash
python -m src.dataset
python -m src.embedding
python -m src.brute_force
```

依次生成 `data/prompts.csv`、`data/queries.csv`、对应的两个 Embedding `.npy` 文件和 `data/ground_truth.npy`。文本 ID 对应 Embedding 行索引；重新生成时应按此顺序同步更新，避免混用不同批次的文件。

### 2. 运行实验

```bash
python -m experiments.parameter_study
python -m experiments.memory_study
python -m experiments.scalability
python -m experiments.baseline_comparison
```

| 实验 | 输出 |
|---|---|
| Parameter Study | `results/parameter_study.csv` |
| Memory Study | 终端输出 Embedding、超平面、tracked allocation 和 Bucket Entries |
| Scalability | `results/scalability.csv` |
| Baseline | `results/baseline_comparison.csv`、`baseline_exact_hits.csv`、`baseline_rounds.csv` |

Baseline 分开测量 200 条无精确命中的语义查询和 100 条已知精确命中查询。所有方法完整预热一次，重复测量 5 轮，轮换方法顺序，保存各轮均值、中位数、标准差和 P95；汇总表另含各轮均值的标准差。

### 3. 生成图表

```bash
python -m experiments.visualization
```

根据当前 CSV 更新 `results/figures/` 下的参数权衡、候选规模和 Baseline 时延图。上述数据准备、实验和绘图命令会生成或覆盖各自的本地产物。

## 使用统一检索器

### 使用预计算向量

```python
import numpy as np
import pandas as pd

from src.retrieval import SemanticReuseRetriever

prompt_texts = pd.read_csv("data/prompts.csv")["text"].tolist()
prompt_vectors = np.load("data/prompt_embeddings.npy")
retriever = SemanticReuseRetriever(
    prompt_texts, prompt_vectors, threshold=0.8, k=16, L=80, seed=42
)

# ID=0 也是有效命中；此路径不需要 Query Embedding。
exact = retriever.search(prompt_texts[0])
assert exact.status == "exact" and exact.prompt_ids == (0,)

query_texts = pd.read_csv("data/queries.csv")["text"].tolist()
query_vectors = np.load("data/query_embeddings.npy")
result = retriever.search(query_texts[0], query_embedding=query_vectors[0])
print(result.status, result.prompt_ids, result.candidate_count)
```

历史向量必须与 `prompt_texts` 顺序一致，查询向量必须使用同一模型和维度。返回 ID 按历史行号排列，语义相似度与 ID 对齐；精确命中的 `similarities` 为 `None`。

### Exact Miss 后才加载和调用模型

可传入 `embed_query(text)` 回调。以下示例沿用上面的历史文本和向量，模型在第一次需要编码时才加载，之后复用。

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def encode_query(text):
    return get_model().encode(
        text, convert_to_numpy=True, normalize_embeddings=True
    )

online = SemanticReuseRetriever(
    prompt_texts, prompt_vectors, embed_query=encode_query, k=16, L=80, seed=42
)
result = online.search("How can I learn Python?")
```

Exact Miss 时，如果既没有预计算向量，也没有编码回调，接口会抛出 `ValueError`；这不属于 `no_match`。两个底层索引的 `build_index()` 均为全量重建，旧记录和旧桶会被清空，LSH 超平面保持不变。

## 实验结果与指标

### Baseline

以下为已保存的本机测量结果；延迟会随硬件、系统负载和数值库变化。

| 方法 | Macro Recall | 平均候选 / 最终余弦验证数 | Candidate Ratio | 平均延迟 | 中位延迟 |
|---|---:|---:|---:|---:|---:|
| Brute-force | 100% | 10,000 | 100% | 75.889 μs | 73.708 μs |
| Exact Hash Only | 0% | 0 | 0% | 0.229 μs | 0.208 μs |
| Hash + LSH | 97.9425% | 37.165 | 0.37165% | 390.469 μs | 387.708 μs |

Hash + LSH 的 Micro Recall 为 **98.0769%**，最终余弦验证数减少 **99.62835%**，但平均延迟约为 Brute-force 的 **5.15 倍**。额外 100 条精确查询全部返回正确 ID，统一 Retriever 平均耗时约 **0.802 μs**，未调用 Embedding 或 LSH。语义测试中 Exact Hash 的 0% Recall 是文本无交集导致的预期结果。

**计时范围：**三种方法使用预加载文本和预计算向量，包含各自检索、验证及返回结果构造；Hash + LSH 包含 Exact Miss 后的完整语义路径，以及 Retriever 自身的验证、归一化和元数据构造。排除模型加载、Embedding 生成、索引构建、预热和指标汇总。精确命中路径单独报告准确率与时延。

### 指标口径与参数依据

- **Macro Recall：**仅对 Ground Truth 非空的查询计算 Recall 后取平均；CSV 中的 `mean_recall` 使用此定义。
- **Micro Recall：**全部命中的真实邻居关系数除以全部 Ground Truth 邻居关系数。
- **Candidate Ratio：**去重候选数除以当前数据库大小 N，再对全部查询取平均。
- **False Candidates：**候选集中不属于当前 Ground Truth 的历史记录数，不等于最终返回的错误结果数。
- **Cosine Computations：**最终历史向量验证次数，不包含 kL 次随机超平面投影。

随机超平面的单 bit 碰撞概率及多表候选召回概率为：

$$
p(s)=1-\frac{\arccos(s)}{\pi},\qquad P(s)=1-(1-p(s)^k)^L.
$$

增大 k 通常减少候选并降低召回，增大 L 通常提高召回并增加计算和内存开销。k=16、L=80 是偏重候选验证数量的折中配置，不是所有指标上的最优配置；边界相似度的理论碰撞概率也不等于测试集的整体 Recall。

### Scalability 与 Memory

| 历史库大小 N | 有效 Recall 查询数 | Macro Recall | 平均候选数 | 平均 False Candidates |
|---:|---:|---:|---:|---:|
| 2,000 | 163 | 98.42% | 8.125 | 7.085 |
| 5,000 | 166 | 98.57% | 19.125 | 17.795 |
| 10,000 | 173 | 97.94% | 37.165 | 35.380 |

历史 Embedding 缓冲区约 **14.65 MiB**；k=16、L=80 的 tracked index allocation 约 **121.89 MiB**，L=120 时约 **182.79 MiB**。该测量包含超平面及可追踪的索引分配，不包含追踪前加载的 Embedding，不等于进程 RSS；超平面字节数不能再次叠加。

参数与规模实验直接测量 LSH 候选生成和验证，Baseline 使用完整 Retriever，二者的延迟不应直接混用。规模实验为每个 N 重新计算 Ground Truth，但采用包含固定配对记录的嵌套前缀，有效查询集合也随 N 改变。查询来自重复问题对、模型与数据 revision 未在源码固定，因此结论有分布和复现范围限制，不能推广为 LSH 在任意规模下都更快。

## 自动化测试

```bash
python -m unittest discover -s tests -v
```

共 31 项测试，覆盖 Exact / Semantic / No Match、ID=0、延迟编码、候选阈值验证、索引重复及空库重建、超平面不变性，以及 Baseline 的预热、重复测量和指标计算。测试使用小型合成向量与标准库 `unittest`，不下载模型、不依赖本地实验数据，也不重写实验结果。

## 项目结构

```text
src/           数据处理、Embedding、Exact Hash、LSH、统一检索及评估
experiments/   参数、内存、规模、Baseline 与可视化脚本
tests/         自动化回归测试
data/          本地生成的文本、向量和 Ground Truth（不随 Git 分发）
results/       已保存的实验 CSV 与 figures/ 图表
report/        实验报告 Markdown 和 4 页 A4 PDF
requirements.txt  实验环境依赖版本
```

当前已提交的实验结果可直接查看；`.gitignore` 仍忽略新生成的数据和结果文件，现有已跟踪结果不受影响。`src/evaluation.py` 和 `src/lsh.py` 的独立演示默认使用 L=120，正式重点配置以统一 Retriever 和 Baseline 的 k=16、L=80 为准。

如需从 Markdown 重新导出报告，需要额外安装 Pandoc、XeLaTeX，以及报告使用的 Songti SC、Times New Roman、Arial、Menlo 字体：

```bash
pandoc report/report.md --from=markdown --pdf-engine=xelatex --resource-path=report --output=report/report.pdf
```
