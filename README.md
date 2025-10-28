# UF-FLR: LP-Rounded Kruskal + Multi-Source Dijkstra for Community Detection

> **Author:** Chen-Hong 洪禎  
> **Affiliation:** National Chung Cheng University, Graduate Institute of Telecommunications and Communication  
> **License:** MIT  
> **Keywords:** Community Detection, LP Rounding, Kruskal, Dijkstra, Union-Find, Louvain, Leiden, Graph Optimization  

---

## 🧭 Overview

**UF-FLR (Union-Find LP-Rounded)** is a hybrid community detection framework that integrates  
**Kruskal’s Minimum Spanning Tree**, **Union-Find clustering**, **Linear Programming (LP) rounding**, and **multi-source Dijkstra** optimization.

The goal is to enhance the **Louvain** and **Leiden** algorithms by improving **global connectivity**, **semantic coherence**, and **runtime efficiency** while reducing excessive community fragmentation.

This repository provides:
- Reproducible experiment scripts (`run_experiment_csr_igraph.py`)
- Automated result recording (`summary_metrics.csv`)
- CLI configuration for batch experiments and safety execution

---

## ⚙️ Algorithmic Design

| Component | Description |
|------------|-------------|
| **Union-Find (Disjoint Set)** | Efficiently merges subgraphs with path compression to ensure near-constant amortized time. |
| **Kruskal MST Integration** | Builds the foundational connectivity structure to preserve global graph coherence. |
| **LP Rounding (P70 adaptive threshold)** | Rounds fractional LP relaxations into integral clusters, avoiding extreme splits. |
| **Multi-Source Dijkstra** | Measures internal flow cohesion and path-based centrality within merged clusters. |
| **Benchmark Comparison** | Evaluates UF-FLR against Louvain and Leiden on modularity, conductance, and efficiency metrics. |

---

## 🧪 Experimental Framework

The system was designed for **large-scale graph benchmarks** such as:
- DBLP collaboration networks
- Citation graphs
- Financial transaction networks

### Input Format
Plain-text edge list (unweighted or weighted):
```
node_u node_v [weight]
```

### Output
CSV file `summary_metrics.csv` containing:
```
algorithm, modularity, conductance, avg_path_length, semantic_cohesion, num_clusters, runtime_sec, memory_gb
```

---

## 🚀 Usage

### Command-Line Execution
```bash
python run_experiment_csr_igraph.py   --input ./data/com-dblp.ungraph.txt   --save_dir ./results   --repeat 3   --safety_mode   --tqdm   --memory_guard_gb 2
```

### Optional Arguments
| Flag | Description |
|------|--------------|
| `--input` | Path to input edge list file |
| `--save_dir` | Output directory for metrics |
| `--repeat` | Number of experiment repetitions |
| `--safety_mode` | Enable memory guard and process isolation |
| `--tqdm` | Display real-time progress bar |
| `--memory_guard_gb` | Maximum memory usage limit (in GB) |

---

## 📊 Example Results (to be updated)

| Algorithm | Modularity | Conductance | Avg. Path Length | Num. Clusters | Runtime (s) | Memory (GB) |
|------------|-------------|--------------|------------------|----------------|--------------|--------------|
| Louvain | — | — | — | — | — | — |
| Leiden | — | — | — | — | — | — |
| UF-FLR | — | — | — | — | — | — |

---

## 🧠 Research Context

This repository is part of the study:  
**“Improving Louvain and Leiden Community Detection via LP-Rounded Kruskal and Multi-Source Dijkstra”**,  
which explores how LP rounding can bridge the trade-off between **local modularity optimization** and **global connectivity preservation**.

It is also conceptually connected to **UF-FAE (Union-Find-based Feature-Augmented Embedding)**, extending its use from financial transaction clustering to structural community detection.

---

## 📈 Performance Notes

- Optimized for **Python 3.10+**
- Supports **igraph**, **NumPy**, **tqdm**, and **psutil**
- Designed for both CPU and parallel cluster environments
- Empirically shows improvement in **semantic cohesion** and **runtime efficiency**

---

## 🧩 Directory Structure

```
📦 UF-FLR
├── run_experiment_csr_igraph.py      # Main experiment script
├── results/
│   ├── summary_metrics.csv           # Recorded metrics
│   └── logs/                         # Optional log directory
├── data/
│   └── com-dblp.ungraph.txt          # Example dataset
└── README.md                         # Documentation
```

---

## 🔬 Future Extensions

- Integrate LP relaxation solver for adaptive rounding
- Implement dynamic graph update (incremental Union-Find)
- Extend to directed graph clustering and temporal community detection
- Combine with GNN-based embedding frameworks

---

## 🧾 Citation

If you use this repository, please cite as:

```bibtex
@misc{chenhong2025ufflr,
  author       = {Chen-Hong, Hung},
  title        = {UF-FLR: LP-Rounded Kruskal + Multi-Source Dijkstra for Community Detection},
  year         = {2025},
  howpublished = {\url{https://github.com/scuranger0625/UF-FLR}},
  note         = {Master Research Project, National Chung Cheng University}
}
```

---

## 📬 Contact

**Chen-Hong Hung (洪禎)**  
📧 Email: scuranger0625 [at] gmail [dot] com  
🏫 National Chung Cheng University, Taiwan  
🌐 GitHub: [scuranger0625](https://github.com/scuranger0625)

---

> “Balancing local modularity and global flow —  
> UF-FLR redefines community detection for connected intelligence.”

---

# 🇹🇼 中文版說明

## 🌐 專案簡介

**UF-FLR（Union-Find LP 捨入演算法）** 是一個融合了 **Kruskal 最小生成樹**、**Union-Find 分群結構**、**LP 線性規劃捨入** 與 **多源 Dijkstra 路徑估計** 的混合式社群偵測框架。

本研究旨在改良 **Louvain / Leiden** 社群偵測法，透過結構性與語意一致性提升，改善傳統方法過度分裂與模組化過高的問題。

---

## ⚙️ 核心設計模組

| 模組 | 功能說明 |
|------|----------|
| **Union-Find 集合合併** | 透過路徑壓縮加速群組合併，保證近常數時間複雜度。 |
| **Kruskal 最小生成樹** | 維持全域連通性與邊權重平衡，避免社群孤立。 |
| **LP Rounding（P70 門檻）** | 將分數解近似成整數群組，避免過度切割。 |
| **多源 Dijkstra** | 用於群內路徑分析與語意凝聚度測量。 |
| **對照基準** | 與 Louvain / Leiden 演算法比較模組度、導通度與運算效率。 |

---

## 📁 檔案結構

```
📦 UF-FLR
├── run_experiment_csr_igraph.py      # 主程式
├── results/
│   ├── summary_metrics.csv           # 實驗結果輸出
│   └── logs/                         # 日誌
├── data/
│   └── com-dblp.ungraph.txt          # 測試資料集
└── README.md                         # 本文件
```

---

## 🧩 使用方式

```bash
python run_experiment_csr_igraph.py   --input ./data/com-dblp.ungraph.txt   --save_dir ./results   --repeat 3   --safety_mode   --tqdm   --memory_guard_gb 2
```

支援參數：  
- `--input`：輸入檔路徑  
- `--save_dir`：輸出目錄  
- `--repeat`：重複實驗次數  
- `--safety_mode`：啟用記憶體防護  
- `--tqdm`：顯示進度條  
- `--memory_guard_gb`：最大記憶體限制  

---

## 📘 應用場景

- 大規模社群網路（如 DBLP、社交網）  
- 金融交易圖與詐欺偵測網絡  
- 語意關聯圖與主題聚合分析  

---

## 💡 延伸方向

- 動態圖（Dynamic Graph）增量分群  
- LP Relaxation 自動門檻調整  
- 結合 GNN 的社群表徵學習  
- 與 UF-FAE 架構結合以支援跨模態分析  

---

> 「UF-FLR 強調在全域結構與局部語意之間取得平衡，  
> 讓社群偵測不僅是分群，更是語意的重構。」
