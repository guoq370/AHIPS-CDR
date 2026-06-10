# AHIPS-CDR 

### Bridging Education and Employment: An Attention-Driven Cross-Domain Intelligent Recommendation System with Independent Parallel Strategy

---

## 📌 Abstract
**AHIPS-CDR** is an end-to-end heterogeneous graph learning framework designed to bridge the semantic gap between academic trajectory (Coursera) and vocational recruitment (LinkedIn). To mitigate the structural conflicts and feature cancellation inherent in cross-domain recommendation, AHIPS-CDR implements an **Independent Parallel Strategy (IPS)**, which decouples intra-domain behavior learning from inter-domain knowledge transfer. This repository provides the complete implementation of AHIPS-CDR, alongside comprehensive baseline comparisons (HGT, HAGO, MI-DTCDR, SEAGULL).

---

## 🛠️ Environment Prerequisites
The codebase is developed for high-performance research environments.
* **Python**: 3.10+
* **PyTorch**: 2.0+
* **DGL**: 1.1+ (for HIN operations)
* **Dependencies**: Install via `pip install -r requirements.txt`

---

## 📂 Project Structure
```text
AHIPS-CDR/
├── README.md                 # Project documentation
├── requirements.txt          # Unified dependencies
├── config/
│   └── config.yaml           # Global hyperparameters & structural weights
├── data/                     # Data source and generation scripts
│   ├── coursera/             # Education domain datasets
│   └── linkedin/             # Job domain datasets
├── src/                      # Core AHIPS-CDR implementation
│   ├── data_preprocessing/   # Feature engineering & Data Loader
│   ├── models/               # IPS-Attention & Joint Predictor
│   ├── utils/                # Metrics & Loss functions
│   ├── train.py              # Main training orchestrator
│   └── evaluate.py           # Inference & benchmarking
├── baselines/                # Comparative baselines implementation
│   ├── graph_neural_networks/# HGT
│   └── cross_domain_special/ # HAGO, MI-DTCDR, SEAGULL
└── scripts/                  # Replication automation
    ├── run_main_exp.sh       # Main comparison benchmarks
    ├── run_ablation.sh       # Architectural ablation analysis
    └── run_robustness.sh     # Sparsity & Robustness stress-tests
```
---
## 🚀 Quick Start & Replication
1. Data Initialization
Ensure raw CSV files are placed in data/{coursera, linkedin}/. Run the unified data provider to prepare tensors:
```text
python src/data_preprocessing/dataset_loader.py
```
2. Primary Experiments
To replicate the main results reported in Table 5 and Table 6 (comparing AHIPS-CDR against all baselines):
```text
cd scripts
chmod +x run_main_exp.sh
./run_main_exp.sh
```
3. Ablation Studies
To verify the necessity of the IPS module and dual-level attention mechanism:

```text
./run_ablation.sh
```
4. Robustness Analysis
To evaluate model stability under progressive data sparsity (0% to 40% edge dropout):

```text
./run_robustness.sh
```
## 📊 Baseline Implementation
This repository includes native implementations of four state-of-the-art baselines, accessible via baselines/:

HGT: Heterogeneous Graph Transformer

HAGO: Heterogeneous Adaptive Graph coOrdinators

MI-DTCDR: Multi-Interactive Deep Target Cross-Domain Recommendation

SEAGULL: Disentangled representation learning for cross-domain recommendation

To run a specific baseline independently:

```text
python baselines/baselines_runner.py --model {MODEL_NAME} --dataset {linkedin|coursera}
```
## 📝 Configuration
All model constants are decoupled in config/config.yaml.

node_embedding_dim: Latent space dimension.

alpha, beta, lambda: Weights for Joint Optimization Loss (MF + Structure + HeteSim).

learning_rate: Optimization step size.
## Notice
The current implementation is still under refinement.
The training code for AHIPS-CDR  will be released after the associated paper is accepted.
