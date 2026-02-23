# Privacy-Preserving Federated Learning for Cross-Hospital NCD Risk Prediction

A privacy-preserving federated learning system for simultaneous **multi-disease Non-Communicable Disease (NCD) risk prediction** across simulated hospital networks.

## Key Features

- **Multi-Task Learning**: Simultaneous prediction of diabetes, hypertension, and cardiovascular disease risk
- **Privacy-Preserving**: DP-SGD (Opacus) with Rényi DP accounting (ε < 3.0) + Secure Aggregation
- **Non-IID Robust**: FedProx personalized aggregation for heterogeneous hospital data
- **Communication Efficient**: Top-K gradient sparsification (40-60% bandwidth reduction)
- **Explainable**: Local SHAP analysis per hospital node for clinical interpretability
- **Attack Validated**: Empirical defense against gradient inversion and membership inference attacks
- **Fair**: Demographic fairness analysis (Equalized Odds, Demographic Parity)

## Architecture

```
Hospital Nodes (×10)          Central Server
┌──────────────────┐          ┌─────────────────────┐
│ Local Data       │          │ SecAgg+ Aggregation  │
│ Local Training   │  ──────► │ FedProx Global Update│
│ DP-SGD (Opacus)  │  ◄────── │ Global Model         │
│ Top-K Compress   │          │   Broadcast          │
│ SHAP (local XAI) │          └─────────────────────┘
└──────────────────┘
```

## Project Structure

```
├── data/
│   ├── raw/              ← Downloaded BRFSS + NHANES datasets
│   ├── processed/        ← Cleaned, merged datasets
│   └── partitions/       ← Non-IID Dirichlet hospital splits
├── src/
│   ├── config.py         ← Central configuration
│   ├── data_prep.py      ← Data preprocessing pipeline
│   ├── partition.py      ← Dirichlet non-IID partitioning
│   ├── model.py          ← Multi-task neural network
│   ├── client.py         ← Flower FL client (DP-SGD + FedProx)
│   ├── server.py         ← Flower FL server (FedProx strategy)
│   ├── privacy.py        ← DP-SGD + RDP accounting
│   ├── compression.py    ← Top-K gradient sparsification
│   ├── explainability.py ← SHAP computation per node
│   └── utils.py          ← Metrics, plotting, fairness
├── experiments/
│   ├── centralized.py    ← Centralized baseline
│   ├── fedavg.py         ← Vanilla FedAvg baseline
│   └── run_all.py        ← Run all experiments
├── results/              ← Metrics, plots, logs
├── download_data.py      ← BRFSS + NHANES data downloader
└── requirements.txt
```

## Datasets

- **BRFSS**: [CDC BRFSS Annual Data](https://www.cdc.gov/brfss/annual_data/annual_data.htm) — 400K+ entries/year
- **NHANES**: [CDC NHANES](https://www.cdc.gov/nchs/nhanes/index.htm) — Different measurement protocols

---

## Federated Learning Pipeline

> Step-by-step execution guide for data preparation, partitioning, and experiment runs.

### Step 0: Activate Virtual Environment

```powershell
.\virt\Scripts\Activate.ps1
```

### Step 1: Clean Previous Results *(Optional — Fresh Start)*

```powershell
Remove-Item data\processed -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item data\partitions -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item results -Recurse -Force -ErrorAction SilentlyContinue
```

### Step 2: Download Raw Data

> Skip if already downloaded.

```powershell
python download_data.py
```

### Step 3: Process & Harmonize Raw Data

Loads BRFSS (1.16 GB) + NHANES → harmonizes features → train/test split → saves CSVs + scaler.

```powershell
python -m src.data_prep --output data/processed
```

### Step 4: Partition Data for Federated Learning

Dirichlet non-IID split → 5 BRFSS + 5 NHANES hospital nodes.

```powershell
python -m src.partition --input data/processed --num-clients 10 --alpha 0.5
```

### Step 5: Run Quick Test

Runs: Centralized → Local-Only → FedAvg → FedAvg+DP → Full System → Sweeps (all abbreviated).  
5 rounds, 3 clients — validates everything works.

```powershell
python -m experiments.run_all --quick
```

### Step 6: Run Full Experiment Suite

Full 50-round training for all baselines + privacy-utility sweep + non-IID robustness sweep.  
**Production run — takes several hours.**

```powershell
python -m experiments.run_all
```

### Or Run Individual Experiments

```powershell
# Centralized baseline (upper bound)
python -m experiments.centralized

# Local-only baseline (lower bound)
python -m experiments.fedavg --baseline local_only

# Vanilla FedAvg (no DP)
python -m experiments.fedavg --baseline fedavg

# FedAvg + DP
python -m experiments.fedavg --baseline fedavg_dp

# Full system (FedProx + DP + Compression)
python -m experiments.run_all --quick   # use --quick for fast validation
```

---

## Quick Copy-Paste (All Steps at Once)

```powershell
.\virt\Scripts\Activate.ps1

# Clean slate
Remove-Item data\processed -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item data\partitions -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item results -Recurse -Force -ErrorAction SilentlyContinue

# Pipeline
python download_data.py
python -m src.data_prep --output data/processed
python -m src.partition --input data/processed --num-clients 10 --alpha 0.5
python -m experiments.run_all --quick
```

> **Note:** Steps 3 & 4 (data prep + partition) are also automatically handled by `run_all` if `data/processed` doesn't exist. But running them separately lets you validate each stage independently before committing to the full experiment run.

---

## Hardware Notes

| Phase | Hardware | Notes |
|-------|----------|-------|
| Data prep, partitioning | Any CPU | Fast, <5 min |
| Centralized baseline | CPU OK | ~10-20 min |
| FL training (no DP) | GPU recommended | ~30 min |
| FL training (with DP) | **GPU required** | ~2-4 hours |
| Attack simulation | GPU recommended | ~30-60 min |

## Key Evaluation Metrics

| Metric | Target |
|--------|--------|
| AUC-ROC (per disease) | > 0.80 |
| Privacy Budget ε | < 3.0 |
| MI Attack Success | < 55% |
| Communication Savings | 40%+ vs FedAvg |
| Equalized Odds Gap | < 0.05 |

## References

1. McMahan et al. (2017) — FedAvg
2. Li et al. (2020) — FedProx
3. Abadi et al. (2016) — DP-SGD
4. Mironov (2017) — Rényi DP
5. Bonawitz et al. (2017) — Secure Aggregation
6. Lundberg & Lee (2017) — SHAP
