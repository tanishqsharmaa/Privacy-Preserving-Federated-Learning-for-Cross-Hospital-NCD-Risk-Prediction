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
│   ├── attack_sim.py     ← Gradient inversion + MI attacks
│   └── utils.py          ← Metrics, plotting, fairness
├── experiments/
│   ├── centralized.py    ← Centralized baseline
│   ├── fedavg.py         ← Vanilla FedAvg baseline
│   └── run_all.py        ← Run all experiments
├── results/              ← Metrics, plots, logs
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Synthetic for testing)
```bash
python -m src.data_prep --synthetic --output data/processed
python -m src.partition --input data/processed --num-clients 10 --alpha 0.5
```

### 3. Run Centralized Baseline
```bash
python -m experiments.centralized
```

### 4. Run Federated Learning (⚠️ Use GPU machine for this)
```bash
python -m src.server --num-rounds 50 --num-clients 10 --noise-multiplier 1.1
```

### 5. Run All Experiments
```bash
python -m experiments.run_all
```

## Datasets

- **BRFSS**: [CDC BRFSS Annual Data](https://www.cdc.gov/brfss/annual_data/annual_data.htm) — 400K+ entries/year
- **NHANES**: [CDC NHANES](https://www.cdc.gov/nchs/nhanes/index.htm) — Different measurement protocols

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
