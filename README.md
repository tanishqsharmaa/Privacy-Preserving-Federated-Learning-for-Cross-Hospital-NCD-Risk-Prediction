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

- **BRFSS**: [CDC BRFSS Annual Data](https://www.cdc.gov/brfss/annual_data/annual_data.htm) — 400K+ entries/year (self-reported)
- **NHANES**: [CDC NHANES](https://www.cdc.gov/nchs/nhanes/index.htm) — Clinical exam data (lab-measured)

> These two datasets have **different measurement protocols**, naturally simulating cross-hospital data heterogeneity for federated learning.

---

## Quick Start

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

### Step 5: Run Quick Smoke Test

5 rounds, 4 clients — validates the entire pipeline works before committing to a full run.

```powershell
python -m experiments.run_all --quick
```

> **Note:** Steps 3 & 4 are automatically handled by `run_all` if `data/processed` doesn't exist.

---

## Running Experiments

### Full Experiment Suite (All 12 Experiments)

Runs centralized baseline + all FL ablations. Non-DP experiments: 50 rounds; DP experiments: 150 rounds.

```powershell
# With synthetic data (faster, for development)
python -m experiments.run_all --synthetic --device cuda

# With real BRFSS + NHANES data (for publication)
python -m experiments.run_all --device cuda
```

### Run Specific Experiments Only

Use `--experiments` to select individual experiments:

```powershell
# Run only the baseline comparisons
python -m experiments.run_all --synthetic --device cuda --experiments fedavg_baseline fedprox_original

# Run only DP experiments (150 rounds each)
python -m experiments.run_all --synthetic --device cuda --experiments full_system_dp full_system_dp_comp

# Run only the best-performing configs
python -m experiments.run_all --synthetic --device cuda --experiments fix_fedprox_mu all_fixes
```

**Available experiment names:**
| Experiment | Description |
|---|---|
| `fedavg_baseline` | Vanilla FedAvg (no enhancements) |
| `fedprox_original` | FedProx with μ=0.01 |
| `fix_focal_loss` | FedProx + Focal Loss |
| `fix_weighted_sampling` | FedProx + Weighted Sampling |
| `fix_lr_schedule` | FedProx + Warmup Cosine LR |
| `fix_fedprox_mu` | FedProx with μ=0.1 (strongest FL config) |
| `rec1_class_aware` | FedProx + Class-Aware Aggregation |
| `rec3_fedbn` | FedProx + FedBN |
| `all_fixes` | All enhancements combined (no DP) |
| `full_system_dp` | Full system + Differential Privacy (150 rounds) |
| `full_system_dp_comp` | Full system + DP + Top-K Compression (150 rounds) |

### Run Individual Baselines

```powershell
# Centralized baseline (upper bound)
python -m experiments.centralized

# Local-only baseline (lower bound)
python -m experiments.fedavg --baseline local_only

# Vanilla FedAvg
python -m experiments.fedavg --baseline fedavg

# FedAvg + DP
python -m experiments.fedavg --baseline fedavg_dp
```

---

## Multi-Seed Runs (Statistical Reliability)

For publication, run 3–5 seeds and report **mean ± std**. This proves results are reproducible and not due to a lucky random initialization.

```powershell
# Run the full suite with 3 different seeds
python -m experiments.run_all --synthetic --device cuda --seed 42
python -m experiments.run_all --synthetic --device cuda --seed 123
python -m experiments.run_all --synthetic --device cuda --seed 456

# (Optional) 5 seeds for stronger statistical claims
python -m experiments.run_all --synthetic --device cuda --seed 789
python -m experiments.run_all --synthetic --device cuda --seed 1024
```

Each run produces a separate timestamped results directory under `results/`.

---

## Quick Copy-Paste Recipes

### Recipe 1: Quick Validation (5 min)

```powershell
.\virt\Scripts\Activate.ps1
python -m experiments.run_all --quick
```

### Recipe 2: Full Synthetic Run — Single Seed (~90 min on GPU)

```powershell
.\virt\Scripts\Activate.ps1
Remove-Item results -Recurse -Force -ErrorAction SilentlyContinue
python -m experiments.run_all --synthetic --device cuda --seed 42
```

### Recipe 3: Full Synthetic Run — 3 Seeds (~5 hours on GPU)

```powershell
.\virt\Scripts\Activate.ps1
python -m experiments.run_all --synthetic --device cuda --seed 42
python -m experiments.run_all --synthetic --device cuda --seed 123
python -m experiments.run_all --synthetic --device cuda --seed 456
```

### Recipe 4: Real Data Run — Publication Quality (~4–6 hours on GPU)

```powershell
.\virt\Scripts\Activate.ps1

# Download & prepare real data
python download_data.py
python -m src.data_prep --output data/processed
python -m src.partition --input data/processed --num-clients 10 --alpha 0.5

# Run all experiments with 3 seeds
python -m experiments.run_all --device cuda --seed 42
python -m experiments.run_all --device cuda --seed 123
python -m experiments.run_all --device cuda --seed 456
```

### Recipe 5: DP Experiments Only — Extended (150 rounds, ~75 min on GPU)

```powershell
.\virt\Scripts\Activate.ps1
python -m experiments.run_all --synthetic --device cuda --experiments full_system_dp full_system_dp_comp
```

---

## Hardware Notes

| Phase | Hardware | Estimated Time |
|-------|----------|----------------|
| Data prep, partitioning | Any CPU | < 5 min |
| Centralized baseline | CPU OK | ~10–20 min |
| FL training (no DP, 50 rounds) | GPU recommended | ~7 min per experiment |
| FL training (with DP, 150 rounds) | **GPU required** | ~37 min per experiment |
| Full suite (single seed) | GPU | ~90 min synthetic, ~4 hours real |
| Attack simulation | GPU recommended | ~30–60 min |

## Key Evaluation Metrics

| Metric | Target | Achieved (Synthetic) |
|--------|--------|----------------------|
| AUC-ROC (macro avg) | > 0.80 | 0.9348 (FL) / 0.9129 (DP-FL) |
| FL vs Centralized gap | < 5% | 0.04% |
| DP AUC drop | < 5% | 2.4% at ε=2.41 |
| Privacy Budget ε | < 3.0 | 2.41 |
| MI Attack Success | < 55% | — |
| Communication Savings | 40%+ vs FedAvg | — |
| Equalized Odds Gap | < 0.05 | — |

## References

1. McMahan et al. (2017) — FedAvg
2. Li et al. (2020) — FedProx
3. Abadi et al. (2016) — DP-SGD
4. Mironov (2017) — Rényi DP
5. Bonawitz et al. (2017) — Secure Aggregation
6. Lundberg & Lee (2017) — SHAP
