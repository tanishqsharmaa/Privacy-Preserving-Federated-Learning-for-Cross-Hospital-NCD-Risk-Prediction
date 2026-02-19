"""
Utility functions: metrics computation, fairness analysis, plotting, and logging.
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    precision_score, recall_score
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "results", level: int = logging.INFO) -> logging.Logger:
    """Configure logging with both file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("ppfl-ncd")
    logger.setLevel(level)
    
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "experiment.log"))
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(fh)
    
    return logger


# ---------------------------------------------------------------------------
# Seed Management
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Multi-Task Metrics
# ---------------------------------------------------------------------------

DISEASE_NAMES = ["diabetes", "hypertension", "cardiovascular_disease"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics for a single task.
    
    Args:
        y_true: Ground truth binary labels, shape (N,)
        y_pred_proba: Predicted probabilities, shape (N,)
        threshold: Classification threshold
    
    Returns:
        Dictionary with auc_roc, f1, accuracy, precision, recall
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    
    # AUC-ROC requires at least 2 classes present
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba))
    else:
        metrics["auc_roc"] = 0.0
    
    return metrics


def compute_multitask_metrics(
    y_true_list: List[np.ndarray],
    y_pred_proba_list: List[np.ndarray],
    disease_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for all three NCD prediction tasks.
    
    Args:
        y_true_list: List of 3 ground truth arrays
        y_pred_proba_list: List of 3 prediction probability arrays
        disease_names: Names for each task
        threshold: Classification threshold
    
    Returns:
        Nested dict: {disease_name: {metric_name: value}}
    """
    if disease_names is None:
        disease_names = DISEASE_NAMES
    
    all_metrics = {}
    for name, y_true, y_pred in zip(disease_names, y_true_list, y_pred_proba_list):
        all_metrics[name] = compute_metrics(y_true, y_pred, threshold)
    
    # Compute macro-averaged metrics
    macro = {}
    for metric_key in ["accuracy", "f1", "auc_roc", "precision", "recall"]:
        vals = [all_metrics[d][metric_key] for d in disease_names]
        macro[metric_key] = float(np.mean(vals))
    all_metrics["macro_avg"] = macro
    
    return all_metrics


# ---------------------------------------------------------------------------
# Fairness Metrics
# ---------------------------------------------------------------------------

def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray
) -> Dict[str, float]:
    """
    Compute Equalized Odds gap across sensitive groups.
    
    Equalized Odds requires equal TPR and FPR across groups.
    Returns the maximum gap in TPR and FPR across all group pairs.
    """
    groups = np.unique(sensitive_attr)
    tprs, fprs = [], []
    
    for g in groups:
        mask = sensitive_attr == g
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        
        # TPR = TP / (TP + FN)
        pos_mask = yt == 1
        tpr = yp[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
        
        # FPR = FP / (FP + TN)
        neg_mask = yt == 0
        fpr = yp[neg_mask].mean() if neg_mask.sum() > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    tpr_gap = max(tprs) - min(tprs) if tprs else 0.0
    fpr_gap = max(fprs) - min(fprs) if fprs else 0.0
    
    return {
        "tpr_gap": float(tpr_gap),
        "fpr_gap": float(fpr_gap),
        "equalized_odds_gap": float(max(tpr_gap, fpr_gap))
    }


def demographic_parity_gap(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray
) -> float:
    """
    Compute Demographic Parity gap.
    
    Demographic Parity requires equal positive prediction rates across groups.
    Returns the max gap in positive prediction rate.
    """
    groups = np.unique(sensitive_attr)
    rates = []
    
    for g in groups:
        mask = sensitive_attr == g
        if mask.sum() == 0:
            continue
        rates.append(float(y_pred[mask].mean()))
    
    return float(max(rates) - min(rates)) if rates else 0.0


def compute_fairness_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_attrs: Dict[str, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, Dict]:
    """
    Compute fairness metrics across multiple sensitive attributes.
    
    Args:
        y_true: Ground truth labels
        y_pred_proba: Prediction probabilities
        sensitive_attrs: Dict mapping attribute name to group labels
        threshold: Classification threshold
    
    Returns:
        Dict mapping attribute name to fairness metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    report = {}
    
    for attr_name, attr_values in sensitive_attrs.items():
        eo = equalized_odds_gap(y_true, y_pred, attr_values)
        dp = demographic_parity_gap(y_pred, attr_values)
        report[attr_name] = {
            "equalized_odds": eo,
            "demographic_parity_gap": dp
        }
    
    return report


# ---------------------------------------------------------------------------
# Plotting Utilities (Publication Quality)
# ---------------------------------------------------------------------------

def set_plot_style():
    """Set publication-quality plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_privacy_utility_tradeoff(
    epsilons: List[float],
    auc_scores: Dict[str, List[float]],
    save_path: str = "results/privacy_utility_tradeoff.png"
):
    """Plot AUC-ROC vs. privacy budget ε for each disease."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    markers = ["o", "s", "^"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    
    for i, (disease, scores) in enumerate(auc_scores.items()):
        ax.plot(epsilons, scores, marker=markers[i], color=colors[i],
                label=disease.replace("_", " ").title(), linewidth=2, markersize=8)
    
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Privacy-Utility Tradeoff")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_communication_efficiency(
    k_ratios: List[float],
    bytes_transmitted: List[float],
    accuracies: List[float],
    save_path: str = "results/communication_efficiency.png"
):
    """Plot accuracy vs. communication cost for different compression ratios."""
    set_plot_style()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = "#2196F3"
    ax1.set_xlabel("Top-K Ratio")
    ax1.set_ylabel("AUC-ROC", color=color1)
    ax1.plot(k_ratios, accuracies, "o-", color=color1, linewidth=2, markersize=8)
    ax1.tick_params(axis="y", labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = "#FF5722"
    ax2.set_ylabel("Bytes Transmitted (MB)", color=color2)
    ax2.plot(k_ratios, bytes_transmitted, "s--", color=color2, linewidth=2, markersize=8)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    ax1.set_title("Communication Efficiency vs. Accuracy")
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_noniid_robustness(
    alphas: List[float],
    fedavg_scores: List[float],
    fedprox_scores: List[float],
    save_path: str = "results/noniid_robustness.png"
):
    """Plot accuracy vs. Dirichlet α for FedAvg vs FedProx."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(alphas, fedavg_scores, "o-", color="#FF5722", label="FedAvg",
            linewidth=2, markersize=8)
    ax.plot(alphas, fedprox_scores, "s-", color="#4CAF50", label="FedProx",
            linewidth=2, markersize=8)
    
    ax.set_xlabel("Dirichlet α (higher = more IID)")
    ax.set_ylabel("AUC-ROC (Macro Avg)")
    ax.set_title("Non-IID Robustness: FedAvg vs. FedProx")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison(
    methods: List[str],
    metrics_per_method: Dict[str, Dict[str, float]],
    save_path: str = "results/baseline_comparison.png"
):
    """Bar plot comparing all baselines across AUC-ROC per disease."""
    set_plot_style()
    diseases = DISEASE_NAMES
    x = np.arange(len(diseases))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#9E9E9E", "#FF9800", "#2196F3", "#9C27B0", "#4CAF50"]
    
    for i, method in enumerate(methods):
        scores = [metrics_per_method[method].get(d, {}).get("auc_roc", 0) 
                  for d in diseases]
        ax.bar(x + i * width, scores, width, label=method, color=colors[i % len(colors)])
    
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Baseline Comparison: AUC-ROC per Disease")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([d.replace("_", " ").title() for d in diseases])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1.0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_convergence_curve(
    rounds: List[int],
    losses: Dict[str, List[float]],
    save_path: str = "results/convergence.png"
):
    """Plot training loss over FL rounds for different methods."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for i, (method, loss_values) in enumerate(losses.items()):
        ax.plot(rounds[:len(loss_values)], loss_values, 
                color=colors[i % len(colors)], label=method, linewidth=2)
    
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence")
    ax.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, filepath: str):
    """Save metrics dictionary to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(filepath: str) -> dict:
    """Load metrics dictionary from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)
