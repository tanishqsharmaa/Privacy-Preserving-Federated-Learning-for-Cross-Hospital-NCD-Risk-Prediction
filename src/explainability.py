"""
SHAP-based local explainability for multi-task NCD model.

After each FL round, each hospital node independently computes SHAP values
using its own local data. No SHAP values are shared with the server —
explainability is local and privacy-preserving by design.
"""

import os
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppfl-ncd.explainability")

DISEASE_NAMES = ["diabetes", "hypertension", "cardiovascular_disease"]
FEATURE_NAMES = [
    "age", "sex", "bmi", "race_ethnicity",
    "smoking_status", "alcohol_consumption", "physical_activity",
    "blood_pressure_systolic", "blood_pressure_diastolic",
    "cholesterol_total", "cholesterol_hdl", "blood_glucose",
    "has_kidney_disease", "has_stroke_history", "has_arthritis",
]


class MultiTaskSHAPWrapper:
    """
    Wrapper to make the multi-task model compatible with SHAP.
    SHAP expects a model that returns a single output, so we
    create separate wrappers for each disease head.
    """
    
    def __init__(self, model, task_index: int, device: torch.device):
        self.model = model
        self.task_index = task_index
        self.device = device
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            preds = self.model(x_tensor)
            return preds[self.task_index].cpu().numpy()


def compute_local_shap(
    model: torch.nn.Module,
    X_local: np.ndarray,
    feature_names: List[str] = None,
    background_samples: int = 100,
    num_explain: int = 200,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Compute SHAP values for each NCD prediction head.
    
    Args:
        model: Trained multi-task model
        X_local: Local dataset features, shape (N, D)
        feature_names: Feature names for labeling
        background_samples: Number of background samples for SHAP
        num_explain: Number of samples to explain
        device: Device for computation
    
    Returns:
        Dict mapping disease name to SHAP values array (num_explain, D)
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. pip install shap>=0.44.0")
        return {}
    
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Background data (representative subset)
    bg_idx = np.random.choice(len(X_local), min(background_samples, len(X_local)), replace=False)
    background = X_local[bg_idx]
    
    # Samples to explain
    explain_idx = np.random.choice(len(X_local), min(num_explain, len(X_local)), replace=False)
    X_explain = X_local[explain_idx]
    
    shap_results = {}
    
    for task_idx, disease in enumerate(DISEASE_NAMES):
        logger.info(f"  Computing SHAP for {disease}...")
        
        # Create task-specific wrapper
        wrapper = MultiTaskSHAPWrapper(model, task_idx, device)
        
        try:
            # Use KernelExplainer (model-agnostic, works with any PyTorch model)
            explainer = shap.KernelExplainer(wrapper, background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            shap_results[disease] = shap_values
            logger.info(f"    {disease}: SHAP values shape = {shap_values.shape}")
            
        except Exception as e:
            logger.warning(f"    SHAP failed for {disease}: {e}")
            shap_results[disease] = np.zeros((len(X_explain), X_local.shape[1]))
    
    return shap_results


def get_top_features(
    shap_values: np.ndarray,
    feature_names: List[str] = None,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Get top-K most important features by mean absolute SHAP value.
    
    Args:
        shap_values: SHAP values array (N, D)
        feature_names: Feature names
        top_k: Number of top features
    
    Returns:
        List of (feature_name, mean_abs_shap) tuples, sorted descending
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_k]
    
    return [(feature_names[i], float(mean_abs[i])) for i in top_indices]


def compute_shap_consistency(
    client_top_features: Dict[int, List[str]],
    top_k: int = 5
) -> float:
    """
    Compute SHAP feature consistency across hospital nodes.
    
    Measures how much overlap there is in the top-K features across
    different clients. High consistency = stable model explanations.
    
    Args:
        client_top_features: Dict mapping client_id to list of top feature names
        top_k: Number of top features to compare
    
    Returns:
        Consistency score (0-1, fraction of pairwise overlap)
    """
    client_ids = list(client_top_features.keys())
    if len(client_ids) < 2:
        return 1.0
    
    overlaps = []
    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            set_i = set(client_top_features[client_ids[i]][:top_k])
            set_j = set(client_top_features[client_ids[j]][:top_k])
            overlap = len(set_i & set_j) / top_k
            overlaps.append(overlap)
    
    return float(np.mean(overlaps))


def plot_shap_summary(
    shap_values: np.ndarray,
    X_data: np.ndarray,
    feature_names: List[str] = None,
    disease_name: str = "",
    save_path: str = "results/shap_summary.png"
):
    """Generate SHAP summary bar plot."""
    try:
        import shap
        
        if feature_names is None:
            feature_names = FEATURE_NAMES
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_data,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=15,
        )
        plt.title(f"SHAP Feature Importance — {disease_name.replace('_', ' ').title()}")
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP plot saved to {save_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate SHAP plot: {e}")


def plot_multi_client_shap(
    client_shap_results: Dict[int, Dict[str, List[Tuple[str, float]]]],
    disease: str = "diabetes",
    top_k: int = 10,
    save_path: str = "results/multi_client_shap.png"
):
    """
    Bar plot comparing top features across multiple hospital nodes.
    
    Args:
        client_shap_results: {client_id: {disease: [(feature, importance), ...]}}
        disease: Disease to plot
        top_k: Number of features to show
        save_path: Save path
    """
    fig, axes = plt.subplots(1, min(len(client_shap_results), 4), 
                              figsize=(5 * min(len(client_shap_results), 4), 6),
                              sharey=True)
    
    if len(client_shap_results) == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, top_k))
    
    for ax_idx, (client_id, results) in enumerate(list(client_shap_results.items())[:4]):
        if disease not in results:
            continue
        
        features, importances = zip(*results[disease][:top_k])
        features = [f.replace("_", "\n") for f in features]
        
        axes[ax_idx].barh(range(len(features)), importances, color=colors)
        axes[ax_idx].set_yticks(range(len(features)))
        axes[ax_idx].set_yticklabels(features, fontsize=8)
        axes[ax_idx].set_title(f"Hospital {client_id}", fontsize=10)
        axes[ax_idx].set_xlabel("Mean |SHAP|")
    
    plt.suptitle(f"Feature Importance Across Hospitals — {disease.replace('_', ' ').title()}")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_explainability_analysis(
    model: torch.nn.Module,
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    feature_names: List[str] = None,
    top_k: int = 10,
    results_dir: str = "results",
    device: str = "cpu",
):
    """
    Run full SHAP analysis across all hospital nodes.
    
    Args:
        model: Trained global model
        client_data: List of (X, Y) per client
        feature_names: Feature names
        top_k: Top features to report
        results_dir: Results directory
        device: Computation device
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    
    all_client_results = {}
    all_client_top_features = {}
    
    for cid, (X, Y) in enumerate(client_data):
        if len(X) < 10:
            logger.warning(f"Client {cid}: too few samples for SHAP")
            continue
        
        logger.info(f"Client {cid}: Computing SHAP values ({len(X)} samples)...")
        shap_values = compute_local_shap(
            model, X, feature_names,
            background_samples=min(100, len(X)),
            num_explain=min(200, len(X)),
            device=device,
        )
        
        client_results = {}
        client_tops = {}
        
        for disease, sv in shap_values.items():
            top_features = get_top_features(sv, feature_names, top_k)
            client_results[disease] = top_features
            client_tops[disease] = [f[0] for f in top_features]
            
            # Save plot per client per disease
            plot_shap_summary(
                sv, X, feature_names, disease,
                save_path=os.path.join(results_dir, f"shap_client{cid}_{disease}.png")
            )
        
        all_client_results[cid] = client_results
        all_client_top_features[cid] = client_tops
    
    # Compute consistency across clients for each disease
    consistency = {}
    for disease in DISEASE_NAMES:
        disease_tops = {
            cid: tops.get(disease, [])
            for cid, tops in all_client_top_features.items()
        }
        consistency[disease] = compute_shap_consistency(disease_tops, top_k=5)
        logger.info(f"SHAP consistency ({disease}): {consistency[disease]:.2%}")
    
    # Multi-client comparison plot
    for disease in DISEASE_NAMES:
        plot_multi_client_shap(
            all_client_results, disease, top_k,
            save_path=os.path.join(results_dir, f"multi_client_shap_{disease}.png")
        )
    
    # Save results
    import json
    results = {
        "consistency": consistency,
        "top_features_per_client": {
            str(cid): {
                disease: [(f, float(v)) for f, v in feats]
                for disease, feats in client_results.items()
            }
            for cid, client_results in all_client_results.items()
        }
    }
    
    filepath = os.path.join(results_dir, "shap_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"SHAP results saved to {filepath}")
    return results
