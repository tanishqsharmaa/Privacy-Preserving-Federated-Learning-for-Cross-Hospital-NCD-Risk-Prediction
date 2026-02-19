"""
Dirichlet-based non-IID data partitioning for federated learning.

Implements realistic hospital data distribution using Dirichlet distribution,
the standard method used in top FL papers (FedAvg, FedProx, SCAFFOLD).
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppfl-ncd.partition")


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_client: int = 10
) -> List[List[int]]:
    """
    Partition data indices using Dirichlet distribution.
    
    Small alpha (e.g., 0.1) → highly non-IID (each client gets mostly one class)
    Large alpha (e.g., 10.0) → nearly IID (uniform distribution across clients)
    
    Args:
        labels: Array of integer class labels, shape (N,)
        num_clients: Number of FL clients (hospital nodes)
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility
        min_samples_per_client: Minimum samples guaranteed per client
    
    Returns:
        List of index lists, one per client
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for k_idx, k in enumerate(unique_labels):
        # Indices of samples with label k
        idx_k = np.where(labels == k)[0]
        rng.shuffle(idx_k)
        
        # Draw Dirichlet distribution for this class
        proportions = rng.dirichlet([alpha] * num_clients)
        
        # Calculate split sizes
        splits = (proportions * len(idx_k)).astype(int)
        
        # Adjust last split to account for rounding
        splits[-1] = len(idx_k) - splits[:-1].sum()
        
        # Distribute indices to clients
        start = 0
        for cid in range(num_clients):
            count = max(0, splits[cid])
            client_indices[cid].extend(idx_k[start:start + count].tolist())
            start += count
    
    # Ensure minimum samples per client
    all_indices = set(range(len(labels)))
    assigned = set()
    for ci in client_indices:
        assigned.update(ci)
    unassigned = list(all_indices - assigned)
    rng.shuffle(unassigned)
    
    for cid in range(num_clients):
        deficit = min_samples_per_client - len(client_indices[cid])
        if deficit > 0 and unassigned:
            transfer = unassigned[:deficit]
            client_indices[cid].extend(transfer)
            unassigned = unassigned[deficit:]
    
    # Shuffle each client's data
    for ci in client_indices:
        rng.shuffle(ci)
    
    return client_indices


def multi_label_dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    primary_label_col: int = 0
) -> List[List[int]]:
    """
    Partition multi-label data using Dirichlet distribution.
    
    Uses the primary label column for partitioning (since Dirichlet
    partitioning is designed for single-label settings). The other
    labels come along with the samples.
    
    Args:
        targets: Multi-label array, shape (N, num_tasks)
        num_clients: Number of FL clients
        alpha: Dirichlet concentration parameter
        seed: Random seed
        primary_label_col: Which target column to use for Dirichlet split
    
    Returns:
        List of index lists, one per client
    """
    # Create composite label for more balanced partitioning
    # Combine all three binary targets into an 8-class composite
    composite = np.zeros(len(targets), dtype=int)
    for col in range(targets.shape[1]):
        composite += (targets[:, col].astype(int) << col)
    
    return dirichlet_partition(composite, num_clients, alpha, seed)


def partition_by_source(
    df: pd.DataFrame,
    num_brfss_clients: int = 5,
    num_nhanes_clients: int = 5,
    alpha: float = 0.5,
    seed: int = 42,
    target_columns: List[str] = None
) -> List[List[int]]:
    """
    Partition data into hospital nodes, respecting data source.
    
    BRFSS samples → first N_brfss clients
    NHANES samples → last N_nhanes clients
    Within each source, use Dirichlet partitioning.
    
    Args:
        df: DataFrame with 'source' column
        num_brfss_clients: Number of BRFSS hospital nodes
        num_nhanes_clients: Number of NHANES hospital nodes
        alpha: Dirichlet concentration parameter
        seed: Random seed
        target_columns: Target column names for multi-label partitioning
    
    Returns:
        List of index lists (length = num_brfss + num_nhanes)
    """
    if target_columns is None:
        target_columns = ["diabetes", "hypertension", "cardiovascular_disease"]
    
    all_partitions = []
    
    for source, num_clients in [("brfss", num_brfss_clients), 
                                 ("nhanes", num_nhanes_clients)]:
        mask = df["source"] == source
        source_indices = np.where(mask)[0]
        
        if len(source_indices) == 0:
            logger.warning(f"No samples from source '{source}'. "
                           f"Creating {num_clients} empty partitions.")
            all_partitions.extend([[] for _ in range(num_clients)])
            continue
        
        # Get targets for this source
        targets = df.iloc[source_indices][target_columns].values
        
        # Partition within this source
        local_partitions = multi_label_dirichlet_partition(
            targets, num_clients, alpha, seed
        )
        
        # Map local indices back to global indices
        for local_partition in local_partitions:
            global_partition = [int(source_indices[i]) for i in local_partition 
                                if i < len(source_indices)]
            all_partitions.append(global_partition)
    
    return all_partitions


def compute_partition_stats(
    df: pd.DataFrame,
    partitions: List[List[int]],
    target_columns: List[str] = None
) -> Dict:
    """Compute statistics for each partition."""
    if target_columns is None:
        target_columns = ["diabetes", "hypertension", "cardiovascular_disease"]
    
    stats = {}
    for cid, indices in enumerate(partitions):
        if not indices:
            stats[f"client_{cid}"] = {"num_samples": 0}
            continue
        
        client_df = df.iloc[indices]
        client_stats = {
            "num_samples": len(indices),
            "source": client_df["source"].value_counts().to_dict() if "source" in client_df else {},
        }
        
        for col in target_columns:
            if col in client_df.columns:
                client_stats[f"{col}_prevalence"] = float(client_df[col].mean())
                client_stats[f"{col}_count"] = int(client_df[col].sum())
        
        stats[f"client_{cid}"] = client_stats
    
    return stats


def visualize_partitions(
    df: pd.DataFrame,
    partitions: List[List[int]],
    target_columns: List[str] = None,
    save_path: str = "results/partition_distribution.png"
):
    """Create visualization of label distribution across clients."""
    if target_columns is None:
        target_columns = ["diabetes", "hypertension", "cardiovascular_disease"]
    
    num_clients = len(partitions)
    fig, axes = plt.subplots(1, len(target_columns), figsize=(5 * len(target_columns), 6))
    
    if len(target_columns) == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, num_clients))
    
    for ax_idx, col in enumerate(target_columns):
        prevalences = []
        sizes = []
        labels = []
        
        for cid, indices in enumerate(partitions):
            if not indices:
                prevalences.append(0)
                sizes.append(0)
            else:
                client_df = df.iloc[indices]
                prevalences.append(client_df[col].mean())
                sizes.append(len(indices))
            labels.append(f"H{cid}")
        
        bars = axes[ax_idx].bar(labels, prevalences, color=colors[:num_clients])
        axes[ax_idx].set_title(col.replace("_", " ").title())
        axes[ax_idx].set_ylabel("Prevalence")
        axes[ax_idx].set_ylim(0, 1)
        axes[ax_idx].tick_params(axis="x", rotation=45)
        
        # Add sample count on top of bars
        for bar, size in zip(bars, sizes):
            axes[ax_idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={size}", ha="center", va="bottom", fontsize=8
            )
    
    plt.suptitle(f"Label Distribution Across {num_clients} Hospital Nodes", fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Partition distribution plot saved to {save_path}")


def save_partitions(
    partitions: List[List[int]],
    output_dir: str = "data/partitions",
    alpha: float = 0.5
):
    """Save partition indices to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    partition_data = {
        "alpha": alpha,
        "num_clients": len(partitions),
        "partitions": {f"client_{i}": indices 
                       for i, indices in enumerate(partitions)}
    }
    
    import json
    filepath = os.path.join(output_dir, f"partitions_alpha{alpha}.json")
    with open(filepath, "w") as f:
        json.dump(partition_data, f)
    
    logger.info(f"Partitions saved to {filepath}")
    
    # Also save per-client CSV indices
    for i, indices in enumerate(partitions):
        np.save(
            os.path.join(output_dir, f"client_{i}_alpha{alpha}.npy"),
            np.array(indices)
        )


def load_partitions(
    output_dir: str = "data/partitions",
    alpha: float = 0.5,
    num_clients: int = 10
) -> List[np.ndarray]:
    """Load partition indices from disk."""
    partitions = []
    for i in range(num_clients):
        filepath = os.path.join(output_dir, f"client_{i}_alpha{alpha}.npy")
        if os.path.exists(filepath):
            partitions.append(np.load(filepath))
        else:
            logger.warning(f"Partition file not found: {filepath}")
            partitions.append(np.array([]))
    return partitions


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition data for FL")
    parser.add_argument("--input", type=str, default="data/processed",
                        help="Directory with processed train.csv")
    parser.add_argument("--output", type=str, default="data/partitions")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-brfss-clients", type=int, default=5)
    parser.add_argument("--num-nhanes-clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration (0.1=non-IID, 10=IID)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", default=True)
    
    args = parser.parse_args()
    
    from src.utils import setup_logging
    setup_logging()
    
    # Load processed data
    train_path = os.path.join(args.input, "train.csv")
    if not os.path.exists(train_path):
        logger.error(f"Train data not found at {train_path}. Run data_prep.py first.")
        exit(1)
    
    df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(df)} training samples")
    
    # Partition
    partitions = partition_by_source(
        df,
        num_brfss_clients=args.num_brfss_clients,
        num_nhanes_clients=args.num_nhanes_clients,
        alpha=args.alpha,
        seed=args.seed
    )
    
    # Stats
    stats = compute_partition_stats(df, partitions)
    for cid, s in stats.items():
        logger.info(f"  {cid}: {s}")
    
    # Save
    save_partitions(partitions, args.output, args.alpha)
    
    # Visualize
    if args.visualize:
        visualize_partitions(df, partitions, save_path=f"results/partition_dist_alpha{args.alpha}.png")
