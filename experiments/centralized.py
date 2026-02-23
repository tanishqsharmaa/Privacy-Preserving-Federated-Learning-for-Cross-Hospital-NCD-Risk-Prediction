"""
Centralized training baseline.

Trains the multi-task NCD model on the ENTIRE pooled dataset (no FL, no privacy).
This is the theoretical UPPER BOUND on accuracy — the best possible result
when data is fully shared.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultiTaskNCD, MultiTaskLoss
from src.data_prep import prepare_dataset, HARMONIZED_FEATURES, TARGET_COLUMNS
from src.utils import (
    setup_logging, set_seed, compute_multitask_metrics,
    save_metrics, DISEASE_NAMES
)

logger = logging.getLogger("ppfl-ncd.centralized")


def train_centralized(
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_dims: list = None,
    dropout: float = 0.3,
    use_synthetic: bool = False,
    synthetic_samples: int = 50000,
    device: str = "auto",
    results_dir: str = "results/centralized",
    seed: int = 42,
):
    """
    Train centralized baseline model.
    
    This runs on ANY hardware (CPU is fine for centralized training).
    """
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Device: {device}")
    
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    # Load data
    train_df, test_df = prepare_dataset(
        use_synthetic=use_synthetic,
        synthetic_samples=synthetic_samples,
        seed=seed,
    )
    
    X_train = train_df[HARMONIZED_FEATURES].values.astype(np.float32)
    Y_train = train_df[TARGET_COLUMNS].values.astype(np.float32)
    X_test = test_df[HARMONIZED_FEATURES].values.astype(np.float32)
    Y_test = test_df[TARGET_COLUMNS].values.astype(np.float32)
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    Y_train = np.nan_to_num(Y_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    Y_test = np.nan_to_num(Y_test, nan=0.0)
    
    input_dim = X_train.shape[1]
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples:     {len(X_test)}")
    logger.info(f"Input dimension:  {input_dim}")
    
    # Create model
    model = MultiTaskNCD(input_dim, hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = MultiTaskLoss()
    
    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    history = {"train_loss": [], "test_metrics": []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            targets = (Y_batch[:, 0:1], Y_batch[:, 1:2], Y_batch[:, 2:3])
            preds = model(X_batch)
            
            loss, per_task = loss_fn(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(per_task["total"])
        
        avg_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(avg_loss)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_test).to(device)
                preds = model(X_test_t)
                preds_np = [p.cpu().numpy().flatten() for p in preds]
            
            y_true_list = [Y_test[:, i] for i in range(3)]
            metrics = compute_multitask_metrics(y_true_list, preds_np)
            history["test_metrics"].append({"epoch": epoch + 1, "metrics": metrics})
            
            macro_auc = metrics["macro_avg"]["auc_roc"]
            logger.info(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss={avg_loss:.4f} | "
                f"AUC: D={metrics['diabetes']['auc_roc']:.4f} "
                f"H={metrics['hypertension']['auc_roc']:.4f} "
                f"C={metrics['cardiovascular_disease']['auc_roc']:.4f} "
                f"Macro={macro_auc:.4f}"
            )
    
    # Save model and results
    torch.save(model.state_dict(), os.path.join(results_dir, "centralized_model.pth"))
    save_metrics(history, os.path.join(results_dir, "centralized_results.json"))
    
    # Final metrics
    final_metrics = history["test_metrics"][-1]["metrics"]
    logger.info("\n=== CENTRALIZED BASELINE RESULTS ===")
    for disease in DISEASE_NAMES:
        m = final_metrics[disease]
        logger.info(f"  {disease}: AUC={m['auc_roc']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")
    logger.info(f"  Macro Avg: AUC={final_metrics['macro_avg']['auc_roc']:.4f}")
    
    return model, final_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=50000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    setup_logging("results/centralized")
    
    train_centralized(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_synthetic=args.synthetic,
        synthetic_samples=args.synthetic_samples,
        device=args.device,
        seed=args.seed,
    )
