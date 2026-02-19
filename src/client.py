"""
Flower FL Client for privacy-preserving federated learning.

Each hospital node is a Flower NumPyClient that:
1. Loads its local data partition
2. Trains with DP-SGD (via Opacus) + FedProx proximal term
3. Applies Top-K gradient compression before sending updates
4. Tracks privacy budget via RDP accountant
5. Optionally computes local SHAP values
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.model import MultiTaskNCD, MultiTaskLoss, FedProxLoss
from src.compression import topk_compress, ErrorFeedbackCompressor
from src.privacy import PrivacyAccountant, setup_dp_training, get_epsilon

logger = logging.getLogger("ppfl-ncd.client")


class NCDClient(fl.client.NumPyClient):
    """
    Flower FL client for a single hospital node.
    
    Handles local training with:
    - DP-SGD (Opacus) for differential privacy
    - FedProx proximal term for non-IID robustness
    - Top-K compression for communication efficiency
    - SHAP computation for local explainability
    """
    
    def __init__(
        self,
        client_id: int,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        input_dim: int = 15,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        local_epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        # FedProx
        fedprox_mu: float = 0.01,
        # Privacy
        enable_dp: bool = True,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        dp_delta: float = 1e-5,
        max_epsilon: float = 3.0,
        # Compression
        enable_compression: bool = True,
        k_ratio: float = 0.1,
        # Device
        device: str = "auto",
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fedprox_mu = fedprox_mu
        self.enable_dp = enable_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.dp_delta = dp_delta
        self.enable_compression = enable_compression
        self.k_ratio = k_ratio
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create model
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.model = MultiTaskNCD(input_dim, hidden_dims, dropout).to(self.device)
        
        # Create data loaders
        X_train, Y_train = train_data
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(Y_train)
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # Required for Opacus
        )
        
        self.val_loader = None
        if val_data is not None:
            X_val, Y_val = val_data
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(Y_val)
            )
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        
        # Privacy
        self.privacy_engine = None
        self.privacy_accountant = PrivacyAccountant(
            noise_multiplier, max_grad_norm, dp_delta, max_epsilon
        )
        
        if self.enable_dp:
            try:
                self.model, self.optimizer, self.train_loader, self.privacy_engine = \
                    setup_dp_training(
                        self.model, self.optimizer, self.train_loader,
                        noise_multiplier, max_grad_norm, dp_delta
                    )
                logger.info(f"Client {client_id}: DP-SGD enabled (σ={noise_multiplier})")
            except Exception as e:
                logger.warning(f"Client {client_id}: DP setup failed: {e}. Running without DP.")
                self.enable_dp = False
        
        # Compression
        self.compressor = ErrorFeedbackCompressor(k_ratio) if enable_compression else None
        
        # Loss functions
        self.loss_fn = MultiTaskLoss()
        self.fedprox_loss_fn = FedProxLoss(mu=fedprox_mu)
        
        # Track global model parameters for FedProx
        self.global_params: Optional[List[torch.Tensor]] = None
        
        # Metrics
        self.round_metrics: Dict = {}
        
        logger.info(
            f"Client {client_id}: initialized | "
            f"samples={len(X_train)} | device={self.device} | "
            f"DP={'ON' if self.enable_dp else 'OFF'} | "
            f"compression={'ON' if enable_compression else 'OFF'}"
        )
    
    def get_parameters(self, config: Dict = None) -> NDArrays:
        """Return model parameters as NumPy arrays."""
        params = [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]
        
        # Apply compression if enabled
        if self.enable_compression and self.compressor is not None:
            params, comp_stats = self.compressor.compress(params)
            self.round_metrics["compression"] = comp_stats
        
        return params
    
    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)
        
        # Store as global params for FedProx
        self.global_params = [
            p.clone().detach().to(self.device)
            for p in self.model.parameters()
        ]
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.
        
        This is the core FL training step. Called by the Flower server
        each round.
        """
        # Set global model parameters
        self.set_parameters(parameters)
        
        # Get training config from server
        local_epochs = config.get("local_epochs", self.local_epochs)
        current_round = config.get("current_round", 0)
        
        # Check privacy budget
        if self.enable_dp and self.privacy_accountant.should_stop():
            logger.warning(f"Client {self.client_id}: Privacy budget exhausted. Skipping training.")
            return self.get_parameters(), len(self.train_dataset), {"status": "budget_exhausted"}
        
        # Train
        self.model.train()
        epoch_losses = []
        
        for epoch in range(local_epochs):
            batch_losses = []
            
            for batch_idx, (X_batch, Y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                # Split targets into per-task tensors
                targets = (
                    Y_batch[:, 0:1],  # diabetes
                    Y_batch[:, 1:2],  # hypertension
                    Y_batch[:, 2:3],  # CVD
                )
                
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                
                # FedProx loss (includes proximal term)
                if self.global_params is not None:
                    loss, per_task = self.fedprox_loss_fn(
                        preds, targets, self.model, self.global_params
                    )
                else:
                    loss, per_task = self.loss_fn(preds, targets)
                
                loss.backward()
                self.optimizer.step()
                batch_losses.append(per_task["total"])
            
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
        
        # Track privacy cost
        epsilon = 0.0
        if self.enable_dp and self.privacy_engine is not None:
            epsilon = get_epsilon(self.privacy_engine, self.dp_delta)
            self.privacy_accountant.record_round(epsilon, current_round)
        
        # Collect metrics
        avg_loss = float(np.mean(epoch_losses))
        self.round_metrics.update({
            "client_id": self.client_id,
            "round": current_round,
            "train_loss": avg_loss,
            "epsilon": epsilon,
            "num_samples": len(self.train_dataset),
        })
        
        metrics = {
            "train_loss": float(avg_loss),
            "epsilon": float(epsilon),
            "num_samples": len(self.train_dataset),
        }
        
        logger.info(
            f"Client {self.client_id} | Round {current_round} | "
            f"Loss={avg_loss:.4f} | ε={epsilon:.4f}"
        )
        
        return self.get_parameters(), len(self.train_dataset), metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate model on local validation data."""
        self.set_parameters(parameters)
        
        loader = self.val_loader or self.train_loader
        self.model.eval()
        
        total_loss = 0.0
        all_preds = [[], [], []]
        all_targets = [[], [], []]
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                targets = (Y_batch[:, 0:1], Y_batch[:, 1:2], Y_batch[:, 2:3])
                preds = self.model(X_batch)
                
                loss, _ = self.loss_fn(preds, targets)
                total_loss += loss.item()
                num_batches += 1
                
                for i in range(3):
                    all_preds[i].append(preds[i].cpu().numpy())
                    all_targets[i].append(targets[i].cpu().numpy())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute AUC-ROC per disease
        metrics = {"val_loss": float(avg_loss)}
        disease_names = ["diabetes", "hypertension", "cvd"]
        
        for i, name in enumerate(disease_names):
            y_true = np.concatenate(all_targets[i]).flatten()
            y_pred = np.concatenate(all_preds[i]).flatten()
            
            if len(np.unique(y_true)) > 1:
                from sklearn.metrics import roc_auc_score
                metrics[f"auc_{name}"] = float(roc_auc_score(y_true, y_pred))
            else:
                metrics[f"auc_{name}"] = 0.0
        
        return float(avg_loss), len(loader.dataset), metrics


def create_client_fn(
    train_data_partitions: List[Tuple[np.ndarray, np.ndarray]],
    val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    input_dim: int = 15,
    config: Optional[object] = None,
) -> callable:
    """
    Factory function to create a client_fn for Flower simulation.
    
    Args:
        train_data_partitions: List of (X, Y) tuples for each client
        val_data: Optional shared validation data
        input_dim: Number of input features
        config: ExperimentConfig instance
    
    Returns:
        client_fn callable for flwr.simulation
    """
    def client_fn(cid: str) -> NCDClient:
        client_id = int(cid)
        
        kwargs = {
            "client_id": client_id,
            "train_data": train_data_partitions[client_id],
            "val_data": val_data,
            "input_dim": input_dim,
        }
        
        if config is not None:
            kwargs.update({
                "hidden_dims": config.model.hidden_dims,
                "dropout": config.model.dropout,
                "local_epochs": config.fl.local_epochs,
                "batch_size": config.fl.batch_size,
                "learning_rate": config.fl.learning_rate,
                "fedprox_mu": config.fl.fedprox_mu,
                "enable_dp": config.privacy.enable_dp,
                "noise_multiplier": config.privacy.noise_multiplier,
                "max_grad_norm": config.privacy.max_grad_norm,
                "dp_delta": config.privacy.delta,
                "max_epsilon": config.privacy.max_epsilon,
                "enable_compression": config.compression.enable_compression,
                "k_ratio": config.compression.k_ratio,
                "device": config.device,
            })
        
        return NCDClient(**kwargs)
    
    return client_fn
