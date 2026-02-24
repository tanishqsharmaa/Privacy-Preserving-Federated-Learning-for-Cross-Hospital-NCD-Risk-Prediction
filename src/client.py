"""
Flower FL Client for privacy-preserving federated learning.

Each hospital node is a Flower NumPyClient that:
1. Loads its local data partition
2. Trains with DP-SGD (via Opacus) + FedProx proximal term
3. Applies Top-K gradient compression before sending updates
4. Tracks privacy budget via RDP accountant
5. Reports per-task class ratios for class-aware aggregation (Rec 1)
6. Uses WeightedRandomSampler for local oversampling (Issue #1 fix)
7. Receives server-coordinated LR per round (Issue #4 fix)
8. Properly handles Opacus model modifications (Issue #7 fix)

Key fixes from original:
  - Removed broken per-client CosineAnnealingLR (Issue #4)
  - Uses GroupNorm model (no Opacus param mismatch, Issue #7)
  - WeightedRandomSampler oversamples minority class (Issue #1)
  - Reports class ratios to server for class-aware aggregation (Rec 1)
  - Uses FocalLoss with dynamic pos_weight (Phase 2)
  - Receives learning_rate from server per round (Phase 1)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.model import MultiTaskNCD, MultiTaskLoss, FedProxLoss
from src.compression import topk_compress, ErrorFeedbackCompressor
from src.privacy import PrivacyAccountant, setup_dp_training, get_epsilon
from src.config import get_best_device

logger = logging.getLogger("ppfl-ncd.client")


def compute_class_stats(Y: np.ndarray) -> Dict[str, float]:
    """
    Compute class statistics for a multi-label dataset.

    Args:
        Y: Target array of shape (N, 3) for [diabetes, hypertension, CVD]

    Returns:
        Dict with per-task positive counts, negative counts, and pos_weight
    """
    task_names = ["diabetes", "hypertension", "cvd"]
    stats = {}
    for i, name in enumerate(task_names):
        n_pos = max(float(Y[:, i].sum()), 1.0)  # Avoid division by zero
        n_neg = max(float(len(Y) - n_pos), 1.0)
        stats[f"n_pos_{name}"] = n_pos
        stats[f"n_neg_{name}"] = n_neg
        stats[f"pos_weight_{name}"] = n_neg / n_pos
        stats[f"prevalence_{name}"] = n_pos / len(Y)
    return stats


def make_weighted_sampler(Y: np.ndarray) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that oversamples minority-class examples.

    Uses a composite weight: for each sample, the weight is the average
    of the pos_weight for each positive task it has. This helps balance
    all three tasks simultaneously.
    """
    n_samples = len(Y)
    stats = compute_class_stats(Y)

    # Compute per-sample weight as the average inverse frequency across tasks
    sample_weights = np.ones(n_samples, dtype=np.float64)
    task_names = ["diabetes", "hypertension", "cvd"]

    for i, name in enumerate(task_names):
        pw = stats[f"pos_weight_{name}"]
        # Positive samples get weight = pos_weight, negatives get weight = 1
        task_weight = np.where(Y[:, i] == 1, pw, 1.0)
        sample_weights *= task_weight

    # Normalize to avoid extreme weights
    sample_weights = np.sqrt(sample_weights)  # dampen extreme ratios
    sample_weights = sample_weights / sample_weights.sum() * n_samples

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=n_samples,
        replacement=True,
    )


class NCDClient(fl.client.NumPyClient):
    """
    Flower FL client for a single hospital node.

    Handles local training with:
    - Focal Loss / Weighted BCE with dynamic pos_weight
    - WeightedRandomSampler for local oversampling
    - DP-SGD (Opacus) for differential privacy
    - FedProx proximal term for non-IID robustness
    - Top-K compression for communication efficiency
    - Server-coordinated learning rate per round
    - Per-task class ratio reporting for class-aware aggregation
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
        learning_rate: float = 0.0003,
        # FedProx
        fedprox_mu: float = 0.1,
        # Loss
        loss_type: str = 'focal',
        focal_gamma: float = 2.0,
        # Privacy
        enable_dp: bool = True,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        dp_delta: float = 1e-5,
        max_epsilon: float = 3.0,
        # Compression
        enable_compression: bool = True,
        k_ratio: float = 0.3,
        # Model
        norm_type: str = 'group',
        # Sampling
        use_weighted_sampling: bool = True,
        # Device
        device: str = "auto",
    ):
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.base_learning_rate = learning_rate
        self.fedprox_mu = fedprox_mu
        self.enable_dp = enable_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.dp_delta = dp_delta
        self.enable_compression = enable_compression
        self.k_ratio = k_ratio
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

        # Device
        if device == "auto":
            self.device = get_best_device()
        else:
            self.device = torch.device(device)

        # Create model with GroupNorm (Opacus-compatible from the start)
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.model = MultiTaskNCD(
            input_dim, hidden_dims, dropout, norm_type=norm_type
        ).to(self.device)

        # Store the model key order BEFORE any Opacus modifications
        self._original_model_keys = list(self.model.state_dict().keys())

        # Compute class statistics and pos_weights from local data
        X_train, Y_train = train_data
        self.class_stats = compute_class_stats(Y_train)
        self.pos_weights = (
            self.class_stats["pos_weight_diabetes"],
            self.class_stats["pos_weight_hypertension"],
            self.class_stats["pos_weight_cvd"],
        )

        # Create data loaders with optional weighted sampling
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(Y_train)
        )

        if use_weighted_sampling and len(X_train) > 0:
            sampler = make_weighted_sampler(Y_train)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=min(batch_size, len(X_train)),
                sampler=sampler,
                drop_last=True,  # Required for Opacus
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=min(batch_size, len(X_train)),
                shuffle=True,
                drop_last=True,
            )

        self.val_loader = None
        if val_data is not None:
            X_val, Y_val = val_data
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(Y_val)
            )
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer — NO per-client scheduler (LR comes from server each round)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # Privacy — setup DP-SGD via Opacus
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
                # After Opacus wrapping, rebuild key list
                self._dp_model_keys = list(self.model.state_dict().keys())
                logger.info(f"Client {client_id}: DP-SGD enabled (sigma={noise_multiplier})")
            except Exception as e:
                logger.warning(f"Client {client_id}: DP setup failed: {e}. Running without DP.")
                self.enable_dp = False
                self._dp_model_keys = self._original_model_keys

        # Compression
        self.compressor = ErrorFeedbackCompressor(k_ratio) if enable_compression else None

        # Loss functions — with local pos_weights for class balancing
        self.loss_fn = MultiTaskLoss(
            pos_weights=self.pos_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )
        self.fedprox_loss_fn = FedProxLoss(
            mu=fedprox_mu,
            pos_weights=self.pos_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )

        # Track global model parameters for FedProx proximal term
        self.global_params: Optional[List[torch.Tensor]] = None
        self._global_state_np: Optional[List[np.ndarray]] = None

        # Metrics
        self.round_metrics: Dict = {}

        logger.info(
            f"Client {client_id}: initialized | "
            f"samples={len(X_train)} | device={self.device} | "
            f"DP={'ON' if self.enable_dp else 'OFF'} | "
            f"compression={'ON' if enable_compression else 'OFF'} | "
            f"loss={loss_type} | "
            f"pos_weights=({self.pos_weights[0]:.1f}, {self.pos_weights[1]:.1f}, {self.pos_weights[2]:.1f})"
        )

    def get_parameters(self, config: Dict = None) -> NDArrays:
        """Return model parameters as NumPy arrays, optionally compressed."""
        params = [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

        # Apply compression on weight deltas if enabled
        if self.enable_compression and self.compressor is not None and self._global_state_np is not None:
            if len(params) == len(self._global_state_np) and all(
                p.shape == g.shape for p, g in zip(params, self._global_state_np)
            ):
                deltas = [p - g for p, g in zip(params, self._global_state_np)]
                compressed_deltas, comp_stats = self.compressor.compress(deltas)
                params = [g + d for g, d in zip(self._global_state_np, compressed_deltas)]
                self.round_metrics["compression"] = comp_stats
            else:
                logger.warning(
                    f"Client {self.client_id}: shape mismatch, skipping compression."
                )

        return params

    def set_parameters(self, parameters: NDArrays):
        """
        Set model parameters from NumPy arrays.

        Handles Opacus model modifications by mapping parameters
        based on the current model's state dict keys.
        """
        model_keys = list(self.model.state_dict().keys())

        if len(parameters) != len(model_keys):
            # Try to load only matching parameters (partial load)
            logger.warning(
                f"Client {self.client_id}: param count mismatch "
                f"(received {len(parameters)}, model has {len(model_keys)}). "
                f"Attempting partial parameter mapping..."
            )
            # Map by name prefix matching between original and DP model
            orig_keys = self._original_model_keys
            if len(parameters) == len(orig_keys):
                # Server sent original model params; map to DP model keys
                current_state = self.model.state_dict()
                new_state = OrderedDict()
                orig_to_param = dict(zip(orig_keys, parameters))

                for key in model_keys:
                    # Try direct match
                    if key in orig_to_param:
                        new_state[key] = torch.from_numpy(orig_to_param[key]).float()
                    else:
                        # Try stripping DP prefix (e.g., "_module.xxx" -> "xxx")
                        stripped = key.replace("_module.", "")
                        if stripped in orig_to_param:
                            new_state[key] = torch.from_numpy(orig_to_param[stripped]).float()
                        else:
                            # Keep current parameter
                            new_state[key] = current_state[key]

                self.model.load_state_dict(new_state, strict=False)
            else:
                logger.warning(
                    f"Client {self.client_id}: cannot map params. "
                    f"Using current model weights."
                )
        else:
            params_dict = zip(model_keys, parameters)
            state_dict = OrderedDict(
                {k: torch.from_numpy(v).float() for k, v in params_dict}
            )
            self.model.load_state_dict(state_dict, strict=True)

        # Store global params for FedProx proximal term
        self.global_params = [
            p.clone().detach().to(self.device)
            for p in self.model.parameters()
        ]

        # Store global state as numpy for compression delta computation
        self._global_state_np = [
            val.cpu().numpy().copy()
            for val in self.model.state_dict().values()
        ]

    def _update_learning_rate(self, lr: float):
        """Update optimizer learning rate (server-coordinated schedule)."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data.

        Receives learning_rate from server (server-coordinated schedule).
        Reports per-task class ratios for class-aware aggregation.
        """
        # Set global model parameters
        self.set_parameters(parameters)

        # Get training config from server
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        current_round = int(config.get("current_round", 0))

        # Server-coordinated learning rate (fixes Issue #4)
        server_lr = config.get("learning_rate", self.base_learning_rate)
        if isinstance(server_lr, (int, float)):
            self._update_learning_rate(float(server_lr))

        # Check privacy budget
        if self.enable_dp and self.privacy_accountant.should_stop():
            logger.warning(f"Client {self.client_id}: Privacy budget exhausted.")
            return self.get_parameters(), len(self.train_dataset), {"status": "budget_exhausted"}

        # Train
        self.model.train()
        epoch_losses = []

        for epoch in range(local_epochs):
            batch_losses = []

            for batch_idx, (X_batch, Y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                targets = (
                    Y_batch[:, 0:1],  # diabetes
                    Y_batch[:, 1:2],  # hypertension
                    Y_batch[:, 2:3],  # CVD
                )

                self.optimizer.zero_grad()
                preds = self.model(X_batch)

                # FedProx loss (includes proximal term + focal/weighted BCE)
                if self.global_params is not None:
                    loss, per_task = self.fedprox_loss_fn(
                        preds, targets, self.model, self.global_params
                    )
                else:
                    loss, per_task = self.loss_fn(preds, targets)

                loss.backward()
                self.optimizer.step()
                batch_losses.append(per_task["total"])

            epoch_loss = np.mean(batch_losses) if batch_losses else 0.0
            epoch_losses.append(epoch_loss)

        # Track privacy cost
        epsilon = 0.0
        if self.enable_dp and self.privacy_engine is not None:
            epsilon = get_epsilon(self.privacy_engine, self.dp_delta)
            self.privacy_accountant.record_round(epsilon, current_round)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Build metrics — include class stats for class-aware aggregation
        metrics = {
            "train_loss": float(avg_loss),
            "epsilon": float(epsilon),
            "num_samples": len(self.train_dataset),
            "client_id": float(self.client_id),
            # Per-task class stats for server-side class-aware aggregation (Rec 1)
            "n_pos_diabetes": self.class_stats["n_pos_diabetes"],
            "n_pos_hypertension": self.class_stats["n_pos_hypertension"],
            "n_pos_cvd": self.class_stats["n_pos_cvd"],
            "n_neg_diabetes": self.class_stats["n_neg_diabetes"],
            "n_neg_hypertension": self.class_stats["n_neg_hypertension"],
            "n_neg_cvd": self.class_stats["n_neg_cvd"],
        }

        logger.info(
            f"Client {self.client_id} | Round {current_round} | "
            f"Loss={avg_loss:.4f} | eps={epsilon:.4f} | "
            f"lr={server_lr:.6f}"
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

        # Use base loss_fn (no FedProx proximal for eval)
        eval_loss_fn = MultiTaskLoss(
            pos_weights=self.pos_weights,
            loss_type=self.loss_type,
            focal_gamma=self.focal_gamma,
        )

        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                targets = (Y_batch[:, 0:1], Y_batch[:, 1:2], Y_batch[:, 2:3])
                preds = self.model(X_batch)

                loss, _ = eval_loss_fn(preds, targets)
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
                "loss_type": getattr(config.model, 'loss_type', 'focal'),
                "focal_gamma": getattr(config.model, 'focal_gamma', 2.0),
                "norm_type": getattr(config.model, 'norm_type', 'group'),
                "use_weighted_sampling": getattr(config.fl, 'use_weighted_sampling', True),
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
