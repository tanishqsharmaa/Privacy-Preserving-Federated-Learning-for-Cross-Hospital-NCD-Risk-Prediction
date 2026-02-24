"""
Knowledge Distillation for Federated Learning (KD-FL).

Implements ensemble distillation where:
1. Server generates a balanced proxy dataset (synthetic)
2. Clients predict soft labels on the proxy dataset
3. Server averages soft labels -> consensus distribution
4. Consensus is sent back to clients as a "teacher"
5. Clients train with combined task loss + KD loss

KD Loss = alpha * CE(student, true_labels) + (1-alpha) * KL(student, teacher)

This allows knowledge transfer without sharing raw data, improving
performance especially on minority classes where individual clients
may have insufficient examples.

Reference: Lin et al., "Ensemble Distillation for Robust Model Fusion
in Federated Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppfl-ncd.distillation")


class KDLoss(nn.Module):
    """
    Knowledge Distillation loss for federated multi-task learning.

    Combines the standard task loss with a KL-divergence loss
    between the local model's predictions and the consensus
    (averaged teacher) predictions.

    L = alpha * L_task + (1-alpha) * T^2 * KL(softmax(z_s/T), softmax(z_t/T))
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        """
        Args:
            temperature: Softening temperature (higher = softer distributions)
            alpha: Weight of task loss vs KD loss (1.0 = pure task loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: Tuple[torch.Tensor, ...],
        teacher_logits: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Compute KD loss between student and teacher logits.

        For binary classification, we use sigmoid + binary KL divergence.

        Args:
            student_logits: Tuple of student model logits per task
            teacher_logits: Tuple of teacher (consensus) logits per task

        Returns:
            KD loss (to be combined with task loss externally)
        """
        kd_loss = torch.tensor(0.0, device=student_logits[0].device)

        for s_logit, t_logit in zip(student_logits, teacher_logits):
            # Soft probabilities with temperature
            s_prob = torch.sigmoid(s_logit / self.temperature)
            t_prob = torch.sigmoid(t_logit / self.temperature)

            # Binary KL divergence
            # KL(t || s) = t * log(t/s) + (1-t) * log((1-t)/(1-s))
            eps = 1e-7
            t_prob = t_prob.clamp(eps, 1 - eps)
            s_prob = s_prob.clamp(eps, 1 - eps)

            kl = t_prob * torch.log(t_prob / s_prob) + \
                 (1 - t_prob) * torch.log((1 - t_prob) / (1 - s_prob))

            kd_loss = kd_loss + kl.mean()

        # Scale by T^2 as per Hinton et al.
        kd_loss = kd_loss * (self.temperature ** 2) / len(student_logits)

        return kd_loss


def create_proxy_dataset(
    num_samples: int = 1000,
    input_dim: int = 15,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create a balanced synthetic proxy dataset for KD.

    The proxy dataset is generated from a standard normal distribution.
    It's designed to cover the feature space so that consensus soft labels
    are informative across the input range.

    Args:
        num_samples: Number of proxy samples
        input_dim: Feature dimension
        seed: Random seed

    Returns:
        Tensor of shape (num_samples, input_dim)
    """
    rng = np.random.RandomState(seed)
    # Mix of uniform and normal to cover feature space
    half = num_samples // 2
    normal_part = rng.randn(half, input_dim).astype(np.float32)
    uniform_part = rng.uniform(-2, 2, (num_samples - half, input_dim)).astype(np.float32)
    X = np.concatenate([normal_part, uniform_part], axis=0)

    return torch.FloatTensor(X)


def collect_soft_labels(
    model: nn.Module,
    proxy_data: torch.Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
) -> Tuple[np.ndarray, ...]:
    """
    Collect soft label predictions from a model on the proxy dataset.

    Returns raw logits (not probabilities) so temperature scaling
    can be applied later.

    Args:
        model: The local model
        proxy_data: Proxy dataset tensor
        device: Compute device
        batch_size: Batch size for inference

    Returns:
        Tuple of 3 numpy arrays (logits per task)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_logits = [[], [], []]

    with torch.no_grad():
        for start in range(0, len(proxy_data), batch_size):
            end = min(start + batch_size, len(proxy_data))
            X_batch = proxy_data[start:end].to(device)

            # Get train-mode logits (not sigmoid)
            model.train()
            preds = model(X_batch)
            model.eval()

            for i, p in enumerate(preds):
                all_logits[i].append(p.cpu().numpy())

    return tuple(np.concatenate(task_logits) for task_logits in all_logits)


def aggregate_soft_labels(
    client_logits: List[Tuple[np.ndarray, ...]],
    client_weights: Optional[List[float]] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Average soft labels from multiple clients to create consensus.

    Args:
        client_logits: List of client soft label tuples
        client_weights: Optional weights per client (default: equal)

    Returns:
        Consensus soft labels as tuple of numpy arrays
    """
    if not client_logits:
        raise ValueError("No client logits to aggregate")

    n_clients = len(client_logits)
    n_tasks = len(client_logits[0])

    if client_weights is None:
        client_weights = [1.0 / n_clients] * n_clients
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    consensus = []
    for task_idx in range(n_tasks):
        weighted_sum = np.zeros_like(client_logits[0][task_idx])
        for client_idx, logits in enumerate(client_logits):
            weighted_sum += client_weights[client_idx] * logits[task_idx]
        consensus.append(weighted_sum)

    return tuple(consensus)


def serialize_soft_labels(soft_labels: Tuple[np.ndarray, ...]) -> List[np.ndarray]:
    """Convert soft labels for Flower parameter transport."""
    return list(soft_labels)


def deserialize_soft_labels(arrays: List[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """Reconstruct soft labels from Flower parameter transport."""
    return tuple(arrays)
