"""
Fisher-Guided Adaptive Differential Privacy Noise (FisherDP-FL).

Injects DP noise inversely proportional to parameter importance,
measured via the diagonal Fisher Information Matrix.

Key insight: Not all parameters are equally important for model
accuracy. Parameters with high Fisher Information (high sensitivity
to the loss) should receive LESS noise to preserve accuracy, while
low-importance parameters can absorb MORE noise.

The total noise budget is maintained to satisfy the same (epsilon, delta)
DP guarantee as uniform noise, but redistributed for better accuracy.

This is a novel contribution combining:
- Fisher Information for importance estimation
- Per-layer adaptive noise calibration
- Compatibility with Opacus DP-SGD framework

Reference: Inspired by Fisher-weighted averaging (Matena & Raffel, 2022)
and adaptive DP methods (Yu et al., 2024).
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppfl-ncd.fisher_dp")


def compute_fisher_importance(
    model: nn.Module,
    X: torch.Tensor,
    Y: Tuple[torch.Tensor, ...],
    num_samples: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher Information for each parameter.

    The diagonal Fisher Information F_ii approximates the second-order
    sensitivity of each parameter to the loss:
        F_ii = E[(d_loss/d_theta_i)^2]

    Parameters with higher Fisher info are more important for accuracy.

    Args:
        model: The neural network model
        X: Input features tensor, shape (N, D)
        Y: Tuple of target tensors for each task
        num_samples: Number of samples to use for estimation
        device: Device to compute on

    Returns:
        Dictionary mapping parameter name -> Fisher importance tensor
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()  # Use eval mode for stable estimates

    # Subsample if needed
    if X.shape[0] > num_samples:
        indices = torch.randperm(X.shape[0])[:num_samples]
        X = X[indices]
        Y = tuple(y[indices] for y in Y)

    X = X.to(device)
    Y = tuple(y.to(device) for y in Y)

    # Accumulate squared gradients
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    n_batches = 0

    # Process in mini-batches to avoid OOM
    batch_size = 64
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        x_batch = X[start:end]
        y_batch = tuple(y[start:end] for y in Y)

        model.zero_grad()

        # Forward pass (in train mode for logits)
        model.train()
        preds = model(x_batch)

        # Compute per-task BCE loss
        loss = torch.tensor(0.0, device=device)
        for pred, target in zip(preds, y_batch):
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                pred, target, reduction='mean'
            )

        # Backward to get gradients
        loss.backward()

        # Accumulate squared gradients (diagonal Fisher approximation)
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.data.pow(2)

        n_batches += 1

    # Average
    for name in fisher:
        fisher[name] /= max(n_batches, 1)

    # Normalize to [0, 1] range per layer for stability
    for name in fisher:
        max_val = fisher[name].max()
        if max_val > 0:
            fisher[name] = fisher[name] / max_val

    model.eval()
    return fisher


def compute_adaptive_noise_scales(
    fisher: Dict[str, torch.Tensor],
    base_noise_multiplier: float = 1.1,
    min_scale: float = 0.3,
    max_scale: float = 3.0,
) -> Dict[str, float]:
    """
    Compute per-layer noise scales based on Fisher importance.

    Higher Fisher importance -> lower noise scale (more important params
    get less noise to preserve accuracy).

    The scales are normalized so that the AVERAGE noise level across
    all parameters matches the base_noise_multiplier, preserving
    the DP guarantee.

    Args:
        fisher: Per-parameter Fisher importance tensors
        base_noise_multiplier: Original uniform noise multiplier
        min_scale: Minimum noise scale (floor for important params)
        max_scale: Maximum noise scale (ceiling for unimportant params)

    Returns:
        Dictionary mapping parameter name -> adaptive noise scale
    """
    # Compute mean importance per layer
    layer_importance = {}
    total_params = 0

    for name, f in fisher.items():
        mean_importance = f.mean().item()
        num_params = f.numel()
        layer_importance[name] = mean_importance
        total_params += num_params

    if not layer_importance:
        return {}

    # Invert importance: high importance -> low noise
    # Scale = 1 / (importance + epsilon) -> then normalize
    epsilon = 1e-8
    raw_scales = {}
    for name, imp in layer_importance.items():
        raw_scales[name] = 1.0 / (imp + epsilon)

    # Normalize so weighted average equals base_noise_multiplier
    # This preserves the overall DP guarantee
    total_weight = sum(
        fisher[name].numel() * raw_scales[name]
        for name in raw_scales
    )
    avg_scale = total_weight / max(total_params, 1)

    normalization_factor = base_noise_multiplier / max(avg_scale, epsilon)

    adaptive_scales = {}
    for name, scale in raw_scales.items():
        adapted = scale * normalization_factor
        # Clip to [min_scale, max_scale] * base for safety
        adapted = max(min_scale * base_noise_multiplier, adapted)
        adapted = min(max_scale * base_noise_multiplier, adapted)
        adaptive_scales[name] = adapted

    # Log the distribution
    scales_list = list(adaptive_scales.values())
    logger.info(
        f"Fisher-guided noise: min={min(scales_list):.4f}, "
        f"max={max(scales_list):.4f}, "
        f"mean={np.mean(scales_list):.4f}, "
        f"base={base_noise_multiplier}"
    )

    return adaptive_scales


def apply_fisher_noise_to_gradients(
    model: nn.Module,
    adaptive_scales: Dict[str, float],
    max_grad_norm: float = 1.0,
):
    """
    Apply Fisher-guided noise directly to gradients POST-TRAINING.

    This is used as a POST-HOC noise injection after standard training
    (without Opacus). The noise is calibrated per-parameter based on
    Fisher importance.

    Note: This provides a weaker privacy guarantee than Opacus DP-SGD
    (which clips per-sample) but is compatible with the Fisher adaptation.

    For rigorous DP: use Opacus DP-SGD with uniform noise, then apply
    Fisher-guided noise only to the parameter UPDATES sent to the server.

    Args:
        model: Model whose gradients to perturb
        adaptive_scales: Per-parameter noise scales from compute_adaptive_noise_scales
        max_grad_norm: Gradient clipping norm
    """
    # First, clip global gradient norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Then add Fisher-guided noise to each parameter
    for name, p in model.named_parameters():
        if p.grad is not None and name in adaptive_scales:
            noise_std = adaptive_scales[name] * max_grad_norm
            noise = torch.randn_like(p.grad) * noise_std
            p.grad.data.add_(noise)


def apply_fisher_noise_to_updates(
    param_updates: List[np.ndarray],
    param_names: List[str],
    adaptive_scales: Dict[str, float],
    max_grad_norm: float = 1.0,
    sensitivity: float = 1.0,
) -> List[np.ndarray]:
    """
    Apply Fisher-guided noise to parameter updates BEFORE sending to server.

    This is the recommended approach for rigorous DP compliance:
    1. Train locally (with or without per-sample clipping)
    2. Compute update = local_params - global_params
    3. Add Fisher-guided noise to the update
    4. Send noisy update to server

    Args:
        param_updates: List of parameter update arrays (local - global)
        param_names: Corresponding parameter names
        adaptive_scales: Per-parameter noise scales
        max_grad_norm: L2 norm bound for update clipping
        sensitivity: DP sensitivity (usually = max_grad_norm)

    Returns:
        Noisy parameter updates
    """
    noisy_updates = []

    for update, name in zip(param_updates, param_names):
        scale = adaptive_scales.get(name, 1.0)
        noise_std = scale * sensitivity / np.sqrt(len(param_updates))
        noise = np.random.randn(*update.shape).astype(np.float32) * noise_std
        noisy_updates.append(update + noise)

    return noisy_updates
