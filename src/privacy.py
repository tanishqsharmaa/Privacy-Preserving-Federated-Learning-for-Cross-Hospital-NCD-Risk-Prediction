"""
Differential Privacy utilities for the FL system.

Provides:
- Opacus DP-SGD configuration helpers
- RDP (Rényi Differential Privacy) accounting wrapper
- Privacy budget enforcement
- Noise calibration experiment utilities
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppfl-ncd.privacy")


class PrivacyAccountant:
    """
    Wrapper around Opacus RDP accountant for tracking cumulative
    privacy loss across FL rounds.
    """
    
    def __init__(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        max_epsilon: float = 3.0,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.max_epsilon = max_epsilon
        
        self.history: List[Dict] = []
        self.current_epsilon = 0.0
        self.budget_exceeded = False
    
    def record_round(
        self,
        epsilon: float,
        fl_round: int,
        num_steps: int = 0,
    ):
        """Record privacy cost for one FL round."""
        self.current_epsilon = epsilon
        self.history.append({
            "round": fl_round,
            "epsilon": epsilon,
            "delta": self.delta,
            "num_steps": num_steps,
            "noise_multiplier": self.noise_multiplier,
        })
        
        if epsilon >= self.max_epsilon:
            self.budget_exceeded = True
            logger.warning(
                f"Privacy budget EXCEEDED at round {fl_round}: "
                f"eps={epsilon:.4f} >= max_eps={self.max_epsilon}"
            )
    
    def should_stop(self) -> bool:
        """Check if training should stop due to budget exhaustion."""
        return self.budget_exceeded
    
    def get_privacy_report(self) -> Dict:
        """Generate a summary report of privacy spending."""
        return {
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "delta": self.delta,
            "max_epsilon": self.max_epsilon,
            "final_epsilon": self.current_epsilon,
            "budget_exceeded": self.budget_exceeded,
            "num_rounds_completed": len(self.history),
            "epsilon_per_round": [h["epsilon"] for h in self.history],
        }


def setup_dp_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
) -> Tuple:
    """
    Wrap model/optimizer/dataloader with Opacus DP-SGD.
    
    ⚠️ This modifies the model in-place (replaces BatchNorm with GroupNorm).
    ⚠️ This is SLOW — use GPU machine for actual training.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        data_loader: Training DataLoader
        noise_multiplier: σ for Gaussian noise (higher = more privacy)
        max_grad_norm: Per-sample gradient clipping bound
        delta: δ parameter for (ε, δ)-DP
    
    Returns:
        (dp_model, dp_optimizer, dp_data_loader, privacy_engine)
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        
        # Fix incompatible layers (BatchNorm → GroupNorm)
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            logger.info("Fixed model for Opacus compatibility (BatchNorm -> GroupNorm)")
        
        privacy_engine = PrivacyEngine(accountant="rdp")
        
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        
        logger.info(
            f"DP-SGD enabled: sigma={noise_multiplier}, C={max_grad_norm}, delta={delta}"
        )
        
        return model, optimizer, data_loader, privacy_engine
        
    except ImportError:
        logger.error(
            "Opacus not installed. Install with: pip install opacus>=1.4.0"
        )
        raise


def get_epsilon(privacy_engine, delta: float = 1e-5) -> float:
    """Get current epsilon from Opacus privacy engine."""
    try:
        return privacy_engine.get_epsilon(delta=delta)
    except Exception as e:
        logger.warning(f"Could not compute epsilon: {e}")
        return float("inf")


def estimate_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    delta: float = 1e-5,
) -> float:
    """
    Estimate epsilon using RDP accounting without running training.
    
    Useful for planning noise_multiplier before actual experiments.
    
    Args:
        noise_multiplier: σ for Gaussian noise
        sample_rate: batch_size / dataset_size
        num_steps: Total number of gradient steps
        delta: δ parameter
    
    Returns:
        Estimated epsilon
    """
    try:
        from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
        
        # RDP orders
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=noise_multiplier,
            steps=num_steps,
            orders=orders,
        )
        
        epsilon, best_alpha = get_privacy_spent(
            orders=orders, rdp=rdp, delta=delta
        )
        
        return epsilon
        
    except ImportError:
        logger.warning("Opacus RDP analysis not available. Returning estimate.")
        # Rough approximation
        return (2 * num_steps * sample_rate ** 2) / (noise_multiplier ** 2)


def noise_calibration_report(
    dataset_size: int,
    batch_size: int,
    num_rounds: int,
    local_epochs: int,
    noise_multipliers: List[float] = None,
    delta: float = 1e-5,
) -> List[Dict]:
    """
    Run noise calibration analysis for different noise_multiplier values.
    
    Args:
        dataset_size: Approximate size of each client's dataset
        batch_size: Training batch size
        num_rounds: Number of FL rounds
        local_epochs: Local epochs per round
        noise_multipliers: List of σ values to test
        delta: δ parameter
    
    Returns:
        List of dicts with noise_multiplier, estimated_epsilon, etc.
    """
    if noise_multipliers is None:
        noise_multipliers = [0.5, 0.8, 1.1, 1.5, 2.0]
    
    sample_rate = min(batch_size / max(dataset_size, 1), 1.0)
    steps_per_round = max(1, (dataset_size // batch_size)) * local_epochs
    total_steps = steps_per_round * num_rounds
    
    results = []
    for sigma in noise_multipliers:
        eps = estimate_epsilon(sigma, sample_rate, total_steps, delta)
        results.append({
            "noise_multiplier": sigma,
            "estimated_epsilon": eps,
            "delta": delta,
            "total_steps": total_steps,
            "sample_rate": sample_rate,
            "within_budget": eps <= 3.0,
        })
        logger.info(
            f"  sigma={sigma:.1f} -> eps~{eps:.4f} "
            f"({'within budget' if eps <= 3.0 else 'EXCEEDS budget'})"
        )
    
    return results
