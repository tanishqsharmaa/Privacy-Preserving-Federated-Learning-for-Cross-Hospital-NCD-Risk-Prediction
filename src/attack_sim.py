"""
Attack simulation for privacy validation.

Implements two attacks to empirically validate DP defense:
1. Gradient Inversion Attack — reconstruct training data from gradients
2. Membership Inference Attack — determine if a sample was in training

Shows attacks succeed without DP and fail with DP.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ppfl-ncd.attack")


# ============================================================================
# GRADIENT INVERSION ATTACK
# ============================================================================

class GradientInversionAttack:
    """
    Reconstruct training data from gradient updates.
    
    Given a model's gradient (computed on some training batch), the
    attacker tries to optimize a dummy input that produces the same
    gradient. If successful, this reveals the training data.
    
    With DP noise, the gradients are noisy and reconstruction fails.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        num_iterations: int = 300,
        lr: float = 0.1,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.num_iterations = num_iterations
        self.lr = lr
    
    def compute_gradient(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute gradient of the model on given data."""
        model.train()
        model.zero_grad()
        
        preds = model(x)
        targets = (y[:, 0:1], y[:, 1:2], y[:, 2:3])
        
        loss = sum(
            nn.BCELoss()(p, t)
            for p, t in zip(preds, targets)
        )
        loss.backward()
        
        return [p.grad.clone().detach() for p in model.parameters() if p.grad is not None]
    
    def add_dp_noise(
        self,
        gradients: List[torch.Tensor],
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> List[torch.Tensor]:
        """Simulate DP-SGD noise addition."""
        noisy = []
        for g in gradients:
            # Clip
            norm = torch.norm(g)
            g_clipped = g * min(1, max_grad_norm / (norm + 1e-8))
            # Add noise
            noise = torch.randn_like(g) * noise_multiplier * max_grad_norm
            noisy.append(g_clipped + noise)
        return noisy
    
    def attack(
        self,
        target_gradient: List[torch.Tensor],
        input_shape: Tuple[int, ...],
        num_targets: int = 3,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run gradient inversion attack.
        
        Args:
            target_gradient: The gradient to invert
            input_shape: Shape of a single input sample
            num_targets: Number of target columns
        
        Returns:
            (reconstructed_data, final_loss)
        """
        # Initialize random dummy data
        dummy_x = torch.randn(1, *input_shape, device=self.device, requires_grad=True)
        dummy_y = torch.sigmoid(torch.randn(1, num_targets, device=self.device))
        dummy_y = dummy_y.detach().requires_grad_(True)
        
        optimizer = torch.optim.LBFGS(
            [dummy_x, dummy_y], lr=self.lr
        )
        
        best_loss = float("inf")
        best_x = dummy_x.clone()
        
        for i in range(self.num_iterations):
            def closure():
                optimizer.zero_grad()
                self.model.zero_grad()
                
                preds = self.model(dummy_x)
                targets = (dummy_y[:, 0:1], dummy_y[:, 1:2], dummy_y[:, 2:3])
                
                loss = sum(
                    nn.BCELoss()(p, t.clamp(0, 1))
                    for p, t in zip(preds, targets)
                )
                loss.backward(retain_graph=True)
                
                # Compute gradient matching loss
                dummy_grad = [
                    p.grad.clone()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
                
                grad_loss = sum(
                    ((dg - tg) ** 2).sum()
                    for dg, tg in zip(dummy_grad, target_gradient)
                )
                
                grad_loss.backward()
                return grad_loss
            
            loss = optimizer.step(closure)
            
            if loss is not None and loss.item() < best_loss:
                best_loss = loss.item()
                best_x = dummy_x.clone().detach()
        
        return best_x, best_loss
    
    def compute_psnr(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between original and reconstructed.
        
        High PSNR (>20 dB) = good reconstruction (attack succeeds)
        Low PSNR (<10 dB) = poor reconstruction (defense works)
        """
        mse = np.mean((original - reconstructed) ** 2)
        if mse < 1e-10:
            return float("inf")
        
        data_range = np.max(original) - np.min(original)
        if data_range < 1e-10:
            data_range = 1.0
        
        psnr = 20 * np.log10(data_range / np.sqrt(mse))
        return float(psnr)


def run_gradient_inversion_experiment(
    model: nn.Module,
    X_sample: np.ndarray,
    Y_sample: np.ndarray,
    noise_multipliers: List[float] = None,
    max_grad_norm: float = 1.0,
    device: str = "cpu",
    results_dir: str = "results",
) -> Dict:
    """
    Run gradient inversion attack with and without DP.
    
    Args:
        model: Trained model
        X_sample: Sample input data (small batch)
        Y_sample: Sample targets
        noise_multipliers: DP noise levels to test (0.0 = no DP)
        max_grad_norm: Gradient clipping norm
        device: Compute device
        results_dir: Directory for results
    
    Returns:
        Dict with PSNR values for each noise level
    """
    if noise_multipliers is None:
        noise_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    device = torch.device(device)
    model = model.to(device).eval()
    
    # Use a small batch for the attack
    x_target = torch.FloatTensor(X_sample[:1]).to(device)
    y_target = torch.FloatTensor(Y_sample[:1]).to(device)
    
    attacker = GradientInversionAttack(model, device)
    
    # Compute true gradient
    true_gradient = attacker.compute_gradient(model, x_target, y_target)
    
    results = {}
    
    for sigma in noise_multipliers:
        logger.info(f"  Gradient inversion attack with sigma={sigma}...")
        
        if sigma > 0:
            noisy_gradient = attacker.add_dp_noise(true_gradient, sigma, max_grad_norm)
        else:
            noisy_gradient = true_gradient
        
        # Run attack
        reconstructed, attack_loss = attacker.attack(
            noisy_gradient,
            input_shape=(X_sample.shape[1],),
        )
        
        # Compute PSNR
        psnr = attacker.compute_psnr(
            x_target.cpu().numpy(),
            reconstructed.cpu().numpy()
        )
        
        results[f"sigma_{sigma}"] = {
            "noise_multiplier": sigma,
            "psnr_db": psnr,
            "attack_loss": float(attack_loss),
            "defense_effective": psnr < 10.0,
        }
        
        logger.info(
            f"    sigma={sigma} -> PSNR={psnr:.2f} dB | "
            f"{'[X] Attack succeeds' if psnr > 15 else '[OK] Defense works'}"
        )
    
    # Save results
    filepath = os.path.join(results_dir, "gradient_inversion_results.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ============================================================================
# MEMBERSHIP INFERENCE ATTACK
# ============================================================================

class MembershipInferenceAttack:
    """
    Determine if a given sample was used in training.
    
    Uses the confidence-based approach: trained models tend to be
    more confident on training data than unseen data. An attack
    model learns to distinguish members from non-members based
    on the target model's output confidence.
    
    Success rate near 50% = good privacy (random guessing).
    Success rate near 100% = poor privacy (model memorizes data).
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        device: torch.device = None,
        num_shadow_models: int = 3,
        attack_epochs: int = 20,
    ):
        self.target_model = target_model
        self.device = device or torch.device("cpu")
        self.num_shadow_models = num_shadow_models
        self.attack_epochs = attack_epochs
    
    def get_confidence_features(
        self,
        model: nn.Module,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Extract confidence features from model predictions.
        
        Returns prediction probabilities for all 3 tasks,
        plus entropy and max confidence.
        """
        model.eval()
        model.to(self.device)
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            preds = model(x_tensor)
            probs = torch.cat(preds, dim=1).cpu().numpy()  # (N, 3)
        
        # Additional features: entropy and max confidence
        entropy = -np.sum(
            probs * np.log(probs + 1e-8) + (1 - probs) * np.log(1 - probs + 1e-8),
            axis=1,
            keepdims=True
        )
        max_conf = np.max(np.abs(probs - 0.5), axis=1, keepdims=True)
        
        return np.concatenate([probs, entropy, max_conf], axis=1)  # (N, 5)
    
    def train_attack_model(
        self,
        member_features: np.ndarray,
        non_member_features: np.ndarray,
    ) -> nn.Module:
        """Train a binary classifier to distinguish members from non-members."""
        X = np.concatenate([member_features, non_member_features])
        y = np.concatenate([
            np.ones(len(member_features)),
            np.zeros(len(non_member_features))
        ])
        
        # Simple logistic regression attack model
        input_dim = X.shape[1]
        attack_model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        attack_model.train()
        for epoch in range(self.attack_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = attack_model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        
        return attack_model
    
    def evaluate_attack(
        self,
        attack_model: nn.Module,
        member_features: np.ndarray,
        non_member_features: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate MI attack success rate."""
        X = np.concatenate([member_features, non_member_features])
        y_true = np.concatenate([
            np.ones(len(member_features)),
            np.zeros(len(non_member_features))
        ])
        
        attack_model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            y_pred = attack_model(x_tensor).cpu().numpy().flatten()
        
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        metrics = {
            "attack_accuracy": float(accuracy_score(y_true, y_pred_binary)),
            "attack_auc": float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5,
            "near_random": float(accuracy_score(y_true, y_pred_binary)) < 0.55,
        }
        
        return metrics


def run_membership_inference_experiment(
    model: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    device: str = "cpu",
    results_dir: str = "results",
    num_shadow_models: int = 3,
) -> Dict:
    """
    Run membership inference attack experiment.
    
    Uses training data as "members" and test data as "non-members."
    
    Target: attack accuracy < 55% (near random guessing = good privacy)
    """
    logger.info("Running membership inference attack...")
    
    device = torch.device(device)
    
    attack = MembershipInferenceAttack(
        model, device,
        num_shadow_models=num_shadow_models
    )
    
    # Subsample for efficiency
    n_samples = min(2000, len(X_train), len(X_test))
    member_idx = np.random.choice(len(X_train), n_samples, replace=False)
    non_member_idx = np.random.choice(len(X_test), n_samples, replace=False)
    
    X_members = X_train[member_idx]
    X_non_members = X_test[non_member_idx]
    
    # Get confidence features
    member_features = attack.get_confidence_features(model, X_members)
    non_member_features = attack.get_confidence_features(model, X_non_members)
    
    # Split into attack train/test
    n_train = int(0.7 * n_samples)
    
    train_member = member_features[:n_train]
    train_non_member = non_member_features[:n_train]
    test_member = member_features[n_train:]
    test_non_member = non_member_features[n_train:]
    
    # Train attack model
    attack_model = attack.train_attack_model(train_member, train_non_member)
    
    # Evaluate
    metrics = attack.evaluate_attack(attack_model, test_member, test_non_member)
    
    logger.info(
        f"  MI Attack Accuracy: {metrics['attack_accuracy']:.2%} | "
        f"AUC: {metrics['attack_auc']:.4f} | "
        f"{'[OK] Near random (good privacy)' if metrics['near_random'] else '[X] Leaking info'}"
    )
    
    # Save results
    import json
    filepath = os.path.join(results_dir, "membership_inference_results.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


# ============================================================================
# FULL ATTACK SIMULATION
# ============================================================================

def run_all_attacks(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    noise_multipliers: List[float] = None,
    device: str = "cpu",
    results_dir: str = "results",
) -> Dict:
    """
    Run all attack simulations.
    
    ⚠️ GPU recommended for faster attack simulation.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ATTACK SIMULATION SUITE")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Gradient Inversion Attack
    logger.info("\n--- Gradient Inversion Attack ---")
    gi_results = run_gradient_inversion_experiment(
        model, X_train, Y_train,
        noise_multipliers=noise_multipliers,
        device=device,
        results_dir=results_dir,
    )
    results["gradient_inversion"] = gi_results
    
    # 2. Membership Inference Attack
    logger.info("\n--- Membership Inference Attack ---")
    mi_results = run_membership_inference_experiment(
        model, X_train, X_test,
        device=device,
        results_dir=results_dir,
    )
    results["membership_inference"] = mi_results
    
    # Save combined results
    filepath = os.path.join(results_dir, "attack_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nAll attack results saved to {filepath}")
    return results
