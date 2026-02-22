"""
Multi-Task Neural Network for simultaneous NCD risk prediction.

Architecture:
  Shared Encoder (3 FC layers with BatchNorm + ReLU + Dropout)
  → 3 separate classification heads (Diabetes, Hypertension, CVD)
  
Each head outputs a sigmoid probability. The combined loss is a
weighted sum of per-task BCELoss values.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class MultiTaskNCD(nn.Module):
    """
    Multi-task neural network for NCD risk prediction.
    
    Shared encoder extracts common health risk representations,
    then three separate heads predict diabetes, hypertension,
    and cardiovascular disease risk independently.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        num_tasks: int = 3
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
            dropout: Dropout rate
            num_tasks: Number of prediction tasks (default: 3)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_tasks = num_tasks
        
        # Build shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Task-specific heads (output raw logits; sigmoid applied at inference)
        last_hidden = hidden_dims[-1]
        self.diabetes_head = nn.Sequential(
            nn.Linear(last_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.hypertension_head = nn.Sequential(
            nn.Linear(last_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cvd_head = nn.Sequential(
            nn.Linear(last_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self._heads = [self.diabetes_head, self.hypertension_head, self.cvd_head]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        During training: returns raw logits (for BCEWithLogitsLoss).
        During eval: returns sigmoid probabilities.
        
        Args:
            x: Input features, shape (batch_size, input_dim)
        
        Returns:
            Tuple of 3 tensors, each shape (batch_size, 1):
              (diabetes_logit_or_prob, hypertension_logit_or_prob, cvd_logit_or_prob)
        """
        z = self.encoder(x)
        logits = (
            self.diabetes_head(z),
            self.hypertension_head(z),
            self.cvd_head(z)
        )
        if self.training:
            return logits
        return tuple(torch.sigmoid(l) for l in logits)
    
    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get shared encoder representation (for SHAP analysis)."""
        return self.encoder(x)
    
    def parameter_count(self) -> dict:
        """Count parameters per component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            "encoder": count_params(self.encoder),
            "diabetes_head": count_params(self.diabetes_head),
            "hypertension_head": count_params(self.hypertension_head),
            "cvd_head": count_params(self.cvd_head),
            "total": count_params(self),
        }


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task binary cross-entropy loss.
    
    L_total = w_1 * BCE(diabetes) + w_2 * BCE(hypertension) + w_3 * BCE(cvd)
    """
    
    def __init__(
        self,
        task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pos_weights: Optional[Tuple[float, float, float]] = None
    ):
        """
        Args:
            task_weights: Weight for each task's loss contribution
            pos_weights: Optional positive class weights for imbalanced data
        """
        super().__init__()
        self.task_weights = task_weights
        
        if pos_weights is not None:
            self.criteria = [
                nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))
                for w in pos_weights
            ]
        else:
            self.criteria = [nn.BCEWithLogitsLoss() for _ in range(3)]
    
    def forward(
        self,
        preds: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute weighted multi-task loss.
        
        Args:
            preds: Tuple of 3 prediction tensors (batch, 1)
            targets: Tuple of 3 target tensors (batch, 1)
        
        Returns:
            (total_loss, per_task_losses_dict)
        """
        task_names = ["diabetes", "hypertension", "cvd"]
        per_task = {}
        total = torch.tensor(0.0, device=preds[0].device)
        
        for name, w, criterion, pred, target in zip(
            task_names, self.task_weights, self.criteria, preds, targets
        ):
            loss = criterion(pred, target)
            per_task[name] = loss.item()
            total = total + w * loss
        
        per_task["total"] = total.item()
        return total, per_task


class FedProxLoss(nn.Module):
    """
    FedProx loss = Multi-task loss + μ/2 * ||w - w_global||²
    
    The proximal term prevents local model from drifting too far
    from the global model, improving convergence on non-IID data.
    """
    
    def __init__(
        self,
        mu: float = 0.01,
        task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        """
        Args:
            mu: Proximal term weight (higher = stricter regularization)
            task_weights: Weight for each task's loss contribution
        """
        super().__init__()
        self.mu = mu
        self.multi_task_loss = MultiTaskLoss(task_weights)
    
    def forward(
        self,
        preds: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...],
        local_model: nn.Module,
        global_params: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute FedProx loss.
        
        Args:
            preds: Model predictions
            targets: Ground truth labels
            local_model: Current local model
            global_params: Parameters from the global model
        
        Returns:
            (total_loss_with_proximal, per_task_losses)
        """
        task_loss, per_task = self.multi_task_loss(preds, targets)
        
        # Proximal term: μ/2 * ||w_local - w_global||²
        proximal = torch.tensor(0.0, device=preds[0].device)
        for local_p, global_p in zip(local_model.parameters(), global_params):
            proximal = proximal + (local_p - global_p.detach()).pow(2).sum()
        
        proximal = (self.mu / 2.0) * proximal
        
        total = task_loss + proximal
        per_task["proximal"] = proximal.item()
        per_task["total"] = total.item()
        
        return total, per_task


def create_model(input_dim: int, config=None) -> MultiTaskNCD:
    """Factory function to create a model from config."""
    if config is not None:
        return MultiTaskNCD(
            input_dim=input_dim,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
        )
    return MultiTaskNCD(input_dim=input_dim)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    # Smoke test
    input_dim = 15  # number of harmonized features
    model = MultiTaskNCD(input_dim)
    print(f"Model architecture:\n{model}")
    print(f"\nParameter counts: {model.parameter_count()}")
    
    # Forward pass test (eval mode → probabilities)
    model.eval()
    batch = torch.randn(32, input_dim)
    d_pred, h_pred, c_pred = model(batch)
    print(f"\nForward pass shapes (eval mode → probabilities):")
    print(f"  Diabetes:      {d_pred.shape}")
    print(f"  Hypertension:  {h_pred.shape}")
    print(f"  CVD:           {c_pred.shape}")
    
    # Loss test (train mode → logits for BCEWithLogitsLoss)
    model.train()
    d_logit, h_logit, c_logit = model(batch)
    
    # Loss test
    targets = (
        torch.randint(0, 2, (32, 1)).float(),
        torch.randint(0, 2, (32, 1)).float(),
        torch.randint(0, 2, (32, 1)).float(),
    )
    
    loss_fn = MultiTaskLoss()
    total_loss, per_task = loss_fn((d_logit, h_logit, c_logit), targets)
    print(f"\nLoss values: {per_task}")
    
    # FedProx loss test
    fedprox_loss = FedProxLoss(mu=0.01)
    global_params = [p.clone().detach() for p in model.parameters()]
    total, per_task = fedprox_loss(
        (d_logit, h_logit, c_logit), targets, model, global_params
    )
    print(f"FedProx loss: {per_task}")
