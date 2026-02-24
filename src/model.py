"""
Multi-Task Neural Network for simultaneous NCD risk prediction.

Architecture:
  Shared Encoder (3 FC layers with GroupNorm + ReLU + Dropout)
  -> 3 separate classification heads (Diabetes, Hypertension, CVD)

Each head outputs a sigmoid probability. The combined loss is a
weighted sum of per-task loss values (BCE, Focal, or weighted BCE).

Key improvements over original:
  - GroupNorm instead of BatchNorm (Opacus-compatible, FedBN-ready)
  - FocalLoss for class-imbalanced NCD prediction
  - Dynamic pos_weight support in MultiTaskLoss
  - get_aggregation_params() for FedBN selective aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Focal Loss — addresses extreme class imbalance (Issue #1, Rec 1)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Downweights easy negatives and focuses learning on hard positives.
    Critical for NCD prediction where positive rates are 0.27%-14%.

    Args:
        gamma: Focusing parameter (0 = standard BCE, 2 = typical for imbalance)
        alpha: Balance weight for positive class (0-1). Higher = more weight on positives.
               If None, no class balancing is applied (only focusing).
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape (N, 1)
            targets: Binary labels, shape (N, 1)
        """
        probs = torch.sigmoid(logits)
        # p_t = probability of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Standard BCE loss (without reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Apply focal modulation
        loss = focal_weight * bce

        # Apply alpha balancing
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# Model Architecture
# ============================================================================

class MultiTaskNCD(nn.Module):
    """
    Multi-task neural network for NCD risk prediction.

    Shared encoder extracts common health risk representations,
    then three separate heads predict diabetes, hypertension,
    and cardiovascular disease risk independently.

    Changes from original:
    - Supports 'group' (default), 'layer', or 'batch' normalization
    - GroupNorm is Opacus-compatible without requiring ModuleValidator.fix()
    - Tracks which params are normalization params for FedBN
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        num_tasks: int = 3,
        norm_type: str = 'group',
        num_groups: int = 8,
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
            dropout: Dropout rate
            num_tasks: Number of prediction tasks (default: 3)
            norm_type: 'group' (Opacus-safe), 'layer', or 'batch'
            num_groups: Number of groups for GroupNorm (default: 8)
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_tasks = num_tasks
        self.norm_type = norm_type

        # Build shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(self._make_norm(h_dim, norm_type, num_groups))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
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
        self._head_names = ['diabetes_head', 'hypertension_head', 'cvd_head']

    @staticmethod
    def _make_norm(dim: int, norm_type: str, num_groups: int = 8) -> nn.Module:
        """Create normalization layer based on type."""
        if norm_type == 'group':
            # Ensure num_groups divides dim
            while dim % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            return nn.GroupNorm(num_groups, dim)
        elif norm_type == 'layer':
            return nn.LayerNorm(dim)
        elif norm_type == 'batch':
            return nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        During training: returns raw logits (for BCEWithLogitsLoss / FocalLoss).
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

    def get_aggregation_mask(self) -> Dict[str, bool]:
        """
        For FedBN: returns dict mapping param name -> whether to aggregate.

        Normalization layer params (weight, bias) are kept LOCAL.
        All other params (Linear weights, biases) are AGGREGATED.
        """
        mask = {}
        norm_keywords = ['groupnorm', 'layernorm', 'batchnorm', 'bn', 'gn', 'ln']
        for name, _ in self.named_parameters():
            # Check if this parameter belongs to a normalization layer
            name_lower = name.lower()
            is_norm = any(kw in name_lower for kw in norm_keywords)
            # Also check by module type
            parts = name.split('.')
            try:
                module = self
                for part in parts[:-1]:
                    if part.isdigit():
                        module = list(module.children())[int(part)]
                    else:
                        module = getattr(module, part)
                is_norm = is_norm or isinstance(module, (
                    nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d
                ))
            except (AttributeError, IndexError):
                pass
            mask[name] = not is_norm
        return mask

    def get_head_param_names(self) -> Dict[str, List[str]]:
        """Return parameter names grouped by task head (for class-aware aggregation)."""
        head_params = {}
        for head_name in self._head_names:
            prefix = f"{head_name}."
            head_params[head_name] = [
                name for name, _ in self.named_parameters()
                if name.startswith(prefix)
            ]
        # Encoder params
        head_params['encoder'] = [
            name for name, _ in self.named_parameters()
            if name.startswith('encoder.')
        ]
        return head_params

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


# ============================================================================
# Loss Functions
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss supporting BCE, Focal, and weighted BCE.

    L_total = w_1 * Loss(diabetes) + w_2 * Loss(hypertension) + w_3 * Loss(cvd)
    """

    def __init__(
        self,
        task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pos_weights: Optional[Tuple[float, float, float]] = None,
        loss_type: str = 'focal',
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            task_weights: Weight for each task's loss contribution.
            pos_weights: Positive class weights (N_neg/N_pos per task).
                         For BCE: used as pos_weight. For Focal: used as alpha.
            loss_type: 'bce', 'weighted_bce', or 'focal'
            focal_gamma: Gamma for focal loss (only when loss_type='focal')
        """
        super().__init__()
        self.task_weights = task_weights
        self.loss_type = loss_type

        self.criteria = nn.ModuleList()
        for i in range(3):
            if loss_type == 'focal':
                # For focal loss, convert pos_weight to alpha
                alpha = None
                if pos_weights is not None:
                    # alpha = pos_weight / (1 + pos_weight) gives class-balanced alpha
                    pw = pos_weights[i]
                    alpha = pw / (1.0 + pw)
                self.criteria.append(FocalLoss(gamma=focal_gamma, alpha=alpha))
            elif loss_type == 'weighted_bce' and pos_weights is not None:
                self.criteria.append(
                    nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weights[i]]))
                )
            else:
                self.criteria.append(nn.BCEWithLogitsLoss())

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

    def update_pos_weights(self, pos_weights: Tuple[float, float, float]):
        """Dynamically update pos_weights (for class-aware FL)."""
        for i, pw in enumerate(pos_weights):
            criterion = self.criteria[i]
            if isinstance(criterion, FocalLoss):
                criterion.alpha = pw / (1.0 + pw)
            elif isinstance(criterion, nn.BCEWithLogitsLoss):
                criterion.pos_weight = torch.tensor([pw])


class FedProxLoss(nn.Module):
    """
    FedProx loss = Multi-task loss + mu/2 * ||w - w_global||^2

    The proximal term prevents local model from drifting too far
    from the global model, improving convergence on non-IID data.
    """

    def __init__(
        self,
        mu: float = 0.1,
        task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pos_weights: Optional[Tuple[float, float, float]] = None,
        loss_type: str = 'focal',
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            mu: Proximal term weight (higher = stricter regularization)
            task_weights: Weight for each task's loss contribution
            pos_weights: Optional positive class weights for imbalanced data
            loss_type: 'bce', 'weighted_bce', or 'focal'
            focal_gamma: Gamma for focal loss
        """
        super().__init__()
        self.mu = mu
        self.multi_task_loss = MultiTaskLoss(
            task_weights, pos_weights, loss_type, focal_gamma
        )

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

        # Proximal term: mu/2 * ||w_local - w_global||^2
        proximal = torch.tensor(0.0, device=preds[0].device)
        for local_p, global_p in zip(local_model.parameters(), global_params):
            proximal = proximal + (local_p - global_p.detach()).pow(2).sum()

        proximal = (self.mu / 2.0) * proximal

        total = task_loss + proximal
        per_task["proximal"] = proximal.item()
        per_task["total"] = total.item()

        return total, per_task

    def update_pos_weights(self, pos_weights: Tuple[float, float, float]):
        """Dynamically update pos_weights."""
        self.multi_task_loss.update_pos_weights(pos_weights)


def create_model(input_dim: int, config=None) -> MultiTaskNCD:
    """Factory function to create a model from config."""
    if config is not None:
        return MultiTaskNCD(
            input_dim=input_dim,
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
            norm_type=getattr(config.model, 'norm_type', 'group'),
        )
    return MultiTaskNCD(input_dim=input_dim)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    input_dim = 15
    model = MultiTaskNCD(input_dim, norm_type='group')
    print(f"Model architecture:\n{model}")
    print(f"\nParameter counts: {model.parameter_count()}")
    print(f"\nAggregation mask (FedBN):")
    for name, agg in model.get_aggregation_mask().items():
        print(f"  {name}: {'aggregate' if agg else 'LOCAL (keep)'}")

    # Forward pass test
    model.eval()
    batch = torch.randn(32, input_dim)
    d_pred, h_pred, c_pred = model(batch)
    print(f"\nForward pass shapes (eval -> probabilities):")
    print(f"  Diabetes:      {d_pred.shape}")
    print(f"  Hypertension:  {h_pred.shape}")
    print(f"  CVD:           {c_pred.shape}")

    # Loss test with FocalLoss (train mode -> logits)
    model.train()
    d_logit, h_logit, c_logit = model(batch)

    targets = (
        torch.randint(0, 2, (32, 1)).float(),
        torch.randint(0, 2, (32, 1)).float(),
        torch.randint(0, 2, (32, 1)).float(),
    )

    # Test all loss types
    for loss_type in ['bce', 'weighted_bce', 'focal']:
        pw = (6.4, 370.0, 10.4) if loss_type != 'bce' else None
        loss_fn = MultiTaskLoss(pos_weights=pw, loss_type=loss_type)
        total_loss, per_task = loss_fn((d_logit, h_logit, c_logit), targets)
        print(f"\n{loss_type.upper()} loss: {per_task}")

    # FedProx loss test
    fedprox = FedProxLoss(
        mu=0.1, pos_weights=(6.4, 370.0, 10.4), loss_type='focal'
    )
    global_params = [p.clone().detach() for p in model.parameters()]
    total, per_task = fedprox(
        (d_logit, h_logit, c_logit), targets, model, global_params
    )
    print(f"\nFedProx + Focal loss: {per_task}")

    # Opacus compatibility check
    try:
        from opacus.validators import ModuleValidator
        valid = ModuleValidator.is_valid(model)
        print(f"\nOpacus compatible: {valid}")
    except ImportError:
        print("\nOpacus not installed, skipping compatibility check")
