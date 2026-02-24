"""
Central configuration for the Privacy-Preserving Federated Learning system.
All hyperparameters and experiment settings in one place.

Key changes from original:
  - LR reduced: 0.001 -> 0.0003 (fixes training divergence)
  - FedProx mu increased: 0.01 -> 0.1 (prevents client drift)
  - Top-K ratio increased: 0.1 -> 0.3 (reduces info loss)
  - Added: loss_type, norm_type, FedBN, SCAFFOLD, Fisher DP, curriculum fields
  - Added: data_sources for multi-dataset support
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import argparse
import logging
import torch
import os

logger = logging.getLogger("ppfl-ncd.config")


def _xpu_available() -> bool:
    """Check if Intel XPU is available, either natively (PyTorch 2.4+) or via IPEX."""
    # PyTorch >= 2.4 has native XPU support — no IPEX needed
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return True
    # Older PyTorch: try loading IPEX to enable XPU
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except (ImportError, RuntimeError) as e:
        logger.debug(f"IPEX not loaded: {e}")
    return False


def get_best_device(preference: str = "auto") -> torch.device:
    """
    Resolve the best available compute device.

    Priority order: explicit preference > XPU (Intel Arc/Data Center) > CUDA > CPU.

    Supports both native PyTorch XPU (>= 2.4) and IPEX-based XPU.

    Args:
        preference: "auto", "cpu", "cuda", or "xpu"

    Returns:
        torch.device for the best available accelerator
    """
    if preference == "xpu":
        if _xpu_available():
            logger.info("Using Intel XPU device")
            return torch.device("xpu")
        else:
            logger.warning("XPU requested but not available, falling back to CPU")
            return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    # Auto-detect: XPU > CUDA > CPU
    if _xpu_available():
        logger.info("Auto-detected Intel XPU device")
        return torch.device("xpu")

    if torch.cuda.is_available():
        logger.info("Auto-detected CUDA device")
        return torch.device("cuda")

    logger.info("Using CPU device")
    return torch.device("cpu")


@dataclass

class DataConfig:
    """Data pipeline configuration."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    partitions_dir: str = "data/partitions"

    # Data sources to use (supports multi-dataset FL)
    data_sources: List[str] = field(default_factory=lambda: [
        "brfss", "nhanes"
    ])

    # Feature columns (harmonized across all data sources)
    demographic_features: List[str] = field(default_factory=lambda: [
        "age", "sex", "bmi", "race_ethnicity"
    ])
    lifestyle_features: List[str] = field(default_factory=lambda: [
        "smoking_status", "alcohol_consumption", "physical_activity"
    ])
    clinical_features: List[str] = field(default_factory=lambda: [
        "blood_pressure_systolic", "blood_pressure_diastolic",
        "cholesterol_total", "cholesterol_hdl", "blood_glucose"
    ])
    comorbidity_features: List[str] = field(default_factory=lambda: [
        "has_kidney_disease", "has_stroke_history", "has_arthritis"
    ])

    # Target columns
    target_columns: List[str] = field(default_factory=lambda: [
        "diabetes", "hypertension", "cardiovascular_disease"
    ])

    # Synthetic data settings
    synthetic_num_samples: int = 50000
    synthetic_seed: int = 42

    @property
    def all_features(self) -> List[str]:
        return (self.demographic_features + self.lifestyle_features +
                self.clinical_features + self.comorbidity_features)

    @property
    def input_dim(self) -> int:
        return len(self.all_features)


@dataclass
class ModelConfig:
    """Multi-task neural network configuration."""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Normalization type: 'group' (Opacus-safe), 'layer', 'batch'
    norm_type: str = 'group'
    num_groups: int = 8  # for GroupNorm

    # Loss configuration
    loss_type: str = 'focal'  # 'bce', 'weighted_bce', 'focal'
    focal_gamma: float = 2.0  # focusing parameter for focal loss


@dataclass
class FLConfig:
    """Federated Learning configuration."""
    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.0003  # Reduced from 0.001 to fix divergence

    # FedProx
    fedprox_mu: float = 0.1  # Increased from 0.01 to prevent client drift

    # Early stopping
    early_stop_patience: int = 10

    # Dirichlet partitioning
    dirichlet_alpha: float = 0.5
    num_brfss_clients: int = 5
    num_nhanes_clients: int = 5

    # Fraction of clients per round
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2

    # Learning rate scheduling (server-coordinated)
    lr_decay_strategy: str = 'warmup_cosine'  # 'none', 'cosine', 'warmup_cosine'
    lr_warmup_rounds: int = 5  # Warmup rounds before decay

    # FedBN: keep normalization layers local (Rec 3)
    use_fedbn: bool = True

    # SCAFFOLD variance reduction (Rec 4)
    use_scaffold: bool = False  # Disabled by default — enable for Rec 4 experiments

    # Curriculum learning (Rec 5)
    use_curriculum: bool = True
    curriculum_phases: List[Tuple[int, str]] = field(default_factory=lambda: [
        (15, 'warmup'),    # Rounds 1-15: heavy oversampling + warm-up LR
        (35, 'balanced'),  # Rounds 16-35: focal loss, natural ratios
        (50, 'hard_mine'), # Rounds 36-50: hard-example mining
    ])

    # Knowledge distillation (Rec 6)
    use_knowledge_distillation: bool = False
    kd_temperature: float = 3.0
    kd_alpha: float = 0.5  # Weight of distillation loss vs task loss
    kd_proxy_samples: int = 1000

    # Class-aware aggregation (Rec 1)
    use_class_aware_aggregation: bool = True

    # Weighted random sampling — oversample minority class locally
    use_weighted_sampling: bool = True


@dataclass
class PrivacyConfig:
    """Differential Privacy configuration."""
    enable_dp: bool = True
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    max_epsilon: float = 3.0

    # Fisher-guided adaptive DP noise (Rec 2)
    use_fisher_dp: bool = False  # Disabled by default — enable for Rec 2
    fisher_samples: int = 256  # Samples for Fisher Information estimation

    # Noise calibration experiment values
    noise_multiplier_sweep: List[float] = field(
        default_factory=lambda: [0.5, 0.8, 1.1, 1.5, 2.0]
    )

    # Secure Aggregation
    enable_secagg: bool = True


@dataclass
class CompressionConfig:
    """Top-K gradient compression configuration."""
    enable_compression: bool = True
    k_ratio: float = 0.3  # Increased from 0.1 to reduce info loss

    # Sweep values for experiments
    k_ratio_sweep: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    )
    enable_error_feedback: bool = True


@dataclass
class ExplainabilityConfig:
    """SHAP configuration."""
    enable_shap: bool = True
    background_samples: int = 100
    top_k_features: int = 10
    compute_every_n_rounds: int = 5


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)

    # General
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "xpu"
    results_dir: str = "results"
    use_wandb: bool = False
    wandb_project: str = "ppfl-ncd"
    experiment_name: str = "default"

    def get_device(self) -> torch.device:
        return get_best_device(self.device)


def compute_lr_for_round(
    base_lr: float,
    current_round: int,
    total_rounds: int,
    strategy: str = 'warmup_cosine',
    warmup_rounds: int = 5,
) -> float:
    """
    Compute learning rate for a given round (server-coordinated LR schedule).

    Args:
        base_lr: Base learning rate
        current_round: Current FL round (1-indexed)
        total_rounds: Total number of FL rounds
        strategy: 'none', 'cosine', or 'warmup_cosine'
        warmup_rounds: Number of warmup rounds

    Returns:
        Adjusted learning rate for this round
    """
    import math

    if strategy == 'none':
        return base_lr

    if strategy == 'cosine':
        # Simple cosine annealing
        return base_lr * 0.5 * (1 + math.cos(math.pi * current_round / total_rounds))

    if strategy == 'warmup_cosine':
        if current_round <= warmup_rounds:
            # Linear warmup
            return base_lr * (current_round / warmup_rounds)
        else:
            # Cosine decay after warmup
            progress = (current_round - warmup_rounds) / max(1, total_rounds - warmup_rounds)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    return base_lr


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments into ExperimentConfig."""
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Federated Learning for NCD Prediction"
    )

    # Data
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--input", type=str, default="data/processed",
                        help="Input directory for partition.py")

    # FL
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--fedprox-mu", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter")

    # Novel approaches
    parser.add_argument("--loss-type", type=str, default="focal",
                        choices=["bce", "weighted_bce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-fedbn", action="store_true",
                        help="Disable FedBN (aggregate all params)")
    parser.add_argument("--use-scaffold", action="store_true",
                        help="Enable SCAFFOLD variance reduction")
    parser.add_argument("--use-fisher-dp", action="store_true",
                        help="Enable Fisher-guided adaptive DP noise")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--use-kd", action="store_true",
                        help="Enable knowledge distillation")
    parser.add_argument("--no-class-aware", action="store_true",
                        help="Disable class-aware aggregation")

    # Privacy
    parser.add_argument("--no-dp", action="store_true",
                        help="Disable differential privacy")
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-epsilon", type=float, default=3.0)

    # Compression
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--k-ratio", type=float, default=0.3)

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="default")

    args = parser.parse_args()

    config = ExperimentConfig()
    config.seed = args.seed
    config.device = args.device
    config.results_dir = args.results_dir
    config.use_wandb = args.wandb
    config.experiment_name = args.experiment_name

    config.data.processed_dir = args.output

    config.fl.num_rounds = args.num_rounds
    config.fl.num_clients = args.num_clients
    config.fl.local_epochs = args.local_epochs
    config.fl.batch_size = args.batch_size
    config.fl.learning_rate = args.lr
    config.fl.fedprox_mu = args.fedprox_mu
    config.fl.dirichlet_alpha = args.alpha

    # Novel approaches
    config.model.loss_type = args.loss_type
    config.model.focal_gamma = args.focal_gamma
    config.fl.use_fedbn = not args.no_fedbn
    config.fl.use_scaffold = args.use_scaffold
    config.fl.use_curriculum = not args.no_curriculum
    config.fl.use_knowledge_distillation = args.use_kd
    config.fl.use_class_aware_aggregation = not args.no_class_aware

    config.privacy.enable_dp = not args.no_dp
    config.privacy.noise_multiplier = args.noise_multiplier
    config.privacy.max_grad_norm = args.max_grad_norm
    config.privacy.max_epsilon = args.max_epsilon
    config.privacy.use_fisher_dp = args.use_fisher_dp

    config.compression.enable_compression = not args.no_compression
    config.compression.k_ratio = args.k_ratio

    return config
