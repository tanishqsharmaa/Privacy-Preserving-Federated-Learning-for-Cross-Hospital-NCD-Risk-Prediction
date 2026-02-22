"""
Central configuration for the Privacy-Preserving Federated Learning system.
All hyperparameters and experiment settings in one place.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import argparse
import torch
import os


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    partitions_dir: str = "data/partitions"
    
    # Feature columns (harmonized across BRFSS and NHANES)
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
    task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # diabetes, hypertension, cvd


@dataclass
class FLConfig:
    """Federated Learning configuration."""
    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.001
    
    # FedProx
    fedprox_mu: float = 0.01  # proximal term weight
    
    # Early stopping
    early_stop_patience: int = 10  # stop if no AUC improvement for N rounds
    
    # Dirichlet partitioning
    dirichlet_alpha: float = 0.5
    num_brfss_clients: int = 5
    num_nhanes_clients: int = 5
    
    # Fraction of clients per round
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2


@dataclass
class PrivacyConfig:
    """Differential Privacy configuration."""
    enable_dp: bool = True
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    max_epsilon: float = 3.0  # privacy budget ceiling
    
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
    k_ratio: float = 0.1  # keep top 10% of gradients
    
    # Sweep values for experiments
    k_ratio_sweep: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 1.0]
    )
    enable_error_feedback: bool = True  # accumulate residuals


@dataclass
class ExplainabilityConfig:
    """SHAP configuration."""
    enable_shap: bool = True
    background_samples: int = 100  # samples for DeepExplainer background
    top_k_features: int = 10  # top features to report
    compute_every_n_rounds: int = 5  # compute SHAP every N rounds


@dataclass
class AttackConfig:
    """Attack simulation configuration."""
    # Gradient Inversion
    enable_gradient_inversion: bool = True
    gi_num_iterations: int = 300
    gi_learning_rate: float = 0.1
    
    # Membership Inference
    enable_membership_inference: bool = True
    mi_num_shadow_models: int = 3
    mi_attack_epochs: int = 20


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    
    # General
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    results_dir: str = "results"
    use_wandb: bool = False
    wandb_project: str = "ppfl-ncd"
    experiment_name: str = "default"
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration parameter")
    
    # Privacy
    parser.add_argument("--no-dp", action="store_true",
                        help="Disable differential privacy")
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-epsilon", type=float, default=3.0)
    
    # Compression
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--k-ratio", type=float, default=0.1)
    
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
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
    
    config.privacy.enable_dp = not args.no_dp
    config.privacy.noise_multiplier = args.noise_multiplier
    config.privacy.max_grad_norm = args.max_grad_norm
    config.privacy.max_epsilon = args.max_epsilon
    
    config.compression.enable_compression = not args.no_compression
    config.compression.k_ratio = args.k_ratio
    
    return config
