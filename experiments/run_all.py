"""
Comprehensive experiment runner for FL-NCD research.

Runs ALL baseline comparisons + ablation studies for novel contributions:

  1. Centralized baseline (upper bound)
  2. FedAvg (standard FL baseline)
  3. FedAvg + Focal Loss (class imbalance fix)
  4. FedProx + Class-Aware Aggregation (Rec 1)
  5. FedProx + FedBN (Rec 3)
  6. FedProx + Full System (all fixes + Recs 1,3)
  7. FedProx + Full + DP (with adaptive noise)
  8. FedProx + Full + DP + Compression

Supports --quick mode for fast CPU smoke testing.
"""

import os
import sys
import json
import logging
import time
import copy
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.utils import setup_logging, set_seed, save_metrics
from src.server import run_fl_simulation

logger = logging.getLogger("ppfl-ncd.run_all")


# ============================================================================
# Experiment Definitions
# ============================================================================

def get_experiments(quick_mode: bool = False) -> dict:
    """
    Define all experiment configurations.

    Returns dict of {experiment_name: config_overrides}
    """
    # Base settings
    if quick_mode:
        base = {
            "num_rounds": 5,
            "num_clients": 4,
            "local_epochs": 2,
            "synthetic_samples": 5000,
        }
    else:
        base = {
            "num_rounds": 50,
            "num_clients": 10,
            "local_epochs": 3,
            "synthetic_samples": 50000,
        }

    experiments = {}

    # --- Baselines ---

    # 1. FedAvg baseline (minimal - no enhancements)
    experiments["fedavg_baseline"] = {
        **base,
        "loss_type": "bce",
        "fedprox_mu": 0.0,
        "use_fedbn": False,
        "use_class_aware": False,
        "use_curriculum": False,
        "use_weighted_sampling": False,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "none",
        "learning_rate": 0.001,
    }

    # 2. FedProx baseline (original settings from paper)
    experiments["fedprox_original"] = {
        **base,
        "loss_type": "bce",
        "fedprox_mu": 0.01,
        "use_fedbn": False,
        "use_class_aware": False,
        "use_curriculum": False,
        "use_weighted_sampling": False,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "none",
        "learning_rate": 0.001,
    }

    # --- Ablation: Individual Fixes ---

    # 3. Fix: Focal Loss only
    experiments["fix_focal_loss"] = {
        **base,
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "fedprox_mu": 0.01,
        "use_fedbn": False,
        "use_class_aware": False,
        "use_curriculum": False,
        "use_weighted_sampling": False,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "none",
        "learning_rate": 0.001,
    }

    # 4. Fix: Weighted sampling only
    experiments["fix_weighted_sampling"] = {
        **base,
        "loss_type": "bce",
        "fedprox_mu": 0.01,
        "use_fedbn": False,
        "use_class_aware": False,
        "use_curriculum": False,
        "use_weighted_sampling": True,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "none",
        "learning_rate": 0.001,
    }

    # 5. Fix: LR reduction + warmup cosine
    experiments["fix_lr_schedule"] = {
        **base,
        "loss_type": "bce",
        "fedprox_mu": 0.01,
        "use_fedbn": False,
        "use_class_aware": False,
        "use_curriculum": False,
        "use_weighted_sampling": False,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "warmup_cosine",
        "learning_rate": 0.0003,
    }

    # 6. Fix: Stronger FedProx mu
    experiments["fix_fedprox_mu"] = {
        **base,
        "loss_type": "bce",
        "fedprox_mu": 0.1,
        "use_fedbn": False,
        "use_class_aware": False,
        "use_curriculum": False,
        "use_weighted_sampling": False,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "none",
        "learning_rate": 0.001,
    }

    # --- Novel Contributions (Ablation) ---

    # 7. Rec 1: Class-Aware Aggregation (all fixes + CaFCal)
    experiments["rec1_class_aware"] = {
        **base,
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "fedprox_mu": 0.1,
        "use_fedbn": False,
        "use_class_aware": True,
        "use_curriculum": False,
        "use_weighted_sampling": True,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "warmup_cosine",
        "learning_rate": 0.0003,
    }

    # 8. Rec 3: FedBN (all fixes + FedBN)
    experiments["rec3_fedbn"] = {
        **base,
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "fedprox_mu": 0.1,
        "use_fedbn": True,
        "use_class_aware": False,
        "use_curriculum": False,
        "use_weighted_sampling": True,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "warmup_cosine",
        "learning_rate": 0.0003,
    }

    # --- Combined Systems ---

    # 9. All fixes combined (no DP/compression)
    experiments["all_fixes"] = {
        **base,
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "fedprox_mu": 0.1,
        "use_fedbn": True,
        "use_class_aware": True,
        "use_curriculum": True,
        "use_weighted_sampling": True,
        "enable_dp": False,
        "enable_compression": False,
        "lr_decay_strategy": "warmup_cosine",
        "learning_rate": 0.0003,
    }

    # 10. Full system with DP
    experiments["full_system_dp"] = {
        **base,
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "fedprox_mu": 0.1,
        "use_fedbn": True,
        "use_class_aware": True,
        "use_curriculum": True,
        "use_weighted_sampling": True,
        "enable_dp": True,
        "noise_multiplier": 1.1,
        "enable_compression": False,
        "lr_decay_strategy": "warmup_cosine",
        "learning_rate": 0.0003,
    }

    # 11. Full system with DP + Compression
    experiments["full_system_dp_comp"] = {
        **base,
        "loss_type": "focal",
        "focal_gamma": 2.0,
        "fedprox_mu": 0.1,
        "use_fedbn": True,
        "use_class_aware": True,
        "use_curriculum": True,
        "use_weighted_sampling": True,
        "enable_dp": True,
        "noise_multiplier": 1.1,
        "enable_compression": True,
        "k_ratio": 0.3,
        "lr_decay_strategy": "warmup_cosine",
        "learning_rate": 0.0003,
    }

    return experiments


def apply_overrides(config: ExperimentConfig, overrides: dict) -> ExperimentConfig:
    """Apply experiment-specific overrides to a config."""
    config = copy.deepcopy(config)

    mapping = {
        "num_rounds": ("fl", "num_rounds"),
        "num_clients": ("fl", "num_clients"),
        "local_epochs": ("fl", "local_epochs"),
        "learning_rate": ("fl", "learning_rate"),
        "fedprox_mu": ("fl", "fedprox_mu"),
        "use_fedbn": ("fl", "use_fedbn"),
        "use_class_aware": ("fl", "use_class_aware_aggregation"),
        "use_curriculum": ("fl", "use_curriculum"),
        "use_weighted_sampling": ("fl", "use_weighted_sampling"),
        "lr_decay_strategy": ("fl", "lr_decay_strategy"),
        "loss_type": ("model", "loss_type"),
        "focal_gamma": ("model", "focal_gamma"),
        "enable_dp": ("privacy", "enable_dp"),
        "noise_multiplier": ("privacy", "noise_multiplier"),
        "enable_compression": ("compression", "enable_compression"),
        "k_ratio": ("compression", "k_ratio"),
        "synthetic_samples": ("data", "synthetic_num_samples"),
    }

    for key, value in overrides.items():
        if key in mapping:
            section, attr = mapping[key]
            setattr(getattr(config, section), attr, value)

    return config


# ============================================================================
# Centralized Baseline
# ============================================================================

def run_centralized_baseline(config: ExperimentConfig, use_synthetic: bool = False):
    """Run centralized training as upper bound baseline."""
    from experiments.centralized import train_centralized

    logger.info("=" * 70)
    logger.info("EXPERIMENT: Centralized Baseline (Upper Bound)")
    logger.info("=" * 70)

    results_dir = os.path.join(config.results_dir, "centralized")
    os.makedirs(results_dir, exist_ok=True)

    num_epochs = config.fl.num_rounds  # Use same "compute budget" for fairness

    model, metrics = train_centralized(
        num_epochs=num_epochs,
        batch_size=config.fl.batch_size,
        learning_rate=0.001,  # Standard centralized LR
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        use_synthetic=use_synthetic,
        synthetic_samples=config.data.synthetic_num_samples,
        device=config.device,
        results_dir=results_dir,
        seed=config.seed,
    )

    return metrics


# ============================================================================
# Main Runner
# ============================================================================

def run_all_experiments(
    quick_mode: bool = False,
    use_synthetic: bool = False,
    device: str = "auto",
    experiments_to_run: list = None,
    seed: int = 42,
):
    """
    Run all FL experiments with comprehensive ablation study.

    Args:
        quick_mode: If True, use minimal rounds/clients for smoke testing
        use_synthetic: If True, use synthetic data
        device: "auto", "cpu", "cuda", or "xpu"
        experiments_to_run: Optional list of experiment names (default: all)
        seed: Random seed
    """
    start_time = time.time()

    # Setup
    results_root = f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_root, exist_ok=True)
    setup_logging(results_root)
    set_seed(seed)

    logger.info("=" * 70)
    logger.info(f"FL-NCD COMPREHENSIVE EXPERIMENT SUITE")
    logger.info(f"  Mode: {'QUICK (smoke test)' if quick_mode else 'FULL'}")
    logger.info(f"  Data: {'Synthetic' if use_synthetic else 'Real'}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Results: {results_root}")
    logger.info("=" * 70)

    all_experiments = get_experiments(quick_mode)

    if experiments_to_run:
        experiments = {k: v for k, v in all_experiments.items()
                      if k in experiments_to_run}
    else:
        experiments = all_experiments

    # Results accumulator
    all_results = {}
    experiment_times = {}

    # 1. Centralized baseline
    if experiments_to_run is None or "centralized" in experiments_to_run:
        try:
            base_config = ExperimentConfig()
            base_config.device = device
            base_config.seed = seed
            base_config.results_dir = os.path.join(results_root, "centralized")

            if quick_mode:
                base_config.fl.num_rounds = 5
                base_config.data.synthetic_num_samples = 5000

            t0 = time.time()
            centralized_metrics = run_centralized_baseline(base_config, use_synthetic)
            experiment_times["centralized"] = time.time() - t0
            all_results["centralized"] = centralized_metrics
            logger.info(f"Centralized complete ({experiment_times['centralized']:.0f}s)")
        except Exception as e:
            logger.error(f"Centralized baseline failed: {e}")
            all_results["centralized"] = {"error": str(e)}

    # 2. FL experiments
    for exp_name, overrides in experiments.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT: {exp_name}")
        logger.info(f"{'='*70}")

        try:
            config = ExperimentConfig()
            config.device = device
            config.seed = seed
            config.results_dir = os.path.join(results_root, exp_name)
            config.experiment_name = exp_name
            config = apply_overrides(config, overrides)

            t0 = time.time()
            history, strategy = run_fl_simulation(config, use_synthetic=use_synthetic)
            elapsed = time.time() - t0
            experiment_times[exp_name] = elapsed

            all_results[exp_name] = {
                "best_auc": strategy.best_auc,
                "best_round": strategy.best_round,
                "final_round_metrics": strategy.round_metrics[-1] if strategy.round_metrics else {},
                "config": overrides,
                "elapsed_seconds": elapsed,
            }

            logger.info(
                f"{exp_name} complete | "
                f"Best AUC={strategy.best_auc:.4f} (round {strategy.best_round}) | "
                f"Time={elapsed:.0f}s"
            )

        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {"error": str(e)}
            experiment_times[exp_name] = 0

    # Save complete results
    total_time = time.time() - start_time

    summary = {
        "run_info": {
            "quick_mode": quick_mode,
            "use_synthetic": use_synthetic,
            "device": device,
            "seed": seed,
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
        },
        "experiment_times": experiment_times,
        "results": all_results,
    }

    save_metrics(summary, os.path.join(results_root, "all_results.json"))

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<30} {'Best AUC':>10} {'Time (s)':>10}")
    logger.info("-" * 52)

    for exp_name, result in all_results.items():
        if "error" in result:
            logger.info(f"{exp_name:<30} {'FAILED':>10} {experiment_times.get(exp_name, 0):>10.0f}")
        else:
            auc = result.get("best_auc", result.get("macro_avg", {}).get("auc_roc", 0))
            logger.info(f"{exp_name:<30} {auc:>10.4f} {experiment_times.get(exp_name, 0):>10.0f}")

    logger.info("-" * 52)
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Results saved to: {results_root}/all_results.json")

    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all FL-NCD experiments"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (5 rounds, 4 clients, synthetic)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "xpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiments to run (default: all)")

    args = parser.parse_args()

    # Quick mode auto-enables synthetic
    use_synthetic = args.synthetic or args.quick

    run_all_experiments(
        quick_mode=args.quick,
        use_synthetic=use_synthetic,
        device=args.device,
        experiments_to_run=args.experiments,
        seed=args.seed,
    )
