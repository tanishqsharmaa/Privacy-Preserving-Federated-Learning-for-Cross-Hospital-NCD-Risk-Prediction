"""
Run all experiments and generate comparison results.

Orchestrates all 5 baselines:
1. Centralized (upper bound)
2. Local Only (lower bound)
3. Vanilla FedAvg
4. FedAvg + Basic DP
5. Full System (FedProx + DP-SGD + Compression + SHAP)

Also runs:
- Privacy-utility tradeoff sweep
- Non-IID robustness sweep (alpha values)
- Compression efficiency sweep (k_ratio values)
- Attack simulations
- Fairness analysis
- SHAP explainability

⚠️ FULL EXPERIMENT SUITE TAKES SEVERAL HOURS ON GPU.
    Switch to your high-end PC before running this.
"""

import os
import sys
import logging
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.utils import (
    setup_logging, set_seed, save_metrics, load_metrics,
    plot_privacy_utility_tradeoff, plot_noniid_robustness,
    plot_communication_efficiency, plot_baseline_comparison,
    plot_convergence_curve, compute_fairness_report, DISEASE_NAMES
)

logger = logging.getLogger("ppfl-ncd.run_all")


def run_all_experiments(
    use_synthetic: bool = True,
    device: str = "auto",
    seed: int = 42,
    quick_mode: bool = False,  # Reduced rounds for testing
):
    """
    Run the complete experiment suite.
    
    ⚠️ GPU REQUIRED for DP experiments.
    ⚠️ Estimated time: 4-8 hours on GPU, 12+ hours on CPU.
    
    Set quick_mode=True for a fast test run (5 rounds, 3 clients).
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Quick mode for testing
    num_rounds = 5 if quick_mode else 50
    num_clients = 3 if quick_mode else 10
    
    all_results = {}
    
    # ================================================================
    # BASELINE 1: Centralized (Upper Bound)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1/5: CENTRALIZED BASELINE")
    logger.info("=" * 70)
    
    from experiments.centralized import train_centralized
    _, centralized_metrics = train_centralized(
        num_epochs=num_rounds,
        use_synthetic=use_synthetic,
        device=device,
        seed=seed,
        results_dir=os.path.join(results_dir, "centralized"),
    )
    all_results["centralized"] = centralized_metrics
    
    # ================================================================
    # BASELINE 2: Local Only (Lower Bound)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2/5: LOCAL-ONLY BASELINE")
    logger.info("=" * 70)
    
    from experiments.fedavg import run_local_only_baseline
    local_results = run_local_only_baseline(
        num_epochs=num_rounds,
        num_clients=num_clients,
        use_synthetic=use_synthetic,
        device=device,
        seed=seed,
        results_dir=os.path.join(results_dir, "local_only"),
    )
    all_results["local_only"] = local_results
    
    # ================================================================
    # BASELINE 3: Vanilla FedAvg
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3/5: VANILLA FEDAVG")
    logger.info("=" * 70)
    
    from experiments.fedavg import run_fedavg_baseline
    _, fedavg_strategy = run_fedavg_baseline(
        num_rounds=num_rounds,
        num_clients=num_clients,
        use_synthetic=use_synthetic,
        device=device,
        seed=seed,
        results_dir=os.path.join(results_dir, "fedavg"),
    )
    all_results["fedavg"] = {"best_auc": fedavg_strategy.best_auc}
    
    # ================================================================
    # BASELINE 4: FedAvg + DP
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4/5: FEDAVG + DP")
    logger.info("=" * 70)
    
    from experiments.fedavg import run_fedavg_dp_baseline
    _, fedavg_dp_strategy = run_fedavg_dp_baseline(
        num_rounds=num_rounds,
        num_clients=num_clients,
        use_synthetic=use_synthetic,
        device=device,
        seed=seed,
        results_dir=os.path.join(results_dir, "fedavg_dp"),
    )
    all_results["fedavg_dp"] = {"best_auc": fedavg_dp_strategy.best_auc}
    
    # ================================================================
    # EXPERIMENT 5: Full System (Our Approach)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5/5: FULL SYSTEM (FedProx + DP + Compression)")
    logger.info("=" * 70)
    
    from src.server import run_fl_simulation
    config = ExperimentConfig()
    config.fl.num_rounds = num_rounds
    config.fl.num_clients = num_clients
    config.privacy.enable_dp = True
    config.compression.enable_compression = True
    config.device = device
    config.seed = seed
    config.results_dir = os.path.join(results_dir, "full_system")
    
    _, full_strategy = run_fl_simulation(config, use_synthetic=use_synthetic)
    all_results["full_system"] = {"best_auc": full_strategy.best_auc}
    
    # ================================================================
    # SWEEP: Privacy-Utility Tradeoff
    # ================================================================
    if not quick_mode:
        logger.info("\n" + "=" * 70)
        logger.info("SWEEP: PRIVACY-UTILITY TRADEOFF")
        logger.info("=" * 70)
        
        noise_multipliers = [0.5, 0.8, 1.1, 1.5, 2.0]
        epsilons = []
        auc_per_disease = {d: [] for d in DISEASE_NAMES}
        
        for sigma in noise_multipliers:
            logger.info(f"\n  Running with sigma={sigma}...")
            sweep_config = ExperimentConfig()
            sweep_config.fl.num_rounds = num_rounds
            sweep_config.fl.num_clients = num_clients
            sweep_config.privacy.noise_multiplier = sigma
            sweep_config.device = device
            sweep_config.seed = seed
            sweep_config.results_dir = os.path.join(results_dir, f"sweep_sigma_{sigma}")
            
            _, sweep_strategy = run_fl_simulation(sweep_config, use_synthetic=True)
            
            # Get final epsilon from round metrics
            if sweep_strategy.round_metrics:
                last_round = sweep_strategy.round_metrics[-1]
                epsilons.append(last_round.get("max_epsilon", 0))
                for d in DISEASE_NAMES:
                    auc_per_disease[d].append(
                        last_round.get(f"test_{d}_auc_roc", 0)
                    )
        
        if epsilons:
            plot_privacy_utility_tradeoff(
                epsilons, auc_per_disease,
                save_path=os.path.join(results_dir, "privacy_utility_tradeoff.png")
            )
    
    # ================================================================
    # SWEEP: Non-IID Robustness
    # ================================================================
    if not quick_mode:
        logger.info("\n" + "=" * 70)
        logger.info("SWEEP: NON-IID ROBUSTNESS (alpha values)")
        logger.info("=" * 70)
        
        alphas = [0.1, 0.5, 1.0]
        fedavg_scores = []
        fedprox_scores = []
        
        for alpha in alphas:
            logger.info(f"\n  alpha={alpha}...")
            
            # FedAvg
            _, fa_strat = run_fedavg_baseline(
                num_rounds=num_rounds, num_clients=num_clients,
                alpha=alpha, use_synthetic=True, device=device, seed=seed,
                results_dir=os.path.join(results_dir, f"sweep_alpha_{alpha}_fedavg"),
            )
            fedavg_scores.append(fa_strat.best_auc)
            
            # FedProx (our system without DP for fair comparison)
            fp_config = ExperimentConfig()
            fp_config.fl.num_rounds = num_rounds
            fp_config.fl.num_clients = num_clients
            fp_config.fl.dirichlet_alpha = alpha
            fp_config.privacy.enable_dp = False
            fp_config.compression.enable_compression = False
            fp_config.device = device
            fp_config.seed = seed
            fp_config.results_dir = os.path.join(results_dir, f"sweep_alpha_{alpha}_fedprox")
            
            _, fp_strat = run_fl_simulation(fp_config, use_synthetic=True)
            fedprox_scores.append(fp_strat.best_auc)
        
        plot_noniid_robustness(
            alphas, fedavg_scores, fedprox_scores,
            save_path=os.path.join(results_dir, "noniid_robustness.png")
        )
    
    # ================================================================
    # ATTACK SIMULATION
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ATTACK SIMULATION")
    logger.info("=" * 70)
    
    from src.attack_sim import run_all_attacks
    from src.data_prep import HARMONIZED_FEATURES, TARGET_COLUMNS
    from src.server import prepare_data_for_fl
    
    attack_config = ExperimentConfig()
    attack_config.fl.num_clients = num_clients
    attack_config.device = device
    client_data, test_data, input_dim = prepare_data_for_fl(
        attack_config, use_synthetic=True
    )
    
    # Load best model
    model = torch.load(
        os.path.join(results_dir, "full_system", "best_model.pth"),
        map_location="cpu"
    ) if os.path.exists(os.path.join(results_dir, "full_system", "best_model.pth")) else None
    
    if model is not None:
        from src.model import MultiTaskNCD
        attack_model = MultiTaskNCD(input_dim)
        attack_model.load_state_dict(model)
        
        X_train_all = np.concatenate([x for x, _ in client_data])
        Y_train_all = np.concatenate([y for _, y in client_data])
        
        attack_results = run_all_attacks(
            attack_model, X_train_all, Y_train_all, test_data[0],
            device=device,
            results_dir=os.path.join(results_dir, "attacks"),
        )
        all_results["attacks"] = attack_results
    
    # ================================================================
    # SAVE FINAL COMPARISON
    # ================================================================
    save_metrics(all_results, os.path.join(results_dir, "all_results.json"))
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to: {results_dir}/")
    logger.info("=" * 70)
    
    # Print summary
    logger.info("\n=== RESULTS SUMMARY ===")
    for method, res in all_results.items():
        if isinstance(res, dict) and "best_auc" in res:
            logger.info(f"  {method}: Best AUC = {res['best_auc']:.4f}")
        elif isinstance(res, dict) and "macro_avg" in res:
            logger.info(f"  {method}: Macro AUC = {res['macro_avg']['auc_roc']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (5 rounds, 3 clients)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    setup_logging("results")
    
    run_all_experiments(
        use_synthetic=True,
        device=args.device,
        seed=args.seed,
        quick_mode=args.quick,
    )
