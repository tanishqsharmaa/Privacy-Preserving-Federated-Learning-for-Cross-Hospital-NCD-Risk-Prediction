"""
Vanilla FedAvg baseline.

Standard Federated Averaging with NO privacy, NO personalization,
NO compression. This is the basic FL baseline to compare against.
"""

import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.model import MultiTaskNCD
from src.server import prepare_data_for_fl, FedProxStrategy
from src.client import create_client_fn
from src.config import ExperimentConfig
from src.utils import setup_logging, set_seed, save_metrics

logger = logging.getLogger("ppfl-ncd.fedavg")


def run_fedavg_baseline(
    num_rounds: int = 50,
    num_clients: int = 10,
    alpha: float = 0.5,
    use_synthetic: bool = True,
    device: str = "auto",
    results_dir: str = "results/fedavg",
    seed: int = 42,
):
    """
    Run vanilla FedAvg (no DP, no FedProx, no compression).
    """
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)
    
    # Config with all privacy/compression disabled
    config = ExperimentConfig()
    config.fl.num_rounds = num_rounds
    config.fl.num_clients = num_clients
    config.fl.dirichlet_alpha = alpha
    config.fl.fedprox_mu = 0.0  # Disable FedProx (mu=0 = vanilla FedAvg)
    config.privacy.enable_dp = False  # No DP
    config.compression.enable_compression = False  # No compression
    config.device = device
    config.seed = seed
    config.results_dir = results_dir
    
    if use_synthetic:
        config.data.synthetic_num_samples = 50000
    
    logger.info("=" * 60)
    logger.info("VANILLA FedAvg BASELINE (No DP, No FedProx)")
    logger.info(f"  Rounds: {num_rounds}, Clients: {num_clients}, alpha: {alpha}")
    logger.info("=" * 60)
    
    # Prepare data
    client_data, test_data, input_dim = prepare_data_for_fl(config, use_synthetic)
    
    # Model
    model = MultiTaskNCD(input_dim, config.model.hidden_dims, config.model.dropout)
    
    # Strategy (FedProx with mu=0 = FedAvg)
    strategy = FedProxStrategy(
        model=model,
        test_data=test_data,
        config=config,
    )
    
    # Client function
    client_fn = create_client_fn(
        train_data_partitions=client_data,
        val_data=test_data,
        input_dim=input_dim,
        config=config,
    )
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Save results
    results = {
        "method": "fedavg",
        "best_auc": strategy.best_auc,
        "best_round": strategy.best_round,
        "round_metrics": strategy.round_metrics,
    }
    save_metrics(results, os.path.join(results_dir, "fedavg_results.json"))
    
    logger.info(f"FedAvg Best AUC: {strategy.best_auc:.4f} (Round {strategy.best_round})")
    
    return history, strategy


def run_fedavg_dp_baseline(
    num_rounds: int = 50,
    num_clients: int = 10,
    alpha: float = 0.5,
    noise_multiplier: float = 1.1,
    use_synthetic: bool = True,
    device: str = "auto",
    results_dir: str = "results/fedavg_dp",
    seed: int = 42,
):
    """
    FedAvg + Basic DP baseline (no FedProx, no compression, no SecAgg).
    """
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)
    
    config = ExperimentConfig()
    config.fl.num_rounds = num_rounds
    config.fl.num_clients = num_clients
    config.fl.dirichlet_alpha = alpha
    config.fl.fedprox_mu = 0.0  # No FedProx
    config.privacy.enable_dp = True  # DP enabled
    config.privacy.noise_multiplier = noise_multiplier
    config.compression.enable_compression = False  # No compression
    config.device = device
    config.seed = seed
    config.results_dir = results_dir
    
    if use_synthetic:
        config.data.synthetic_num_samples = 50000
    
    logger.info("=" * 60)
    logger.info(f"FedAvg + DP BASELINE (sigma={noise_multiplier})")
    logger.info("=" * 60)
    
    # ⚠️ DP training is 3-5x slower — GPU recommended!
    client_data, test_data, input_dim = prepare_data_for_fl(config, use_synthetic)
    model = MultiTaskNCD(input_dim, config.model.hidden_dims, config.model.dropout)
    
    strategy = FedProxStrategy(model=model, test_data=test_data, config=config)
    client_fn = create_client_fn(
        train_data_partitions=client_data,
        val_data=test_data,
        input_dim=input_dim,
        config=config,
    )
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    results = {
        "method": "fedavg_dp",
        "noise_multiplier": noise_multiplier,
        "best_auc": strategy.best_auc,
        "best_round": strategy.best_round,
        "round_metrics": strategy.round_metrics,
    }
    save_metrics(results, os.path.join(results_dir, "fedavg_dp_results.json"))
    
    return history, strategy


def run_local_only_baseline(
    num_epochs: int = 50,
    num_clients: int = 10,
    alpha: float = 0.5,
    use_synthetic: bool = True,
    device: str = "auto",
    results_dir: str = "results/local_only",
    seed: int = 42,
):
    """
    Local-only baseline: each hospital trains independently (no federation).
    This is the LOWER BOUND on accuracy.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import MultiTaskLoss
    from src.utils import compute_multitask_metrics
    
    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)
    
    config = ExperimentConfig()
    config.fl.num_clients = num_clients
    config.fl.dirichlet_alpha = alpha
    config.device = device
    
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    
    client_data, test_data, input_dim = prepare_data_for_fl(config, use_synthetic)
    X_test, Y_test = test_data
    
    logger.info("=" * 60)
    logger.info("LOCAL-ONLY BASELINE (No Federation)")
    logger.info("=" * 60)
    
    all_client_metrics = {}
    
    for cid, (X_train, Y_train) in enumerate(client_data):
        logger.info(f"\nClient {cid}: Training on {len(X_train)} samples...")
        
        model = MultiTaskNCD(input_dim).to(device_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = MultiTaskLoss()
        
        dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
        
        for epoch in range(num_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device_t), yb.to(device_t)
                targets = (yb[:, 0:1], yb[:, 1:2], yb[:, 2:3])
                preds = model(xb)
                loss, _ = loss_fn(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test).to(device_t))
            preds_np = [p.cpu().numpy().flatten() for p in preds]
        
        y_true_list = [Y_test[:, i] for i in range(3)]
        metrics = compute_multitask_metrics(y_true_list, preds_np)
        all_client_metrics[cid] = metrics
        
        logger.info(f"  Client {cid} AUC (macro): {metrics['macro_avg']['auc_roc']:.4f}")
    
    # Average across clients
    avg_auc = np.mean([m["macro_avg"]["auc_roc"] for m in all_client_metrics.values()])
    
    results = {
        "method": "local_only",
        "avg_macro_auc": float(avg_auc),
        "per_client": {str(k): v for k, v in all_client_metrics.items()},
    }
    save_metrics(results, os.path.join(results_dir, "local_only_results.json"))
    
    logger.info(f"\nLocal-Only Avg AUC: {avg_auc:.4f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="fedavg",
                        choices=["fedavg", "fedavg_dp", "local_only"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    setup_logging(f"results/{args.baseline}")
    
    if args.baseline == "fedavg":
        run_fedavg_baseline(
            num_rounds=args.rounds, num_clients=args.clients,
            alpha=args.alpha, use_synthetic=args.synthetic,
            device=args.device, seed=args.seed,
        )
    elif args.baseline == "fedavg_dp":
        run_fedavg_dp_baseline(
            num_rounds=args.rounds, num_clients=args.clients,
            alpha=args.alpha, use_synthetic=args.synthetic,
            device=args.device, seed=args.seed,
        )
    elif args.baseline == "local_only":
        run_local_only_baseline(
            num_epochs=args.rounds, num_clients=args.clients,
            alpha=args.alpha, use_synthetic=args.synthetic,
            device=args.device, seed=args.seed,
        )
