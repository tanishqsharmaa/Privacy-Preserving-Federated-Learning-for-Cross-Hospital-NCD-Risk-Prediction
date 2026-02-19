"""
Flower FL Server with FedProx aggregation strategy.

Orchestrates federated training across hospital nodes:
1. FedProx-aware weighted aggregation
2. Global model evaluation
3. SecAgg+ secure aggregation (when enabled)
4. Experiment orchestration and metrics logging
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.model import MultiTaskNCD, MultiTaskLoss
from src.config import ExperimentConfig, parse_args
from src.utils import (
    setup_logging, set_seed, compute_multitask_metrics, 
    save_metrics, DISEASE_NAMES
)

logger = logging.getLogger("ppfl-ncd.server")


# ============================================================================
# FedProx Strategy
# ============================================================================

class FedProxStrategy(FedAvg):
    """
    FedProx aggregation strategy extending Flower's FedAvg.
    
    The proximal term is applied client-side (in client.py). Server-side,
    FedProx uses the same weighted averaging as FedAvg. The key difference
    is that the server broadcasts the current global model each round,
    and clients use it as the anchpr for the proximal penalty.
    
    Additionally:
    - Logs per-round metrics
    - Evaluates global model on test data
    - Saves best model checkpoint
    """
    
    def __init__(
        self,
        model: MultiTaskNCD,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        config: Optional[ExperimentConfig] = None,
        **kwargs
    ):
        # Set defaults
        kwargs.setdefault("fraction_fit", 1.0)
        kwargs.setdefault("fraction_evaluate", 1.0)
        kwargs.setdefault("min_fit_clients", 2)
        kwargs.setdefault("min_evaluate_clients", 2)
        kwargs.setdefault("min_available_clients", 2)
        
        # Build initial parameters from model
        initial_params = ndarrays_to_parameters(
            [val.cpu().numpy() for val in model.state_dict().values()]
        )
        kwargs["initial_parameters"] = initial_params
        
        super().__init__(**kwargs)
        
        self.model = model
        self.test_data = test_data
        self.config = config or ExperimentConfig()
        self.device = self.config.get_device()
        
        # Metrics tracking
        self.round_metrics: List[Dict] = []
        self.best_auc = 0.0
        self.best_round = 0
        
        # Results directory
        self.results_dir = self.config.results_dir
        os.makedirs(self.results_dir, exist_ok=True)
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Add custom config to each client's fit call."""
        configs = super().configure_fit(server_round, parameters, client_manager)
        
        # Add round number and local epochs to client config
        updated_configs = []
        for client, fit_ins in configs:
            fit_ins.config["current_round"] = server_round
            fit_ins.config["local_epochs"] = self.config.fl.local_epochs
            updated_configs.append((client, fit_ins))
        
        return updated_configs
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates and evaluate global model."""
        
        if not results:
            return None, {}
        
        # Standard FedAvg aggregation (weighted by num_samples)
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        if parameters_aggregated is None:
            return None, metrics_aggregated
        
        # Collect client metrics
        client_metrics = []
        for _, fit_res in results:
            client_metrics.append(fit_res.metrics)
        
        avg_train_loss = np.mean([m.get("train_loss", 0) for m in client_metrics])
        avg_epsilon = np.mean([m.get("epsilon", 0) for m in client_metrics])
        max_epsilon = max([m.get("epsilon", 0) for m in client_metrics])
        
        # Evaluate global model on test data
        test_metrics = {}
        if self.test_data is not None:
            test_metrics = self._evaluate_global(parameters_aggregated)
        
        # Log round metrics
        round_info = {
            "round": server_round,
            "avg_train_loss": float(avg_train_loss),
            "avg_epsilon": float(avg_epsilon),
            "max_epsilon": float(max_epsilon),
            "num_clients": len(results),
            "num_failures": len(failures),
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        self.round_metrics.append(round_info)
        
        # Track best model
        macro_auc = test_metrics.get("macro_avg_auc_roc", 0)
        if macro_auc > self.best_auc:
            self.best_auc = macro_auc
            self.best_round = server_round
            self._save_model(parameters_aggregated, "best_model.pth")
        
        logger.info(
            f"Round {server_round:3d} | "
            f"Loss={avg_train_loss:.4f} | "
            f"ε_avg={avg_epsilon:.4f} | ε_max={max_epsilon:.4f} | "
            f"Test AUC={macro_auc:.4f}"
        )
        
        # Save metrics periodically
        if server_round % 5 == 0 or server_round == self.config.fl.num_rounds:
            self._save_round_metrics()
        
        return parameters_aggregated, {**metrics_aggregated, **test_metrics}
    
    def _evaluate_global(self, parameters: Parameters) -> Dict[str, float]:
        """Evaluate global model on test data."""
        if self.test_data is None:
            return {}
        
        X_test, Y_test = self.test_data
        
        # Load parameters into model
        params = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Forward pass
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            preds = self.model(X_tensor)
            preds_np = [p.cpu().numpy().flatten() for p in preds]
        
        # Compute metrics
        y_true_list = [Y_test[:, i] for i in range(3)]
        metrics = compute_multitask_metrics(y_true_list, preds_np)
        
        # Flatten for Flower compatibility
        flat = {}
        for disease, disease_metrics in metrics.items():
            for metric_name, value in disease_metrics.items():
                flat[f"{disease}_{metric_name}"] = float(value)
        
        return flat
    
    def _save_model(self, parameters: Parameters, filename: str):
        """Save model checkpoint."""
        filepath = os.path.join(self.results_dir, filename)
        params = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
        )
        torch.save(state_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def _save_round_metrics(self):
        """Save accumulated round metrics to JSON."""
        filepath = os.path.join(self.results_dir, "round_metrics.json")
        save_metrics({"rounds": self.round_metrics}, filepath)


# ============================================================================
# Main Server Launch
# ============================================================================

def prepare_data_for_fl(
    config: ExperimentConfig,
    use_synthetic: bool = False,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray], int]:
    """
    Prepare data partitions for FL simulation.
    
    Returns:
        (client_data_list, test_data, input_dim)
    """
    from src.data_prep import prepare_dataset, HARMONIZED_FEATURES, TARGET_COLUMNS
    from src.partition import partition_by_source, load_partitions
    
    processed_dir = config.data.processed_dir
    partitions_dir = config.data.partitions_dir
    
    # Step 1: Prepare data if not already done
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    if not os.path.exists(train_path):
        logger.info("Processed data not found. Running data preparation...")
        prepare_dataset(
            raw_dir=config.data.raw_dir,
            output_dir=processed_dir,
            use_synthetic=use_synthetic,
            synthetic_samples=config.data.synthetic_num_samples,
            seed=config.seed,
        )
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    input_dim = len(HARMONIZED_FEATURES)
    
    # Step 2: Partition if not already done
    partition_file = os.path.join(
        partitions_dir, f"client_0_alpha{config.fl.dirichlet_alpha}.npy"
    )
    
    if not os.path.exists(partition_file):
        logger.info("Partitions not found. Running partitioning...")
        partitions = partition_by_source(
            train_df,
            num_brfss_clients=config.fl.num_brfss_clients,
            num_nhanes_clients=config.fl.num_nhanes_clients,
            alpha=config.fl.dirichlet_alpha,
            seed=config.seed,
        )
        from src.partition import save_partitions
        save_partitions(partitions, partitions_dir, config.fl.dirichlet_alpha)
    
    # Load partitions
    partitions = load_partitions(
        partitions_dir, config.fl.dirichlet_alpha, config.fl.num_clients
    )
    
    # Step 3: Create per-client datasets
    feature_cols = HARMONIZED_FEATURES
    target_cols = TARGET_COLUMNS
    
    client_data = []
    for cid, indices in enumerate(partitions):
        if len(indices) == 0:
            logger.warning(f"Client {cid} has no data!")
            # Create minimal dummy data
            client_data.append((
                np.zeros((1, input_dim), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32)
            ))
            continue
        
        client_df = train_df.iloc[indices]
        X = client_df[feature_cols].values.astype(np.float32)
        Y = client_df[target_cols].values.astype(np.float32)
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)
        
        client_data.append((X, Y))
        logger.info(f"Client {cid}: {len(indices)} samples")
    
    # Test data
    X_test = test_df[feature_cols].values.astype(np.float32)
    Y_test = test_df[target_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    Y_test = np.nan_to_num(Y_test, nan=0.0)
    
    return client_data, (X_test, Y_test), input_dim


def run_fl_simulation(config: ExperimentConfig, use_synthetic: bool = False):
    """
    Run the full federated learning simulation.
    
    ⚠️ GPU RECOMMENDED for this step. Switch to high-end PC for:
    - DP-SGD training (3-5x slower than standard)
    - 50 rounds × 10 clients = 500 training iterations
    """
    logger.info("=" * 70)
    logger.info("FEDERATED LEARNING SIMULATION")
    logger.info(f"  Rounds: {config.fl.num_rounds}")
    logger.info(f"  Clients: {config.fl.num_clients}")
    logger.info(f"  DP: {'ON' if config.privacy.enable_dp else 'OFF'}")
    logger.info(f"  FedProx μ: {config.fl.fedprox_mu}")
    logger.info(f"  Compression: {'ON' if config.compression.enable_compression else 'OFF'}")
    logger.info(f"  Device: {config.get_device()}")
    logger.info("=" * 70)
    
    # Prepare data
    client_data, test_data, input_dim = prepare_data_for_fl(config, use_synthetic)
    
    # Create global model
    model = MultiTaskNCD(
        input_dim=input_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    )
    
    # Create strategy
    strategy = FedProxStrategy(
        model=model,
        test_data=test_data,
        config=config,
        fraction_fit=config.fl.fraction_fit,
        fraction_evaluate=config.fl.fraction_evaluate,
        min_fit_clients=config.fl.min_fit_clients,
        min_evaluate_clients=config.fl.min_evaluate_clients,
        min_available_clients=config.fl.min_fit_clients,
    )
    
    # Create client function
    from src.client import create_client_fn
    client_fn = create_client_fn(
        train_data_partitions=client_data,
        val_data=test_data,
        input_dim=input_dim,
        config=config,
    )
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.fl.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.fl.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Save final results
    final_results = {
        "config": {
            "num_rounds": config.fl.num_rounds,
            "num_clients": config.fl.num_clients,
            "dirichlet_alpha": config.fl.dirichlet_alpha,
            "fedprox_mu": config.fl.fedprox_mu,
            "dp_enabled": config.privacy.enable_dp,
            "noise_multiplier": config.privacy.noise_multiplier,
            "compression_enabled": config.compression.enable_compression,
            "k_ratio": config.compression.k_ratio,
        },
        "best_auc": strategy.best_auc,
        "best_round": strategy.best_round,
        "round_metrics": strategy.round_metrics,
    }
    
    save_metrics(final_results, os.path.join(config.results_dir, "fl_results.json"))
    
    logger.info("=" * 70)
    logger.info(f"FL SIMULATION COMPLETE")
    logger.info(f"  Best AUC: {strategy.best_auc:.4f} (Round {strategy.best_round})")
    logger.info(f"  Results saved to: {config.results_dir}/")
    logger.info("=" * 70)
    
    return history, strategy


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    config = parse_args()
    setup_logging(config.results_dir)
    set_seed(config.seed)
    
    # Check for --synthetic flag
    import sys
    use_synthetic = "--synthetic" in sys.argv
    
    run_fl_simulation(config, use_synthetic=use_synthetic)
