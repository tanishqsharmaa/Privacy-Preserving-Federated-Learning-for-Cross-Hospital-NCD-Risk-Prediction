"""
Flower FL Server with Class-Aware FedProx Aggregation.

Orchestrates federated training across hospital nodes:
1. FedProx-aware weighted aggregation with class-aware task head weighting
2. FedBN: selective parameter aggregation (skip normalization layers)
3. Server-coordinated learning rate schedule
4. Global model evaluation
5. Experiment orchestration and metrics logging

Key improvements:
  - Class-aware aggregation: weight client contributions per task head
    by inverse minority-class prevalence (Rec 1)
  - FedBN: normalization params kept local per client (Rec 3)
  - Server-coordinated LR schedule (warmup + cosine decay)
"""

import os
import json
import logging
import math
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
from src.config import ExperimentConfig, parse_args, compute_lr_for_round
from src.utils import (
    setup_logging, set_seed, compute_multitask_metrics,
    save_metrics, DISEASE_NAMES
)

logger = logging.getLogger("ppfl-ncd.server")


# ============================================================================
# FedProx Strategy with Class-Aware Aggregation + FedBN
# ============================================================================

class FedProxStrategy(FedAvg):
    """
    FedProx aggregation strategy extending Flower's FedAvg.

    Enhancements over vanilla FedAvg:
    - Class-aware aggregation: clients with more minority-class samples
      have higher weight for the corresponding task head (Rec 1)
    - FedBN: normalization layer params excluded from aggregation (Rec 3)
    - Server-coordinated LR schedule broadcast to clients
    - Early stopping based on macro AUC-ROC
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

        # Get aggregation mask for FedBN
        self.use_fedbn = getattr(self.config.fl, 'use_fedbn', False)
        self.aggregation_mask = model.get_aggregation_mask() if self.use_fedbn else {}

        # Class-aware aggregation
        self.use_class_aware = getattr(self.config.fl, 'use_class_aware_aggregation', False)

        # Metrics tracking
        self.round_metrics: List[Dict] = []
        self.best_auc = 0.0
        self.best_round = 0

        # Early stopping
        self.early_stop_patience = getattr(self.config.fl, 'early_stop_patience', 10)
        self._rounds_without_improvement = 0
        self.should_stop = False

        # Results directory
        self.results_dir = self.config.results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info(
            f"Strategy initialized: FedBN={'ON' if self.use_fedbn else 'OFF'}, "
            f"ClassAware={'ON' if self.use_class_aware else 'OFF'}, "
            f"LR_schedule={self.config.fl.lr_decay_strategy}"
        )

    def configure_fit(self, server_round, parameters, client_manager):
        """Add custom config to each client's fit call."""
        configs = super().configure_fit(server_round, parameters, client_manager)

        # Compute server-coordinated learning rate
        current_lr = compute_lr_for_round(
            base_lr=self.config.fl.learning_rate,
            current_round=server_round,
            total_rounds=self.config.fl.num_rounds,
            strategy=self.config.fl.lr_decay_strategy,
            warmup_rounds=getattr(self.config.fl, 'lr_warmup_rounds', 5),
        )

        # Add round config to each client
        updated_configs = []
        for client, fit_ins in configs:
            fit_ins.config["current_round"] = server_round
            fit_ins.config["local_epochs"] = self.config.fl.local_epochs
            fit_ins.config["learning_rate"] = current_lr
            updated_configs.append((client, fit_ins))

        logger.info(f"Round {server_round}: broadcasting lr={current_lr:.6f}")
        return updated_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates with class-aware weighting and FedBN."""

        if not results:
            return None, {}

        # Collect client metrics for class-aware weighting
        client_metrics = []
        for _, fit_res in results:
            client_metrics.append(fit_res.metrics)

        avg_train_loss = np.mean([m.get("train_loss", 0) for m in client_metrics])
        avg_epsilon = np.mean([m.get("epsilon", 0) for m in client_metrics])
        max_epsilon = max([m.get("epsilon", 0) for m in client_metrics])

        # Decide aggregation method
        if self.use_class_aware and self._has_class_stats(client_metrics):
            parameters_aggregated = self._class_aware_aggregate(results, client_metrics)
        else:
            # Standard FedAvg aggregation
            agg_params, _ = super().aggregate_fit(server_round, results, failures)
            parameters_aggregated = agg_params

        if parameters_aggregated is None:
            return None, {}

        # If FedBN: restore normalization params from first client
        # (in practice each client keeps its own; here we avoid
        #  server overwriting them by not aggregating those params)
        if self.use_fedbn and self.aggregation_mask:
            parameters_aggregated = self._apply_fedbn(
                parameters_aggregated, results
            )

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

        # Track best model and early stopping
        macro_auc = test_metrics.get("macro_avg_auc_roc", 0)
        if macro_auc > self.best_auc:
            self.best_auc = macro_auc
            self.best_round = server_round
            self._rounds_without_improvement = 0
            self._save_model(parameters_aggregated, "best_model.pth")
        else:
            self._rounds_without_improvement += 1
            if self._rounds_without_improvement >= self.early_stop_patience:
                logger.info(
                    f"Early stopping: no improvement for "
                    f"{self.early_stop_patience} rounds "
                    f"(best AUC={self.best_auc:.4f} at round {self.best_round})"
                )
                self.should_stop = True

        # Log metrics for F1 and AUC separately
        macro_f1 = test_metrics.get("macro_avg_f1", 0)
        logger.info(
            f"Round {server_round:3d} | "
            f"Loss={avg_train_loss:.4f} | "
            f"eps_max={max_epsilon:.4f} | "
            f"AUC={macro_auc:.4f} | F1={macro_f1:.4f}"
        )

        # Save metrics periodically
        if server_round % 5 == 0 or server_round == self.config.fl.num_rounds:
            self._save_round_metrics()

        return parameters_aggregated, {**test_metrics}

    def _has_class_stats(self, client_metrics: List[Dict]) -> bool:
        """Check if clients reported class statistics."""
        return all("n_pos_diabetes" in m for m in client_metrics)

    def _class_aware_aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        client_metrics: List[Dict],
    ) -> Parameters:
        """
        Class-aware aggregation (Rec 1):
        - Encoder params: weighted by total sample count (standard FedAvg)
        - Task head params: weighted by minority class count for that task
        """
        model_keys = list(self.model.state_dict().keys())
        head_param_names = self.model.get_head_param_names()

        # Collect all client parameters
        client_params = []
        for _, fit_res in results:
            client_params.append(parameters_to_ndarrays(fit_res.parameters))

        # Build weight arrays per task
        task_weights = {}
        total_samples = []

        for i, metrics in enumerate(client_metrics):
            total_samples.append(metrics.get("num_samples", 1))

        total_samples = np.array(total_samples, dtype=np.float64)
        total_samples = total_samples / total_samples.sum()  # normalize

        # Per-task weights based on minority class count
        task_name_map = {
            "diabetes_head": "diabetes",
            "hypertension_head": "hypertension",
            "cvd_head": "cvd",
        }

        for head_name, task_short in task_name_map.items():
            minority_counts = []
            for metrics in client_metrics:
                minority_counts.append(metrics.get(f"n_pos_{task_short}", 1.0))
            minority_counts = np.array(minority_counts, dtype=np.float64)
            # Weight proportional to minority class count
            if minority_counts.sum() > 0:
                task_weights[head_name] = minority_counts / minority_counts.sum()
            else:
                task_weights[head_name] = total_samples

        task_weights["encoder"] = total_samples

        # Aggregate parameter by parameter
        aggregated = []
        for param_idx, key in enumerate(model_keys):
            # Determine which component this param belongs to
            component = "encoder"
            for comp_name, param_names in head_param_names.items():
                if key in param_names:
                    component = comp_name
                    break

            weights = task_weights.get(component, total_samples)

            # Weighted average
            weighted_sum = np.zeros_like(client_params[0][param_idx])
            for client_idx, params in enumerate(client_params):
                if param_idx < len(params):
                    weighted_sum += weights[client_idx] * params[param_idx]

            aggregated.append(weighted_sum)

        return ndarrays_to_parameters(aggregated)

    def _apply_fedbn(
        self,
        parameters: Parameters,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Parameters:
        """
        FedBN: restore normalization params (they should NOT be aggregated).
        Since each client keeps its own BN stats, the server should send
        back the aggregated non-BN params but leave BN params as-is.

        In practice, we only aggregate non-norm params and keep norm params
        from the last state.
        """
        model_keys = list(self.model.state_dict().keys())
        aggregated = parameters_to_ndarrays(parameters)
        current_state = [v.cpu().numpy() for v in self.model.state_dict().values()]

        for idx, key in enumerate(model_keys):
            should_aggregate = self.aggregation_mask.get(key, True)
            if not should_aggregate:
                # Keep current (non-aggregated) normalization params
                if idx < len(current_state):
                    aggregated[idx] = current_state[idx]

        return ndarrays_to_parameters(aggregated)

    def _evaluate_global(self, parameters: Parameters) -> Dict[str, float]:
        """Evaluate global model on test data (batched to avoid OOM)."""
        if self.test_data is None:
            return {}

        X_test, Y_test = self.test_data

        # Load parameters into model
        params = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {k: torch.from_numpy(v).float() for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Batched forward pass
        batch_size = 512
        all_preds = [[] for _ in range(3)]

        with torch.no_grad():
            for start in range(0, len(X_test), batch_size):
                end = min(start + batch_size, len(X_test))
                X_batch = torch.FloatTensor(X_test[start:end]).to(self.device)
                preds = self.model(X_batch)
                for i, p in enumerate(preds):
                    all_preds[i].append(p.cpu().numpy().flatten())

        preds_np = [np.concatenate(task_preds) for task_preds in all_preds]

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
    from src.partition import partition_by_source

    processed_dir = config.data.processed_dir
    partitions_dir = config.data.partitions_dir

    # Dynamically split clients across sources
    num_clients = config.fl.num_clients
    num_brfss = max(1, num_clients // 2)
    num_nhanes = num_clients - num_brfss

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
    train_df = train_df.reset_index(drop=True)

    input_dim = len(HARMONIZED_FEATURES)
    n_train = len(train_df)

    # Step 2: Partition
    logger.info(f"Partitioning {n_train} samples into {num_clients} clients "
                f"(BRFSS={num_brfss}, NHANES={num_nhanes}, alpha={config.fl.dirichlet_alpha})")

    partitions = partition_by_source(
        train_df,
        num_brfss_clients=num_brfss,
        num_nhanes_clients=num_nhanes,
        alpha=config.fl.dirichlet_alpha,
        seed=config.seed,
    )

    # Step 3: Create per-client datasets
    feature_cols = HARMONIZED_FEATURES
    target_cols = TARGET_COLUMNS

    client_data = []
    for cid, indices in enumerate(partitions):
        indices = [i for i in indices if 0 <= i < n_train]

        if len(indices) == 0:
            logger.warning(f"Client {cid} has no data!")
            client_data.append((
                np.zeros((1, input_dim), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32)
            ))
            continue

        client_df = train_df.iloc[indices]
        X = client_df[feature_cols].values.astype(np.float32)
        Y = client_df[target_cols].values.astype(np.float32)

        X = np.nan_to_num(X, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)

        client_data.append((X, Y))

        # Log class distribution per client
        pos_counts = Y.sum(axis=0)
        logger.info(
            f"Client {cid}: {len(indices)} samples | "
            f"Pos: D={int(pos_counts[0])}, H={int(pos_counts[1])}, C={int(pos_counts[2])}"
        )

    # Test data
    X_test = test_df[feature_cols].values.astype(np.float32)
    Y_test = test_df[target_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    Y_test = np.nan_to_num(Y_test, nan=0.0)

    return client_data, (X_test, Y_test), input_dim


def run_fl_simulation(config: ExperimentConfig, use_synthetic: bool = False):
    """
    Run the full federated learning simulation.

    GPU RECOMMENDED for this step. Switch to high-end PC for:
    - DP-SGD training (3-5x slower than standard)
    - 50 rounds x 10 clients = 500 training iterations
    """
    logger.info("=" * 70)
    logger.info("FEDERATED LEARNING SIMULATION")
    logger.info(f"  Rounds: {config.fl.num_rounds}")
    logger.info(f"  Clients: {config.fl.num_clients}")
    logger.info(f"  DP: {'ON' if config.privacy.enable_dp else 'OFF'}")
    logger.info(f"  FedProx mu: {config.fl.fedprox_mu}")
    logger.info(f"  Compression: {'ON' if config.compression.enable_compression else 'OFF'}")
    logger.info(f"  Loss: {config.model.loss_type}")
    logger.info(f"  FedBN: {'ON' if config.fl.use_fedbn else 'OFF'}")
    logger.info(f"  Class-Aware Agg: {'ON' if config.fl.use_class_aware_aggregation else 'OFF'}")
    logger.info(f"  LR Schedule: {config.fl.lr_decay_strategy}")
    logger.info(f"  Device: {config.get_device()}")
    logger.info("=" * 70)

    # Prepare data
    client_data, test_data, input_dim = prepare_data_for_fl(config, use_synthetic)

    # Create global model with GroupNorm
    model = MultiTaskNCD(
        input_dim=input_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        norm_type=getattr(config.model, 'norm_type', 'group'),
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
            "learning_rate": config.fl.learning_rate,
            "loss_type": config.model.loss_type,
            "focal_gamma": config.model.focal_gamma,
            "fedbn": config.fl.use_fedbn,
            "class_aware_agg": config.fl.use_class_aware_aggregation,
            "lr_schedule": config.fl.lr_decay_strategy,
            "norm_type": getattr(config.model, 'norm_type', 'group'),
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

    import sys
    use_synthetic = "--synthetic" in sys.argv

    run_fl_simulation(config, use_synthetic=use_synthetic)
