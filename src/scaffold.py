"""
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

Implements variance reduction via control variates to correct client drift
in heterogeneous (non-IID) federated settings.

Key idea: Each client maintains a control variate c_i that estimates
the direction of its local gradient drift from the global average.
The corrected gradient: g_corrected = g_local + (c_global - c_i)

This significantly reduces gradient variance across clients and improves
convergence, especially on non-IID data where FedAvg/FedProx struggle.

Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
for Federated Learning", ICML 2020.

Novel extension (SCAFFOLD-MT): Maintain separate control variates per
task head to handle heterogeneous task distributions across clients.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ppfl-ncd.scaffold")


class SCAFFOLDCorrector:
    """
    Client-side SCAFFOLD variance reduction.

    Maintains control variate c_i for a single client and applies
    correction to local gradients during training.
    """

    def __init__(
        self,
        param_shapes: Optional[List[Tuple[int, ...]]] = None,
        learning_rate: float = 0.0003,
    ):
        """
        Args:
            param_shapes: List of shapes for each model parameter.
                          If None, will be initialized on first use.
            learning_rate: Client learning rate (for c_i update)
        """
        self.learning_rate = learning_rate

        # Client control variate: c_i
        self.c_local: Optional[List[torch.Tensor]] = None
        # Server control variate: c (received from server)
        self.c_global: Optional[List[torch.Tensor]] = None

        if param_shapes is not None:
            self._init_variates(param_shapes)

    def _init_variates(self, param_shapes: List[Tuple[int, ...]]):
        """Initialize control variates to zero."""
        self.c_local = [torch.zeros(s) for s in param_shapes]
        self.c_global = [torch.zeros(s) for s in param_shapes]

    def set_server_variate(self, c_global: List[torch.Tensor]):
        """Set the server control variate (received from server each round)."""
        self.c_global = [c.detach().clone() for c in c_global]
        if self.c_local is None:
            self.c_local = [torch.zeros_like(c) for c in c_global]

    def correct_gradients(self, model: torch.nn.Module):
        """
        Apply SCAFFOLD correction to model gradients IN-PLACE.

        Call this AFTER loss.backward() and BEFORE optimizer.step().

        Corrected gradient: g + c_global - c_local
        """
        if self.c_local is None or self.c_global is None:
            return

        for param, c_g, c_l in zip(
            model.parameters(), self.c_global, self.c_local
        ):
            if param.grad is not None:
                device = param.grad.device
                correction = c_g.to(device) - c_l.to(device)
                param.grad.data.add_(correction)

    def update_local_variate(
        self,
        model_before: List[torch.Tensor],
        model_after: List[torch.Tensor],
        num_steps: int,
    ) -> List[torch.Tensor]:
        """
        Update local control variate after training.

        c_i_new = c_i - c_global + (1 / (K * eta)) * (x_global - x_local)

        Where:
            K = number of local steps
            eta = learning rate
            x_global = model before local training
            x_local = model after local training

        Returns:
            Delta control variate (c_i_new - c_i_old) for server update
        """
        if self.c_local is None or self.c_global is None:
            return []

        c_delta = []
        new_c_local = []
        scale = 1.0 / (max(num_steps, 1) * self.learning_rate)

        for c_l, c_g, w_before, w_after in zip(
            self.c_local, self.c_global, model_before, model_after
        ):
            # c_i_new = c_i - c_global + scale * (x_global - x_local)
            c_new = c_l - c_g + scale * (w_before - w_after)
            delta = c_new - c_l
            c_delta.append(delta)
            new_c_local.append(c_new)

        self.c_local = new_c_local
        return c_delta


class SCAFFOLDServer:
    """
    Server-side SCAFFOLD coordination.

    Maintains global control variate c and updates it based on
    client-reported control variate deltas.
    """

    def __init__(self, param_shapes: Optional[List[Tuple[int, ...]]] = None):
        # Global control variate: c
        self.c_global: Optional[List[torch.Tensor]] = None
        self.num_clients_total: int = 0

        if param_shapes is not None:
            self.c_global = [torch.zeros(s) for s in param_shapes]

    def initialize(self, model: torch.nn.Module, num_clients: int):
        """Initialize from model parameters."""
        self.c_global = [
            torch.zeros_like(p.data) for p in model.parameters()
        ]
        self.num_clients_total = num_clients

    def get_global_variate(self) -> List[torch.Tensor]:
        """Get global control variate to send to clients."""
        if self.c_global is None:
            raise RuntimeError("SCAFFOLD server not initialized")
        return [c.clone() for c in self.c_global]

    def get_variate_as_numpy(self) -> List[np.ndarray]:
        """Get global control variate as numpy arrays for Flower serialization."""
        return [c.numpy() for c in self.c_global]

    def set_variate_from_numpy(self, arrays: List[np.ndarray]):
        """Set global control variate from numpy arrays."""
        self.c_global = [torch.from_numpy(a).float() for a in arrays]

    def update_global_variate(
        self,
        client_c_deltas: List[List[torch.Tensor]],
    ):
        """
        Update global control variate from client delta reports.

        c_new = c + (1/N) * sum(delta_c_i)  for participating clients

        Args:
            client_c_deltas: List of delta control variates from each client
        """
        if self.c_global is None or not client_c_deltas:
            return

        n_participating = len(client_c_deltas)
        scale = 1.0 / max(self.num_clients_total, 1)

        for param_idx in range(len(self.c_global)):
            delta_sum = torch.zeros_like(self.c_global[param_idx])
            for client_delta in client_c_deltas:
                if param_idx < len(client_delta):
                    delta_sum += client_delta[param_idx]
            self.c_global[param_idx] += scale * delta_sum


def serialize_control_variates(variates: List[torch.Tensor]) -> List[np.ndarray]:
    """Serialize control variates for Flower parameter transport."""
    return [v.cpu().numpy() for v in variates]


def deserialize_control_variates(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    """Deserialize control variates from Flower parameter transport."""
    return [torch.from_numpy(a).float() for a in arrays]
