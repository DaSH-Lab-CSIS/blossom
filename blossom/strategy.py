# Standard library imports
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

# PyData imports
import numpy as np

# Flower imports
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg

# Local imports
from blossom.logger import LoggingManager
from blossom.modality import ModalityManager
from blossom.task import should_select_parameter


class MultiFL(FedAvg):
    """Federated learning strategy with multimodal support and partial aggregation."""

    def __init__(
        self,
        num_rounds: int,
        modality_dict: Dict[Tuple[str, ...], int],
        parameter_keys: List[str],
        private_head: bool,
        private_fusion: bool,
        results_dir: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the MultiFL strategy."""
        super().__init__(*args, **kwargs)
        self._current_parameters = None
        self._parameter_keys = parameter_keys
        self.num_rounds = num_rounds
        self.private_head = private_head
        self.private_fusion = private_fusion
        self.modality_manager = ModalityManager(modality_dict)
        self.logging_manager = LoggingManager(results_dir)

    def _initialize_modalities(self, client_manager: ClientManager) -> None:
        """Initialize modality assignments for all clients."""
        all_clients = list(client_manager.all().values())
        sorted_clients = sorted(all_clients, key=lambda c: c.cid)
        for client in sorted_clients:
            self.modality_manager.set_modality(client.cid)
        self.modality_manager.print_client_modality_mapping()

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = super().initialize_parameters(client_manager)
        if initial_parameters:
            self._current_parameters = initial_parameters
        logging.info("Using initial parameters provided by base strategy or client")
        return initial_parameters

    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training round."""
        if not self.modality_manager.all_assigned():
            self._initialize_modalities(client_manager)
        
        self._current_parameters = parameters

        # Prepare base configuration
        base_config = {}
        if self.on_fit_config_fn is not None:
            base_config = self.on_fit_config_fn(server_round)

        base_config["private_head"] = self.private_head
        base_config["private_fusion"] = self.private_fusion
        base_config["server_round"] = server_round
        base_config["modality_dict_json"] = json.dumps(
            {str(k): v for k, v in self.modality_manager.get_modality_dict().items()}
        )

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # Prepare parameters
        full_ndarrays = parameters_to_ndarrays(parameters)
        full_named_ndarrays = OrderedDict(zip(self._parameter_keys, full_ndarrays))

        # Configure each client
        fit_ins_list = []
        logging.info(f"Round {server_round}: Configuring clients for training")
        
        for client in clients:
            modality_id = self.modality_manager.get_modality(client.cid)
            modality_tuple = self.modality_manager.id_to_tuple(modality_id)
            
            # Prepare client-specific config
            client_config = base_config.copy()
            client_config["modality"] = modality_id
            client_config["client_cid"] = client.cid

            # Filter parameters for this client's modalities
            client_named_ndarrays = OrderedDict()
            for name, param in full_named_ndarrays.items():
                if should_select_parameter(name, modality_tuple):
                    client_named_ndarrays[name] = param

            param_keys = list(client_named_ndarrays.keys())
            client_ndarrays = list(client_named_ndarrays.values())
            client_parameters = ndarrays_to_parameters(client_ndarrays)
            client_config["param_keys_json"] = json.dumps(param_keys)

            fit_ins = FitIns(client_parameters, client_config)
            fit_ins_list.append((client, fit_ins))

        logging.info(f"Configured {len(fit_ins_list)} clients for round {server_round} training")
        return fit_ins_list

    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation round."""
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        num_available = client_manager.num_available()
        if num_available < self.min_evaluate_clients:
            logging.warning("Not enough clients available for evaluation, skipping round")
            return []

        # Sample clients
        sample_size = int(self.fraction_evaluate * num_available)
        sample_size = max(sample_size, self.min_evaluate_clients)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )

        # Prepare base config
        base_config = config.copy()
        base_config["private_head"] = self.private_head
        base_config["private_fusion"] = self.private_fusion
        base_config["param_keys_json"] = json.dumps(self._parameter_keys)
        base_config["modality_dict_json"] = json.dumps(
            {str(k): v for k, v in self.modality_manager.get_modality_dict().items()}
        )

        # Configure each client
        evaluate_ins_list = []
        for client in clients:
            client_config = base_config.copy()
            modality = self.modality_manager.get_modality(client.cid)
            client_config["modality"] = modality
            client_config["client_cid"] = client.cid

            evaluate_ins = EvaluateIns(parameters, client_config)
            evaluate_ins_list.append((client, evaluate_ins))

        logging.info(f"Configured {len(evaluate_ins_list)} clients for evaluation round {server_round}")
        return evaluate_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model parameters from training results."""
        if not results:
            logging.warning("aggregate_fit: no results received")
            return self._current_parameters, {}

        if not self.accept_failures and failures:
            logging.warning("aggregate_fit: failures received and accept_failures is False")
            return self._current_parameters, {}

        # Filter successful results
        successful_results = [
            (client, res) for client, res in results 
            if res.status.code == Code.OK
        ]
        
        if not successful_results:
            logging.warning("aggregate_fit: no successful results")
            return self._current_parameters, {}

        # Prepare global parameters structure
        current_global_ndarrays = parameters_to_ndarrays(self._current_parameters)
        global_named_parameters = OrderedDict(zip(self._parameter_keys, current_global_ndarrays))

        # Collect per-layer results from clients who actually trained each layer
        per_layer_results: Dict[str, List[Tuple[np.ndarray, int]]] = {
            key: [] for key in self._parameter_keys
        }

        for client, fit_res in successful_results:
            client_ndarrays_partial = parameters_to_ndarrays(fit_res.parameters)
            modality = self.modality_manager.get_modality(client.cid)
            modality_tuple = self.modality_manager.id_to_tuple(modality)

            # Get client's parameter keys (only layers this client trained)
            client_keys = [
                name for name in self._parameter_keys
                if should_select_parameter(name, modality_tuple)
            ]
            client_updates = dict(zip(client_keys, client_ndarrays_partial))

            # Only add to layers this client actually trained
            for key in client_keys:
                per_layer_results[key].append(
                    (client_updates[key], fit_res.num_examples)
                )

        # Aggregate each layer separately using only contributing clients
        aggregated_ndarrays = []
        for key in self._parameter_keys:
            if per_layer_results[key]:
                # Weighted average from clients who trained this layer
                layer_updates = per_layer_results[key]
                total_examples = sum(n for _, n in layer_updates)
                weighted_sum = sum(
                    param * (n / total_examples) for param, n in layer_updates
                )
                aggregated_ndarrays.append(weighted_sum)
            else:
                # No client trained this layer, keep global parameters
                aggregated_ndarrays.append(global_named_parameters[key])

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate metrics
        fit_metrics = [
            (res.num_examples, res.metrics)
            for _, res in successful_results
            if res.metrics
        ]
        metrics_aggregated = {}
        if fit_metrics:
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        logging.info(f"aggregate_fit: aggregation complete for round {server_round}")
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and track metrics."""
        if not results:
            logging.warning("aggregate_evaluate: no results received")
            return None, {}

        if not self.accept_failures and failures:
            logging.warning("aggregate_evaluate: failures received and accept_failures is False")
            return None, {}

        # Filter successful results
        successful_results = [
            (client, res) for client, res in results 
            if res.status.code == Code.OK
        ]
        
        if not successful_results:
            logging.warning("aggregate_evaluate: no successful results")
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg([
            (res.num_examples, res.loss)
            for _, res in successful_results
            if res.loss is not None
        ])

        # Aggregate metrics
        eval_metrics = [
            (res.num_examples, res.metrics)
            for _, res in successful_results
            if res.metrics
        ]
        metrics_aggregated = {}
        if eval_metrics:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        self.logging_manager.log_aggregated_metrics(server_round, metrics_aggregated)

        # Track client-wise metrics
        client_metrics = {}
        for client, res in successful_results:
            if not res.metrics:
                continue

            client_name = res.metrics.get("client_name")
            if not client_name:
                continue

            client_metrics[client_name] = dict(res.metrics)
            client_metrics[client_name]["round"] = server_round

            self.logging_manager.log_client_metrics(client_name, server_round, res.metrics)

        # Add client metrics to aggregated metrics
        metrics_aggregated["client_metrics"] = client_metrics

        # Save results if final round
        if server_round == self.num_rounds:
            self.logging_manager.save_results()

        return loss_aggregated, metrics_aggregated