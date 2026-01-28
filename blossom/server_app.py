# Standard library imports
from typing import Callable, Dict, List, Tuple

# Torch imports
from torch import nn

# Flower imports
from flwr.common import Context, Scalar, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig

# Local imports
from blossom.models import build_model
from blossom.strategy import MultiFL


def fit_metrics_aggregation_fn(
    metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Aggregate training metrics across clients."""
    aggregated_metrics = {}
    total_examples = 0

    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            if key == "client_name":
                continue
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += value * num_examples
        total_examples += num_examples
    
    for key in aggregated_metrics:
        aggregated_metrics[key] /= total_examples
    
    return aggregated_metrics


def evaluate_metrics_aggregation_fn(
    metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics across clients."""
    aggregated_metrics = {}
    total_examples = 0

    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            if key == "client_name":
                continue
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += value * num_examples
        total_examples += num_examples
    
    for key in aggregated_metrics:
        aggregated_metrics[key] /= total_examples
    
    return aggregated_metrics


def build_server_fn(
    num_rounds: int,
    private_head: bool,
    private_fusion: bool,
    client_config: Dict[Tuple[str, ...], int],
    fraction_fit: float,
    fraction_evaluate: float,
    min_fit_clients: int,
    min_evaluate_clients: int,
    min_avail_clients: int,
    encoders_dict: Dict[str, nn.Module],
    fusion_module: nn.Module,
    head_module: nn.Module,
    results_dir: str,
) -> Callable:
    """Build a Flower server function."""

    def server_fn(context: Context) -> ServerAppComponents:
        """Create a Flower server instance."""
        # Build model and extract parameters
        model = build_model(
            encoders_dict=encoders_dict,
            fusion_module=fusion_module,
            head_module=head_module,
        )
        parameter_keys = list(model.state_dict().keys())
        initial_ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
        parameters = ndarrays_to_parameters(initial_ndarrays)

        # Build MultiFL strategy
        strategy = MultiFL(
            num_rounds=num_rounds,
            modality_dict=client_config,
            parameter_keys=parameter_keys,
            private_head=private_head,
            private_fusion=private_fusion,
            results_dir=results_dir,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_avail_clients,
            initial_parameters=parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        # Create server configuration
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn