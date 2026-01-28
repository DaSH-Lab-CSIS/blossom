# Standard library imports
import json
import warnings
from ast import literal_eval
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple, Type, Union

# Torch imports
import numpy as np
import torch
from torch.utils.data import DataLoader

# Flower imports
from flwr.client import Client, NumPyClient
from flwr.common import Context
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

# Local imports
from blossom.dataloader import MultimodalDataset, load_data
from blossom.models import build_model
from blossom.modality import ModalityManager
from blossom.task import Task, get_parameters, set_parameters, test, train

# Suppress warnings from flwr_datasets
warnings.filterwarnings("ignore", category=UserWarning, module="flwr_datasets")

# Global store for client-specific private head and fusion parameters
CLIENT_PRIVATE_HEAD_STORE: Dict[str, OrderedDict[str, torch.Tensor]] = {}
CLIENT_PRIVATE_FUSION_STORE: Dict[str, OrderedDict[str, torch.Tensor]] = {}

# Global store for client-specific scheduler states
CLIENT_SCHEDULER_STATE_STORE: Dict[str, Dict[str, Any]] = {}


class MultimodalFlowerClient(NumPyClient):
    """Flower client for multimodal federated learning."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        trainloader: DataLoader,
        valloader: DataLoader,
        local_epochs: int,
        device: torch.device,
    ) -> None:
        """Initialize the multimodal Flower client."""
        self.model = model
        self.task = task
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = device
        self.model.to(self.device)

    def _deserialize_modality_dict(self, json_str: str) -> Dict[Tuple[str, ...], int]:
        """Convert JSON string back to modality_dict with tuple keys."""
        str_dict = json.loads(json_str)
        return {literal_eval(k): v for k, v in str_dict.items()}

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Union[float, int, str]]]:
        """Train the model on local data."""
        # Extract configuration
        modality = config["modality"]
        client_cid = config["client_cid"]
        private_head = config["private_head"]
        private_fusion = config["private_fusion"]

        # Reconstruct modality manager
        modality_dict_json = config["modality_dict_json"]
        modality_dict = self._deserialize_modality_dict(modality_dict_json)
        modality_manager = ModalityManager(modality_dict)
        modalities = modality_manager.id_to_tuple(modality)
        all_modalities = modality_manager.get_all_modalities()

        # Initialize private head and fusion state if not done yet
        if client_cid not in CLIENT_PRIVATE_HEAD_STORE:
            CLIENT_PRIVATE_HEAD_STORE[client_cid] = None
        if client_cid not in CLIENT_PRIVATE_FUSION_STORE:
            CLIENT_PRIVATE_FUSION_STORE[client_cid] = None

        # Load global model parameters
        param_keys_json = config["param_keys_json"]
        param_keys = json.loads(param_keys_json)
        set_parameters(self.model, (parameters, param_keys))

        # Load private head and fusion parameters if they exist
        if private_head and CLIENT_PRIVATE_HEAD_STORE[client_cid]:
            head_state_to_load = OrderedDict()
            for k, v_tensor in CLIENT_PRIVATE_HEAD_STORE[client_cid].items():
                head_state_to_load[k] = v_tensor.to(self.device)
            self.model.head_module.load_state_dict(head_state_to_load)
        if private_fusion and CLIENT_PRIVATE_FUSION_STORE[client_cid]:
            fusion_state_to_load = OrderedDict()
            for k, v_tensor in CLIENT_PRIVATE_FUSION_STORE[client_cid].items():
                fusion_state_to_load[k] = v_tensor.to(self.device)
            self.model.fusion_module.load_state_dict(fusion_state_to_load)

        # Load scheduler state if it exists
        scheduler_state = CLIENT_SCHEDULER_STATE_STORE.get(client_cid, None)

        # Train model
        total_loss, total_samples, updated_scheduler_state = train(
            model=self.model,
            task=self.task,
            modalities=modalities,
            all_modalities=all_modalities,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
            scheduler_state=scheduler_state
        )

        # Save training states for next round
        CLIENT_SCHEDULER_STATE_STORE[client_cid] = updated_scheduler_state

        # Save updated private head and fusion parameters
        if private_head:
            current_head_state_cpu = OrderedDict()
            for k, v_tensor in self.model.head_module.state_dict().items():
                current_head_state_cpu[k] = v_tensor.cpu().clone()
            CLIENT_PRIVATE_HEAD_STORE[client_cid] = current_head_state_cpu
        if private_fusion:
            current_fusion_state_cpu = OrderedDict()
            for k, v_tensor in self.model.fusion_module.state_dict().items():
                current_fusion_state_cpu[k] = v_tensor.cpu().clone()
            CLIENT_PRIVATE_FUSION_STORE[client_cid] = current_fusion_state_cpu

        # Get updated parameters
        updated_parameters, updated_keys = get_parameters(self.model, modalities)

        # Prepare metrics
        client_name = f"CLIENT {client_cid} ({modality.upper()})"
        metrics = {
            "train_loss": float(total_loss / total_samples) if total_samples > 0 else 0.0,
            "client_name": str(client_name)
        }

        return updated_parameters, total_samples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Union[float, int, str]]]:
        """Evaluate the model on local data."""
        # Extract configuration
        modality = config["modality"]
        client_cid = config["client_cid"]
        private_head = config["private_head"]
        private_fusion = config["private_fusion"]

        # Reconstruct modality manager
        modality_dict_json = config["modality_dict_json"]
        modality_dict = self._deserialize_modality_dict(modality_dict_json)
        modality_manager = ModalityManager(modality_dict)
        modalities = modality_manager.id_to_tuple(modality)
        all_modalities = modality_manager.get_all_modalities()

        # Initialize private head and fusion state if not done yet
        if client_cid not in CLIENT_PRIVATE_HEAD_STORE:
            CLIENT_PRIVATE_HEAD_STORE[client_cid] = None
        if client_cid not in CLIENT_PRIVATE_FUSION_STORE:
            CLIENT_PRIVATE_FUSION_STORE[client_cid] = None

        # Load global model parameters
        param_keys_json = config["param_keys_json"]
        param_keys = json.loads(param_keys_json)
        set_parameters(self.model, (parameters, param_keys))

        # Load private head and fusion parameters if they exist
        if private_head and CLIENT_PRIVATE_HEAD_STORE[client_cid]:
            head_state_to_load = OrderedDict()
            for k, v_tensor in CLIENT_PRIVATE_HEAD_STORE[client_cid].items():
                head_state_to_load[k] = v_tensor.to(self.device)
            self.model.head_module.load_state_dict(head_state_to_load)
        if private_fusion and CLIENT_PRIVATE_FUSION_STORE[client_cid]:
            fusion_state_to_load = OrderedDict()
            for k, v_tensor in CLIENT_PRIVATE_FUSION_STORE[client_cid].items():
                fusion_state_to_load[k] = v_tensor.to(self.device)
            self.model.fusion_module.load_state_dict(fusion_state_to_load)

        # Evaluate model
        total_loss, total_samples, accumulated_metrics = test(
            model=self.model,
            task=self.task,
            modalities=modalities,
            all_modalities=all_modalities,
            valloader=self.valloader,
            device=self.device
        )

        # Prepare metrics
        metrics = self.task.compute_aggregated_metrics(accumulated_metrics, total_samples, total_loss)
        metrics["client_name"] = f"CLIENT {client_cid} ({modality.upper()})"

        return total_loss, total_samples, metrics


def build_client_fn(
    task: Task,
    dataset_name: str,
    dataset_class: Type[MultimodalDataset],
    test_split_ratio: float,
    batch_size: int,
    local_epochs: int,
    niid: bool,
    alpha: float,
    partition_by: str,
    encoders_dict: Dict[str, torch.nn.Module],
    fusion_module: torch.nn.Module,
    head_module: torch.nn.Module,
    device: str,
    train_dataloader_kwargs: Dict[str, Any] = {},
    test_dataloader_kwargs: Dict[str, Any] = {},
) -> Callable:
    """Build a Flower client function."""
    
    def client_fn(context: Context) -> Client:
        """
        Create a Flower client instance.
        """
        # Get partition configuration
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]

        # Create partitioner
        if not niid:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions, alpha=alpha, partition_by=partition_by
            )

        # Load data
        trainloader, valloader = load_data(
            partition_id=partition_id,
            dataset_name=dataset_name,
            dataset_class=dataset_class,
            partitioner=partitioner,
            test_split_ratio=test_split_ratio,
            batch_size=batch_size,
            train_dataloader_kwargs=train_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
        )

        # Build model
        model = build_model(
            encoders_dict=encoders_dict,
            fusion_module=fusion_module,
            head_module=head_module,
        )

        # Create client
        client = MultimodalFlowerClient(
            task=task,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            local_epochs=local_epochs,
            device=torch.device(device),
        )

        return client.to_client()

    return client_fn