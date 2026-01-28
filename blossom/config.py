# Standard library imports
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ModelComponentTarget:
    """Model component instantiation target."""
    _target_: str


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    huggingface: str
    task: ModelComponentTarget
    dataset_class: ModelComponentTarget
    encoders_dict: ModelComponentTarget
    fusion_module: ModelComponentTarget
    head_module: ModelComponentTarget
    partition_by: ModelComponentTarget


@dataclass
class PartitionerConfig:
    """Data partitioning configuration."""
    name: str
    niid: bool
    alpha: float


@dataclass
class AggregationConfig:
    """Aggregation method configuration."""
    name: str
    private_head: bool
    private_fusion: bool


@dataclass
class ExperimentConfig:
    """Experiment settings for federated learning."""
    num_runs: int
    clients: Dict[str, int]
    num_rounds: int
    local_epochs: int
    test_split_ratio: float
    batch_size: int
    num_cpus_per_client: int
    num_gpus_per_client: float
    device: str
    verbose_logging: bool


@dataclass
class ClientConfigTarget:
    """Client configuration target for parsing."""
    _target_: str
    src_dict: Dict[str, int]


@dataclass
class ServerFnConfig:
    """Server function configuration."""
    _target_: str
    num_rounds: int
    private_head: bool
    private_fusion: bool
    fraction_fit: float
    fraction_evaluate: float
    min_fit_clients: int
    min_evaluate_clients: int
    min_avail_clients: int
    client_config: ClientConfigTarget
    encoders_dict: ModelComponentTarget
    fusion_module: ModelComponentTarget
    head_module: ModelComponentTarget


@dataclass
class ClientFnConfig:
    """Client function configuration."""
    _target_: str
    task: ModelComponentTarget
    dataset_name: str
    dataset_class: ModelComponentTarget
    test_split_ratio: float
    batch_size: int
    local_epochs: int
    niid: bool
    alpha: float
    partition_by: ModelComponentTarget
    encoders_dict: ModelComponentTarget
    fusion_module: ModelComponentTarget
    head_module: ModelComponentTarget
    device: str


@dataclass
class LoggingConfig:
    """Logging and output directory settings."""
    results_dir: str


@dataclass
class Config:
    """Main configuration class that combines all config sections."""
    defaults: List[Any]
    experiment: ExperimentConfig
    server_fn: ServerFnConfig
    client_fn: ClientFnConfig
    logging: LoggingConfig
    dataset: DatasetConfig
    partitioner: PartitionerConfig
    aggregation: AggregationConfig