# Standard library imports
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, Type

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset

# Flower imports
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner

# Global cache for FederatedDataset instance
fds: Optional[FederatedDataset] = None


class MultimodalDataset(Dataset, ABC):
    """
    Abstract base class for multimodal datasets.

    This class provides a standard interface for multimodal datasets that can be
    used with PyTorch DataLoaders in federated learning scenarios.
    """

    def __init__(self, dataset: HFDataset, train_split: bool) -> None:
        """
        Initialize the multimodal dataset.

        Args:
            dataset: Hugging Face dataset instance
            train_split: Whether this is a training split
        """
        self.dataset = dataset
        self.train_split = train_split

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing the sample data
        """
        pass

    @property
    def is_train_split(self) -> bool:
        """Return whether this dataset is a training split."""
        return self.train_split


def load_data(
    partition_id: int,
    dataset_name: str,
    dataset_class: Type[MultimodalDataset],
    partitioner: Partitioner,
    test_split_ratio: float,
    batch_size: int,
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    random_seed: int = 42,
    train_dataloader_kwargs: Dict[str, Any] = {},
    test_dataloader_kwargs: Dict[str, Any] = {},
) -> Tuple[DataLoader, DataLoader]:
    """Load and partition federated data for a specific client."""
    if not 0.0 <= test_split_ratio <= 1.0:
        raise ValueError("test_split_ratio must be between 0.0 and 1.0")

    # Initialize FederatedDataset singleton
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    # Load the client's data partition
    client_partition = fds.load_partition(partition_id)

    # Calculate split sizes
    total_samples = len(client_partition)
    test_samples = int(test_split_ratio * total_samples)
    train_samples = total_samples - test_samples

    # Create deterministic indices split
    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(total_samples, generator=generator).tolist()
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]

    # Create separate HF dataset subsets
    train_subset = client_partition.select(train_indices)
    test_subset = client_partition.select(test_indices)

    # Create separate dataset instances with different training flags
    train_multimodal_dataset = dataset_class(train_subset, train_split=True)
    test_multimodal_dataset = dataset_class(test_subset, train_split=False)

    # Create data loaders
    train_dataloader = DataLoader(
        train_multimodal_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        **train_dataloader_kwargs,
    )
    test_dataloader = DataLoader(
        test_multimodal_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        **test_dataloader_kwargs,
    )

    return train_dataloader, test_dataloader