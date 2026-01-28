# Standard library imports
from typing import Dict, Type, Union

# Torch imports
import torch
import numpy as np
from datasets import Dataset as HFDataset

# Local imports
from blossom.dataloader import MultimodalDataset


# ============================================================================
# Dataset Class
# ============================================================================

class KUHARDataset(MultimodalDataset):
    """KU-HAR Human Activity Recognition dataset with accelerometer and gyroscope data."""

    def __init__(
        self,
        dataset: HFDataset,
        train_split: bool,
    ) -> None:
        super().__init__(dataset, train_split)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        item = self.dataset[idx]

        acc = np.array(item["acc"], dtype=np.float32)
        acc = torch.tensor(acc, dtype=torch.float32)

        gyro = np.array(item["gyro"], dtype=np.float32)
        gyro = torch.tensor(gyro, dtype=torch.float32)

        label = item["label"]

        return {
            "acc": acc,
            "gyro": gyro,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# Factory Function
# ============================================================================

def get_dataset_class() -> Type:
    """Return the KUHARDataset class."""
    return KUHARDataset
