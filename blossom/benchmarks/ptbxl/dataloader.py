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

class PTBXLDataset(MultimodalDataset):
    """PTB-XL ECG dataset with I-AVF and V1-V6 lead signals."""

    def __init__(
        self,
        dataset: HFDataset,
        train_split: bool,
    ) -> None:
        super().__init__(dataset, train_split)
        # Fix PyArrow compatibility issue with datasets library
        self.dataset.set_format(type='numpy')

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        item = self.dataset[idx]

        # Process I-AVF lead data - shape (1000, 6)
        # Leads: I, II, III, AVR, AVL, AVF
        i_to_avf = np.array(item["i_to_avf"], dtype=np.float32)
        i_to_avf = torch.tensor(i_to_avf, dtype=torch.float32)

        # Process V1-V6 lead data - shape (1000, 6)
        # Leads: V1, V2, V3, V4, V5, V6
        v1_to_v6 = np.array(item["v1_to_v6"], dtype=np.float32)
        v1_to_v6 = torch.tensor(v1_to_v6, dtype=torch.float32)

        label = int(item["label"])

        return {
            "iToAvf": i_to_avf,
            "v1ToV6": v1_to_v6,
            "label": label,
        }


# ============================================================================
# Factory Function
# ============================================================================

def get_dataset_class() -> Type:
    """Return the PTBXLDataset class."""
    return PTBXLDataset
