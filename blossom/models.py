# Standard library imports
from typing import Dict
from copy import deepcopy

# Torch imports
import torch
from torch import nn

# Constants for layer prefixes
ENCODER_PREFIX: str = "encoders."
FUSION_PREFIX: str = "fusion_module."
HEAD_PREFIX: str = "head_module."


class LateFusion(nn.Module):
    """A generic architecture for late fusion multimodal models."""

    def __init__(
        self,
        encoders: nn.ModuleDict,
        fusion_module: nn.Module,
        head_module: nn.Module,
    ) -> None:
        """Initialize the late fusion model."""
        super().__init__()
        # Sort encoders by key for consistency
        self.encoders = nn.ModuleDict({k: encoders[k] for k in sorted(encoders.keys())})
        self.fusion_module = fusion_module
        self.head_module = head_module

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the late fusion model."""
        embeddings = {}
        for key, encoder in self.encoders.items():
            assert key in modalities, f"{key} missing in input"
            embeddings[key] = encoder(modalities[key])
        fused = self.fusion_module(embeddings)
        return self.head_module(fused)


def build_model(
    encoders_dict: Dict[str, nn.Module],
    fusion_module: nn.Module,
    head_module: nn.Module,
) -> nn.Module:
    """Build a late fusion multimodal model."""
    # Deepcopy each encoder into a ModuleDict to avoid shared state
    encoders_copy = nn.ModuleDict({k: deepcopy(v) for k, v in encoders_dict.items()})
    fusion_copy = deepcopy(fusion_module)
    head_copy = deepcopy(head_module)

    model = LateFusion(
        encoders=encoders_copy,
        fusion_module=fusion_copy,
        head_module=head_copy,
    )

    return model