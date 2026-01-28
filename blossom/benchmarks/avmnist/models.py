# Standard library imports
from typing import Dict

# Third-party imports
import torch
from torch import nn
from torchmultimodal.modules.fusions.attention_fusion import AttentionFusionModule
from torchmultimodal.modules.layers.mlp import MLP


# ============================================================================
# Image Encoder
# ============================================================================

class CNNImageEncoder(nn.Module):
    """CNN encoder for MNIST images (28x28 grayscale)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        conv_layers = []

        # First conv block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1, self.pool1]

        # Second conv block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2, self.pool2]

        # Third conv block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3, self.pool3]

        self.conv = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        # Calculate output size: 32 channels x 3x3 features = 288
        self.project = nn.Linear(32 * 3 * 3, embed_dim)
        nn.init.kaiming_normal_(self.project.weight, a=0.1)
        self.project.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        return self.project(x)


# ============================================================================
# Audio Encoder
# ============================================================================

class AudioEncoder(nn.Module):
    """CNN encoder for audio spectrograms."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        conv_layers = []

        # First conv block - expecting stereo mel spectrogram input (2 channels)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second conv block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third conv block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth conv block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.conv = nn.Sequential(*conv_layers)
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.project = nn.Linear(in_features=64, out_features=embed_dim)
        nn.init.kaiming_normal_(self.project.weight, a=0.1)
        self.project.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        return self.project(x)


# ============================================================================
# Factory Functions
# ============================================================================

def get_encoders_dict() -> Dict[str, nn.Module]:
    """Build a dictionary of modality encoders."""
    image_encoder = CNNImageEncoder(embed_dim=32)
    audio_encoder = AudioEncoder(embed_dim=256)
    return {
        "image": image_encoder,
        "audio": audio_encoder,
    }


def get_fusion_module() -> nn.Module:
    """Build the fusion module for combining modality embeddings.""" 
    return AttentionFusionModule(
        channel_to_encoder_dim={"image": 32, "audio": 256},
        encoding_projection_dim=128
    )


def get_head_module() -> nn.Module:
    """Build the classification head module."""
    return MLP(
        in_dim=128,
        out_dim=10,
        hidden_dims=[256, 128],
        activation=nn.ReLU,
        dropout=0.3,
        normalization=nn.BatchNorm1d,
    )