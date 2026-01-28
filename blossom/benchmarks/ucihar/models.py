# Standard library imports
from typing import Dict

# Torch imports
import torch
from torch import nn, Tensor
from torchmultimodal.modules.fusions.attention_fusion import AttentionFusionModule
from torchmultimodal.modules.layers.mlp import MLP


# ============================================================================
# Conv1D Encoder
# ============================================================================

class Conv1dEncoder(nn.Module):
    """1D Convolutional encoder for time-series sensor data."""
    
    def __init__(
        self,
        input_dim: int,
        n_filters: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the convolutional encoder."""
        x = x.float()
        x = x.permute(0, 2, 1)  # [B, D, T]
        
        # conv1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        
        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)  # [B, T//8, n_filters*4]
        return x


# ============================================================================
# Accelerometer Encoder
# ============================================================================

class AccelerometerEncoder(nn.Module):
    """Encoder for accelerometer data using Conv1D + GRU."""

    def __init__(
        self,
        input_dim: int = 3,
        n_filters: int = 32,
        d_hid: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = Conv1dEncoder(
            input_dim=input_dim,
            n_filters=n_filters,
            dropout=dropout
        )
        
        self.rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.d_hid = d_hid

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through accelerometer encoder."""
        x = self.conv(x)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        return x


# ============================================================================
# Gyroscope Encoder
# ============================================================================

class GyroscopeEncoder(nn.Module):
    """Encoder for gyroscope data using Conv1D + GRU."""

    def __init__(
        self,
        input_dim: int = 3,
        n_filters: int = 32,
        d_hid: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = Conv1dEncoder(
            input_dim=input_dim,
            n_filters=n_filters,
            dropout=dropout
        )
        
        self.rnn = nn.GRU(
            input_size=n_filters * 4,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.d_hid = d_hid

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through gyroscope encoder."""
        x = self.conv(x)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        return x


# ============================================================================
# Factory Functions
# ============================================================================

def get_encoders_dict() -> Dict[str, nn.Module]:
    """Build a dictionary of modality encoders."""
    acc_encoder = AccelerometerEncoder(
        input_dim=3,      
        n_filters=32,     
        d_hid=128,        
        dropout=0.1
    )
    gyro_encoder = GyroscopeEncoder(
        input_dim=3,      
        n_filters=32,     
        d_hid=128,        
        dropout=0.1
    )
    return {
        "acc": acc_encoder,
        "gyro": gyro_encoder,
    }


def get_fusion_module() -> nn.Module:
    """Build the fusion module for combining modality embeddings."""
    return AttentionFusionModule(
        channel_to_encoder_dim={"acc": 128, "gyro": 128},
        encoding_projection_dim=256
    )


def get_head_module() -> nn.Module:
    """Build the classification head module."""
    return MLP(
        in_dim=256,
        out_dim=6,  
        hidden_dims=[64],
        activation=nn.ReLU,
        dropout=0.1,
    )
