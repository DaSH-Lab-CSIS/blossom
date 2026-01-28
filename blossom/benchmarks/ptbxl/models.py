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
    """1D Convolutional encoder for time-series ECG data."""
    
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
# I-AVF Lead Encoder
# ============================================================================

class ItoAVFEncoder(nn.Module):
    """Encoder for I-AVF leads using Conv1D + GRU."""

    def __init__(
        self,
        input_dim: int = 6,      # I, II, III, AVR, AVL, AVF
        n_filters: int = 32,
        d_hid: int = 64,
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
        """Forward pass through I-AVF encoder."""
        x = self.conv(x)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        return x


# ============================================================================
# V1-V6 Lead Encoder
# ============================================================================

class V1toV6Encoder(nn.Module):
    """Encoder for V1-V6 leads using Conv1D + GRU."""

    def __init__(
        self,
        input_dim: int = 6,      # V1, V2, V3, V4, V5, V6
        n_filters: int = 32,
        d_hid: int = 64,
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
        """Forward pass through V1-V6 encoder."""
        x = self.conv(x)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        return x


# ============================================================================
# Factory Functions
# ============================================================================

def get_encoders_dict() -> Dict[str, nn.Module]:
    """Build a dictionary of modality encoders."""
    i_to_avf_encoder = ItoAVFEncoder(
        input_dim=6,      
        n_filters=32,     
        d_hid=64,         
        dropout=0.1
    )
    v1_to_v6_encoder = V1toV6Encoder(
        input_dim=6,      
        n_filters=32,     
        d_hid=64,         
        dropout=0.1
    )
    return {
        "iToAvf": i_to_avf_encoder,
        "v1ToV6": v1_to_v6_encoder,
    }


def get_fusion_module() -> nn.Module:
    """Build the fusion module for combining modality embeddings."""
    return AttentionFusionModule(
        channel_to_encoder_dim={"iToAvf": 64, "v1ToV6": 64},
        encoding_projection_dim=128
    )


def get_head_module() -> nn.Module:
    """Build the classification head module."""
    return MLP(
        in_dim=128,       
        out_dim=5,       
        hidden_dims=[64],
        activation=nn.ReLU,
        dropout=0.1,
    )
