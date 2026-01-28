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
    """1D Convolutional encoder for time-series data.
    
    Copied from fed-multimodal/fed_multimodal/model/mm_models.py
    """
    
    def __init__(
        self,
        input_dim: int,
        n_filters: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        # Conv modules
        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the convolutional encoder.
        
        Args:
            x: shape [batch_size (B), num_data (T), feature_dim (D)]
        Returns:
            shape [B, T//8, n_filters*4]
        """
        x = x.float()
        x = x.permute(0, 2, 1)  # [B, D, T]
        
        # conv
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
# Audio Encoder 
# ============================================================================

class AudioEncoder(nn.Module):
    """Encoder for audio fbank features using Conv1D + GRU.
    
    Based on audio processing in fed-multimodal SERClassifier.
    """

    def __init__(
        self,
        input_dim: int = 80,    # fbank features
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
        """Forward pass through audio encoder.
        
        Args:
            x: (B, T, 80) fbank features
        Returns:
            (B, d_hid) audio embeddings
        """
        x = self.conv(x)
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        
        return x


# ============================================================================
# Text Encoder 
# ============================================================================

class TextEncoder(nn.Module):
    """Encoder for MobileBERT text features using GRU.
    
    Based on text processing in fed-multimodal SERClassifier.
    Input: Pre-extracted MobileBERT features (512-dim)
    """

    def __init__(
        self,
        input_dim: int = 512,   # MobileBERT hidden size
        d_hid: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=d_hid,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.d_hid = d_hid

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through text encoder.
        
        Args:
            x: (B, seq_len, 512) MobileBERT features
        Returns:
            (B, d_hid) text embeddings
        """
        x, _ = self.rnn(x)
        x = torch.mean(x, dim=1)
        
        return x


# ============================================================================
# Factory Functions
# ============================================================================

# Model parameters 
AUDIO_INPUT_DIM = 80   # fbank features
TEXT_INPUT_DIM = 512   # MobileBERT hidden size
N_FILTERS = 32
D_HID = 64
DROPOUT = 0.1
NUM_CLASSES = 4  # anger, joy, neutral, sadness


def get_encoders_dict() -> Dict[str, nn.Module]:
    """Build a dictionary of modality encoders."""
    audio_encoder = AudioEncoder(
        input_dim=AUDIO_INPUT_DIM,
        n_filters=N_FILTERS,
        d_hid=D_HID,
        dropout=DROPOUT
    )
    text_encoder = TextEncoder(
        input_dim=TEXT_INPUT_DIM,
        d_hid=D_HID,
        dropout=DROPOUT
    )
    return {
        "audio": audio_encoder,
        "text": text_encoder,
    }


def get_fusion_module() -> nn.Module:
    """Build the fusion module for combining modality embeddings."""
    return AttentionFusionModule(
        channel_to_encoder_dim={"audio": D_HID, "text": D_HID},
        encoding_projection_dim=D_HID * 2
    )


def get_head_module() -> nn.Module:
    """Build the classification head module."""
    return MLP(
        in_dim=D_HID * 2,  
        out_dim=NUM_CLASSES,
        hidden_dims=[64],
        activation=nn.ReLU,
        dropout=DROPOUT,
    )
